"""以图搜图插件：Google Lens + SauceNAO + trace.moe，双通道（LLM 工具 / 命令直达）。"""

import asyncio
import base64
import contextlib
import html
import importlib.util
import io
import json
import re
import time
from pathlib import Path

import httpx

from core.plugin import BasePlugin, logger, on, Priority, register
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat import MessageChain
from core.chat.message_elements import Text, Image, Reply
from core.provider import LLMRequest
from core.utils.path_utils import get_data_path

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_RECENT_TTL = 600  # 最近图片缓存 10 分钟
_RECENT_MAX = 5
_SAUCENAO_MIN_INTERVAL = 8  # 免费 key 限速
_IQDB_MIN_INTERVAL = 5
_BAIDU_MIN_INTERVAL = 3
_YANDEX_MIN_INTERVAL = 3  # Yandex 免 key 限速
_CACHE_MAX = 500


class _QwenResponsesHttpError(RuntimeError):
    """Responses API 非 2xx 错误：带 status/message，便于上层按错误内容降级重试。"""

    def __init__(self, status: int, message: str):
        super().__init__(f"qwen responses {status}: {message}")
        self.status = status
        self.message = message


class ImageSearchPlugin(BasePlugin):
    # 老版本配置的安全迁移表：只补缺失键，绝不覆盖用户已保存的值。
    _MIGRATION_DEFAULTS = {
        "section_deepseek": {
            "enable_deepseek": True,
            "deepseek_key": "",
            "deepseek_model": "deepseek-flash",
            "deepseek_base_url": "https://api.deepseek.com",
            "deepseek_extra_headers": {},
            "deepseek_extra_body": {},
            "deepseek_timeout": 60,
            "deepseek_in_tool": True,
            "deepseek_require_search": False,
        },
        "section_anime": {"iqdb_in_tool": False, "saucenao_merge_versions": True},
    }

    def _deepseek_chat_url(self) -> str:
        """把用户填的 Base URL 归一化成 /chat/completions 完整地址。

        兼容这些写法（大小写不敏感、允许尾斜杠）：
          https://api.deepseek.com
          https://api.deepseek.com/
          https://opencode.ai/zen/go/v1
          https://opencode.ai/zen/go/v1/
          https://opencode.ai/zen/go/v1/chat/completions
          https://opencode.ai/zen/go/v1/chat/completions/
        已带 /chat/completions 的原样使用，不会重复拼接。
        """
        base = str(getattr(self, "deepseek_base_url", "") or "").strip().rstrip("/")
        if not base:
            base = "https://api.deepseek.com"
        low = base.lower()
        if low.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @staticmethod
    def _as_dict(value) -> dict:
        """把 json 配置解析成 dict：兼容 WebUI 的 dict 与用户手填的 JSON 字符串。"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                logger.warning("[image_search] DeepSeek 额外头/体不是合法 JSON 对象，已忽略")
        return {}

    def _migrate_config(self):
        """老用户配置安全迁移：补齐新版本新增的键，已有值原样保留。"""
        added = []
        cfg = self.plugin_cfg if isinstance(self.plugin_cfg, dict) else {}
        for section, defaults in self._MIGRATION_DEFAULTS.items():
            sec = cfg.get(section)
            if not isinstance(sec, dict):
                # 整个 section 缺失时交给框架 _ensure_plugin_config 补全
                continue
            for key, value in defaults.items():
                if key not in sec:
                    sec[key] = value
                    added.append(f"{section}.{key}")
        if added:
            logger.info(
                f"[image_search] 检测到旧版配置，已安全补全 {len(added)} 项（不覆盖已有值）: "
                + ", ".join(added)
            )

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._migrate_config()

        basic = cfg.get("section_basic", {})
        self.enabled = basic.get("enabled", True)
        self.max_results = int(basic.get("max_results", 5) or 5)
        self.timeout = int(basic.get("timeout", 20) or 20)
        self.cache_days = int(basic.get("cache_days", 0) or 0)
        self.enable_min_sim = basic.get("enable_min_sim", True)
        self.min_similarity = int(basic.get("min_similarity", 40) or 0)

        net = cfg.get("section_network", {})
        self.proxy = str(net.get("proxy", "") or "").strip()
        self.domestic_direct = bool(net.get("domestic_direct", True))

        lens = cfg.get("section_lens", {})
        # 兜底值与 schema 默认值保持一致（schema: 默认关闭）
        self.enable_serpapi = lens.get("enable_serpapi", False)
        self.serpapi_key = lens.get("serpapi_key", "")

        gem = cfg.get("section_gemini", {})
        self.enable_gemini = gem.get("enable_gemini", True)
        self.gemini_key = gem.get("gemini_key", "")
        self.gemini_model = (gem.get("gemini_model", "") or "gemini-3.5-flash").strip()

        baidu = cfg.get("section_baidu", {})
        self.enable_baidu = baidu.get("enable_baidu", True)

        yandex = cfg.get("section_yandex", {})
        self.enable_yandex = yandex.get("enable_yandex", True)

        ds = cfg.get("section_deepseek", {})
        self.enable_deepseek = ds.get("enable_deepseek", True)
        self.deepseek_key = ds.get("deepseek_key", "")
        self.deepseek_model = (ds.get("deepseek_model", "") or "deepseek-flash").strip()
        self.deepseek_base_url = str(
            ds.get("deepseek_base_url", "") or "https://api.deepseek.com"
        ).strip().rstrip("/")
        self.deepseek_extra_headers = self._as_dict(ds.get("deepseek_extra_headers"))
        self.deepseek_extra_body = self._as_dict(ds.get("deepseek_extra_body"))
        self.deepseek_timeout = int(ds.get("deepseek_timeout", 60) or 60)
        self.deepseek_in_tool = ds.get("deepseek_in_tool", True)
        self.deepseek_require_search = ds.get("deepseek_require_search", False)

        qwen = cfg.get("section_qwen", {})
        self.enable_qwen = qwen.get("enable_qwen", True)
        self.qwen_key = qwen.get("qwen_key", "")
        self.qwen_region = qwen.get("qwen_region", "beijing")
        self.qwen_model = (qwen.get("qwen_model", "") or "qwen3.7-plus").strip()
        self.enable_qwen_image_search = qwen.get("enable_qwen_image_search", True)
        self.enable_qwen_web_search = qwen.get("enable_qwen_web_search", False)
        self.enable_qwen_in_tool = qwen.get("enable_qwen_in_tool", False)
        self.qwen_timeout = int(qwen.get("qwen_timeout", 180) or 180)

        anime = cfg.get("section_anime", {})
        self.enable_saucenao = anime.get("enable_saucenao", True)
        self.saucenao_key = anime.get("saucenao_key", "")
        self.saucenao_merge_versions = anime.get("saucenao_merge_versions", True)
        self.enable_iqdb = anime.get("enable_iqdb", True)
        self.iqdb_in_tool = anime.get("iqdb_in_tool", False)
        self.enable_tracemoe = anime.get("enable_tracemoe", True)
        self.tracemoe_in_image_search = anime.get("tracemoe_in_image_search", False)

        cmd = cfg.get("section_command", {})
        # 兜底值与 schema 默认值保持一致（两个命令默认关闭，命令词默认两条）
        self.enable_command = cmd.get("enable_command", False)
        _words = cmd.get("command_words")
        if _words is None:
            _words = ["/搜图", "/识别图片"]
        self.command_words = [str(w).strip() for w in _words if str(w).strip()]
        self.enable_command_anime = cmd.get("enable_command_anime", False)
        _words_anime = cmd.get("command_words_anime")
        if _words_anime is None:
            _words_anime = ["/搜番"]
        self.command_words_anime = [str(w).strip() for w in _words_anime if str(w).strip()]
        self.send_thumbnail = cmd.get("send_thumbnail", False)
        self.send_forward = cmd.get("send_forward", True)
        self.allowed_users = [str(uid) for uid in (cmd.get("allowed_users") or [])]

        tool_cfg = cfg.get("section_tool", {})
        self.enable_tool = tool_cfg.get("enable_tool", True)
        self.enable_tool_anime = tool_cfg.get("enable_tool_anime", True)

        # sid -> [(ts, Image)]
        self._recent: dict[str, list[tuple[float, Image]]] = {}
        self._saucenao_last = 0.0
        self._iqdb_last = 0.0
        self._baidu_last = 0.0
        self._yandex_last = 0.0
        # 引擎限速锁：锁内完成 check-sleep-set，避免并发协程同时放行
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, dict] = {}
        self._cache_file: Path | None = None

    async def initialize(self):
        if not self.enabled:
            logger.info("[image_search] 插件已禁用")
            return
        data_dir = self.ctx.get_plugin_data_dir()
        if data_dir is None:
            # plugin_mgr 未绑定/热重载早期调用时的兜底，保证插件仍能工作
            data_dir = Path(get_data_path()) / "plugin_data" / "image_search"
            logger.warning(f"[image_search] get_plugin_data_dir() 返回空，回退到 {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = data_dir / "cache.json"
        self._load_cache()
        engines = []
        if self.enable_saucenao:
            engines.append("saucenao" if self.saucenao_key else "saucenao(缺key,已停用)")
        if self.enable_iqdb:
            engines.append("iqdb")
        if self.enable_tracemoe:
            engines.append("tracemoe")
        if self.enable_baidu:
            engines.append("baidu")
        if self.enable_yandex:
            engines.append("yandex")
        if self.enable_gemini:
            engines.append("gemini" if self.gemini_key else "gemini(缺key,已停用)")
        if self.enable_qwen:
            engines.append("qwen" if self.qwen_key else "qwen(缺key,已停用)")
        if self.enable_deepseek:
            engines.append("deepseek" if self.deepseek_key else "deepseek(缺key,已停用)")
        if self.enable_serpapi:
            engines.append("serpapi-lens" if self.serpapi_key else "serpapi-lens(缺key,已停用)")
        logger.info(
            f"[image_search] ready, engines={engines}, command={self.enable_command and bool(self.command_words)}, "
            f"tool={self.enable_tool}"
        )
        if self.enable_saucenao and not self.saucenao_key:
            logger.warning("[image_search] 未配置 SauceNAO key，主力识别引擎不可用！请在 saucenao.com 免费注册获取")
        if self.enable_qwen and self.qwen_key:
            if importlib.util.find_spec("PIL") is None:
                logger.warning("[image_search] 未安装 Pillow，千问图片预压缩已跳过（可选依赖：pip install Pillow）")

    async def terminate(self):
        self._recent.clear()

    # ---------------- 缓存 ----------------

    def _load_cache(self):
        try:
            if self._cache_file and self._cache_file.exists():
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
                else:
                    logger.warning("[image_search] 缓存文件格式不是对象，已忽略")
                    self._cache = {}
        except Exception as e:
            logger.warning(f"[image_search] 缓存读取失败: {e}")
            self._cache = {}

    def _save_cache(self):
        try:
            if not self._cache_file:
                return
            if len(self._cache) > _CACHE_MAX:
                # 按时间淘汰最旧
                items = sorted(self._cache.items(), key=lambda kv: kv[1].get("ts", 0))
                self._cache = dict(items[-_CACHE_MAX:])
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[image_search] 缓存写入失败: {e}")

    def _cache_get(self, md5: str) -> dict | None:
        if self.cache_days <= 0:
            return None
        entry = self._cache.get(md5)
        if not isinstance(entry, dict):
            if entry is not None:
                self._cache.pop(md5, None)
            return None
        if time.time() - entry.get("ts", 0) > self.cache_days * 86400:
            self._cache.pop(md5, None)
            return None
        # ts 是内部元数据，不返回给调用方，
        # 保证缓存命中的结果与首次逐字节一致。
        return {k: v for k, v in entry.items() if k != "ts"}

    def _cache_set(self, md5: str, result: dict):
        # 0 或负数都表示关闭缓存（负数不应写盘，否则读时立刻判过期、纯浪费 I/O）
        if self.cache_days <= 0 or not md5:
            return
        self._cache[md5] = {"ts": int(time.time()), **result}
        self._save_cache()

    async def _rate_limit(self, name: str, min_interval: float, attr: str):
        """引擎限速：锁内计算 gap、sleep、写时间戳，消除 check-then-sleep 竞态。"""
        lock = self._rate_locks.setdefault(name, asyncio.Lock())
        async with lock:
            gap = min_interval - (time.time() - getattr(self, attr, 0.0))
            if gap > 0:
                await asyncio.sleep(gap)
            setattr(self, attr, time.time())

    # ---------------- 引擎：IQDB（免 key，booru 匹配） ----------------

    async def _iqdb(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        await self._rate_limit("iqdb", _IQDB_MIN_INTERVAL, "_iqdb_last")
        resp = await client.post(
            "https://iqdb.org/",
            data={"url": ""},
            files={"file": (filename, img_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        matches = self._parse_iqdb(resp.text)
        return {"engine": "iqdb", "matches": matches}

    @staticmethod
    def _parse_iqdb(html: str) -> list[dict]:
        """解析 IQDB 结果页：提取 站点/链接/相似度/尺寸。IQDB 是服务端渲染的简单 HTML。"""
        matches: list[dict] = []
        # 每个匹配块: <th>Best match</th> 或 <th>Additional match</th> ... 到下一个 <th> 或 </table>
        blocks = re.split(r"<th>(?:Best|Additional|Possible) match(?:es)?</th>", html)
        for block in blocks[1:]:
            block = block.split("</table>")[0]
            # 首个外部链接 = 帖子地址（协议相对 //...）
            m = re.search(r'href="(//(?:danbooru|gelbooru|konachan|yande\.re|anime-pictures|e-shuushuu|www\.zerochan|sankaku)[^"]*)"', block)
            if not m:
                continue
            url = "https:" + m.group(1)
            # 站点名：链接后的文本（如 "Danbooru Gelbooru"）
            site = ""
            sm = re.search(r"</a>\s*(?:<[^>]+>\s*)*([A-Za-z][\w.\- ]{1,40}?)\s*(?:<br|\d+×)", block)
            if sm:
                site = sm.group(1).strip()
            sim_m = re.search(r"(\d+)% similarity", block)
            sim = float(sim_m.group(1)) if sim_m else 0.0
            size_m = re.search(r"(\d+×\d+)", block)
            size = size_m.group(1) if size_m else ""
            # 站点名兜底：从域名推断
            if not site:
                dm = re.search(r"//(?:www\.)?([\w\-]+)\.", url)
                site = dm.group(1) if dm else url
            matches.append({"site": site, "url": url, "sim": sim, "size": size})
        # 按相似度降序
        matches.sort(key=lambda x: -x["sim"])
        return matches

    # ---------------- 引擎：Google Lens SerpApi ----------------

    async def _lens_serpapi(self, client: httpx.AsyncClient, image_url: str) -> dict:
        resp = await client.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_lens",
                "url": image_url,
                "api_key": self.serpapi_key,
                "hl": "zh-cn",
            },
        )
        resp.raise_for_status()
        j = resp.json()
        if "error" in j:
            raise RuntimeError(str(j["error"]))
        knowledge = []
        for kg in j.get("knowledge_graph") or []:
            title = kg.get("title") or ""
            subtitle = kg.get("subtitle") or ""
            if title:
                knowledge.append(f"{title}({subtitle})" if subtitle else title)
        matches = []
        for m in (j.get("visual_matches") or [])[: self.max_results]:
            title = m.get("title") or ""
            link = m.get("link") or ""
            source = m.get("source") or ""
            if title and link:
                matches.append((f"{title} — {source}" if source else title, link))
        thumb = next(
            (m.get("thumbnail") for m in (j.get("visual_matches") or []) if m.get("thumbnail")),
            "",
        )
        return {
            "engine": "serpapi",
            "page_url": (j.get("reverse_image_search") or {}).get("link", ""),
            "knowledge": knowledge[:2],
            "matches": matches,
            "thumb": thumb,
        }

    # ---------------- 引擎：Gemini 谷歌AI搜索（Google Search grounding） ----------------

    _GEMINI_PROMPT = (
        "识别这张图片。如果图中是动漫/游戏角色，给出角色名和出处作品；"
        "如果是表情包/梗图/网络图片，说明它的内容、含义、来源；"
        "请用谷歌搜索验证你的判断。用简体中文简明回答（150字以内），不确定就直说不确定。"
    )

    async def _gemini_lens(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        b64 = base64.b64encode(img_bytes).decode()
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        body = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime, "data": b64}},
                        {"text": self._GEMINI_PROMPT},
                    ]
                }
            ],
            "tools": [{"google_search": {}}],
        }
        # 模型可能被谷歌停用(404)或配额耗尽(429)，按序回退候选模型（配额按模型独立计）
        candidates = [self.gemini_model, "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
        models = list(dict.fromkeys(m for m in candidates if m))
        last_err: Exception | None = None
        for model in models:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": self.gemini_key},
                json=body,
                timeout=max(float(self.timeout), 30.0),
            )
            if resp.status_code in (404, 429):
                reason = "模型不可用" if resp.status_code == 404 else "配额耗尽"
                last_err = RuntimeError(f"{model} {reason}({resp.status_code})")
                logger.warning(f"[image_search] gemini {model} {resp.status_code}({reason})，尝试下一个模型")
                continue
            if resp.status_code != 200:
                try:
                    msg = (resp.json().get("error") or {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                raise RuntimeError(f"gemini {resp.status_code}: {msg}")
            return self._parse_gemini(resp.json(), model)
        raise RuntimeError(f"所有候选 Gemini 模型均不可用（最后错误: {last_err}）")

    def _parse_gemini(self, j: dict, model: str) -> dict:
        cand = (j.get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts).strip()
        gm = cand.get("groundingMetadata") or {}
        queries = [str(q) for q in (gm.get("webSearchQueries") or [])][:5]
        sources = []
        for c in (gm.get("groundingChunks") or []):
            web = c.get("web") or {}
            uri, title = web.get("uri", ""), web.get("title", "")
            if uri:
                sources.append((title or uri, uri))
        return {
            "engine": "gemini",
            "text": text,
            "queries": queries,
            "sources": sources[: self.max_results],
        }

    # ---------------- 引擎：千问图搜图（阿里云百炼 Responses API） ----------------
    #
    # Responses API + image_search(图搜图) 单工具：prompt 约束只搜一次（tool_choice=required
    # 曾导致模型被强制每轮都调工具、陷入 image_search 循环直到超时被杀，已移除），
    # enable_thinking=false 关思考（默认 xhigh 档很慢），实测 10~30s；enable_qwen_web_search
    # 可追加 web_search 联网核实（更准但更慢更贵，约 4 元/千次）。
    # 仅 qwen3.5/3.6/3.7/3.8 系列支持 image_search 工具，且 3.5/3.6 系仅支持 Responses API。
    # 注意：image_search 工具按次收费（约 0.048 元/次）。
    # 旧命名模型（qwen-plus/qwen-flash 等）退回 Chat Completions + enable_search（只有文本联网搜索）。

    _QWEN_PROMPT = (
        "请识别这张图片。调用一次图搜图(image_search)查找相似图片，根据相似图所在网页的标题判断图中主体"
        "（动漫/游戏角色、人物、事物、表情包），再结合你自己的知识给出结论：这是谁/什么、出自哪部作品，"
        "附一句判断依据。若是表情包/梗图，说明其含义和来源。只搜索一次，不要反复搜索。"
        "用简体中文，200字以内；无法确定就直说不确定，不要编造。"
    )

    _QWEN_CHAT_PROMPT = (
        "识别这张图片。如果图中是动漫/游戏角色，给出角色名、出处作品；"
        "如果是表情包/梗图/网络图片，说明其含义、来源。"
        "请自己调用联网搜索核实信息。用简体中文简明回答（200字以内）。"
    )

    # 关闭 image_search 走 web_search 单工具分支时使用（不能提图搜图，工具不存在）
    _QWEN_WEBSEARCH_PROMPT = (
        "识别这张图片。如果图中是动漫/游戏角色，给出角色名、出处作品；"
        "如果是表情包/梗图/网络图片，说明其含义、来源。"
        "调用联网搜索(web_search)核实信息。用简体中文简明回答（200字以内），不确定就直说不确定。"
    )

    # 拒答套话（模型称看不到图片）：命中则不能当有效结果返回（防污染主 LLM）
    _QWEN_REFUSAL_PATTERNS = (
        "无法查看图片", "无法查看这张图片", "无法查看您提供的图片",
        "无法直接查看图片", "无法直接查看这张图片",
        "不能查看图片", "无法识别这张图片", "无法识别图片内容",
        "cannot see the image", "cannot view the image",
        "unable to view the image", "unable to see the image",
        "请提供图片的详细描述", "请上传图片",
    )

    @staticmethod
    def _qwen_is_refusal(text: str) -> bool:
        """千问拒答检测：text 为空返回 False；命中任一拒答模式（忽略大小写）返回 True。"""
        if not text:
            return False
        low = text.lower()
        return any(p.lower() in low for p in ImageSearchPlugin._QWEN_REFUSAL_PATTERNS)

    @staticmethod
    def _compress_for_qwen(img_bytes: bytes) -> bytes:
        """千问图片预压缩：转 RGB、最长边>1024 缩到 1024、存 JPEG(quality=85)，减小上传体积。
        Pillow 未安装或任何异常都原样返回（identity 返回，调用方据此判断 mime）。"""
        try:
            from PIL import Image as PILImage
        except Exception:
            return img_bytes
        try:
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = img.size
            longest = max(w, h)
            if longest > 1024:
                ratio = 1024 / longest
                img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"[image_search] 千问图片预压缩失败（回退原图）: {type(e).__name__}: {e}")
            return img_bytes

    async def _qwen_lens(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str,
                         source_url: str | None = None, budget: float | None = None) -> dict:
        logger.info(f"[image_search] qwen 入参: filename={filename}, img_bytes={len(img_bytes)}, budget={budget}")
        # 预压缩减小 base64 体积加快上传；压缩后统一 JPEG，未压缩时按原文件名后缀定 mime
        compressed = self._compress_for_qwen(img_bytes)
        if compressed is not img_bytes:
            img_bytes, mime = compressed, "image/jpeg"
        else:
            mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode()
        region = (self.qwen_region or "beijing").lower()
        model = self.qwen_model.strip().lower()

        _FIXES = {
            "qwenplus": "qwen-plus", "qwenturbo": "qwen-turbo", "qwenflash": "qwen-flash",
            "qwenmax": "qwen-max", "qwen3max": "qwen3-max",
            "qwen37max": "qwen3.7-max", "qwen37plus": "qwen3.7-plus",
            "qwen38max": "qwen3.8-max-preview",
            "qwen36flash": "qwen3.6-flash", "qwen35flash": "qwen3.5-flash",
            "qwen36plus": "qwen3.6-plus", "qwen35plus": "qwen3.5-plus",
        }
        if model in _FIXES:
            logger.warning(f"[image_search] qwen 模型名修正: '{model}' → '{_FIXES[model]}'")
            model = _FIXES[model]

        base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if region == "singapore":
            base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

        data_url = f"data:{mime};base64,{b64}"

        # qwen3.5/3.6/3.7/3.8 系列：Responses API（3.5/3.6 系不支持 Chat Completions，
        # 关闭图搜图时也要用 web_search 单工具走 Responses）。
        # 单工具 + prompt 约束只搜一次 + enable_thinking=false 关思考，实测 10~30s。
        # 流式提前截获图搜结果，模型总结卡 15s 提前断开，不再陪跑到 budget 耗尽。
        # input_image 实测支持 data: base64 直连（image_search 工具也能用），无需图床/上传。
        if model.startswith(("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8")):
            limit = float(budget if budget is not None else self.qwen_timeout)
            deadline = time.time() + max(limit, 15.0)
            # 模型不支持 image_search 工具(400)时按序回退：qwen3.7-plus 是官方图搜图推荐模型
            #（官方示例均用此）；qwen3.6-flash 更快但官方注明效果不如推荐模型
            candidates = list(dict.fromkeys([model, "qwen3.7-plus", "qwen3.6-flash", "qwen3.5-plus"]))
            last_err: Exception | None = None
            for m in candidates:
                remaining = deadline - time.time()
                if remaining < 10:
                    break
                # 单次尝试上限：预算的 60%（至少 25s），给后续回退模型留余量（F5）
                attempt_cap = min(remaining, max(limit * 0.6, 25.0))
                attempts = 2  # 连接类抖动（ConnectError/ConnectTimeout）允许对当前模型原样重试 1 次
                give_up_all = False
                t0 = time.time()
                while True:
                    try:
                        return await asyncio.wait_for(
                            self._qwen_responses_search(client, base, m, data_url, attempt_cap),
                            timeout=attempt_cap,
                        )
                    except Exception as e:
                        elapsed = time.time() - t0
                        if attempts > 1 and isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                            attempts -= 1
                            # 重试前重算剩余预算，不足 10s 放弃重试，换下一个候选模型（F4）
                            remaining = deadline - time.time()
                            if remaining < 10:
                                logger.warning(
                                    f"[image_search] qwen {m} 连接失败，剩余预算 {remaining:.1f}s 不足，放弃重试: {type(e).__name__}: {e}"
                                )
                                last_err = RuntimeError(f"{type(e).__name__}: {e}")
                                break
                            attempt_cap = min(remaining, max(limit * 0.6, 25.0))
                            logger.warning(
                                f"[image_search] qwen {m} 连接失败，原样重试 1 次(耗时 {elapsed:.1f}s): {type(e).__name__}: {e}"
                            )
                            continue
                        last_err = RuntimeError(f"{type(e).__name__}: {e}")
                        logger.warning(f"[image_search] qwen {m} 图搜图失败(耗时 {elapsed:.1f}s): {type(e).__name__}: {e}")
                        # 超时类错误不再整体放弃：预算充足(>=15s)则跳过当前模型、尝试下一个候选
                        if isinstance(e, (TimeoutError, httpx.TimeoutException)):
                            remaining = deadline - time.time()
                            if remaining >= 15:
                                logger.warning(
                                    f"[image_search] qwen {m} 超时，跳过当前模型，尝试下一个(剩余 {remaining:.1f}s)"
                                )
                            else:
                                logger.warning(
                                    f"[image_search] qwen {m} 超时，剩余预算 {remaining:.1f}s 不足，放弃后续候选"
                                )
                                give_up_all = True
                        break
                if give_up_all:
                    break
            # 最终降级：Responses 路径全部失败且预算还剩 >=15s 时，
            # 用 qwen-plus Chat Completions 文本联网搜索兜底
            remaining = deadline - time.time()
            if remaining >= 15:
                logger.warning("[image_search] Responses 路径全部失败，降级 qwen-vl-max 视觉识别")
                try:
                    return await asyncio.wait_for(
                        self._qwen_chat_simple(client, base, "qwen-vl-max", data_url,
                                               timeout_s=min(remaining, 30.0)),
                        timeout=remaining,
                    )
                except Exception as e:
                    logger.warning(f"[image_search] qwen-vl-max 降级也失败: {type(e).__name__}: {e}")
            raise RuntimeError(f"千问图搜图失败（最后错误: {last_err}）")

        # 旧命名模型（qwen-plus/qwen-flash 等）：Chat Completions + enable_search
        return await self._qwen_chat_simple(client, base, model, data_url)

    async def _qwen_responses_search(self, client, base: str, model: str, img_url: str,
                                     timeout_s: float = 240.0, summary_grace_s: float = 15.0) -> dict:
        """Responses API 识图：默认仅 image_search 单工具 + enable_thinking=false（关思考省数秒~数十秒）。
        只在 prompt 里约束"只搜索一次"（不传 tool_choice=required——它会强制模型每一轮都调工具，
        陷入 image_search 循环直到超时被杀）。enable_qwen_web_search 开启时追加 web_search 核实
        （更准但更慢更贵）。

        工具调用耗时长，官方建议流式；收 response.completed 事件拿完整响应，
        同时监听 output_item.done 提前截获图搜结果，模型总结卡住时提前断开。
        """
        if self.enable_qwen_image_search:
            tools = [{"type": "image_search"}]
            if self.enable_qwen_web_search:
                tools.append({"type": "web_search"})
            prompt = self._QWEN_PROMPT
        else:
            # 关闭图搜图时退化为 web_search 单工具（3.5/3.6 系只能走 Responses）
            tools = [{"type": "web_search"}]
            prompt = self._QWEN_WEBSEARCH_PROMPT
        body = {
            "model": model,
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": img_url},
                {"type": "input_text", "text": prompt},
            ]}],
            "tools": tools,
            # 关思考双保险：qwen3.6/3.7 系是混合思考模型，Responses API 的 reasoning.effort
            # 默认 xhigh，模型会长考几十秒才发第一个事件（"50s 零事件"主因）。
            # reasoning 优先级高于 enable_thinking（官方将废弃后者）：reasoning 管新端点，
            # enable_thinking 兜底老端点，两处都传。
            "reasoning": {"effort": "none"},
            "enable_thinking": False,
            "max_output_tokens": 400,
            "stream": True,
        }
        # 部分端点/代理不认识 reasoning/enable_thinking 参数（400 Unknown parameter 等），
        # 逐级降级重试：先去 reasoning（保留 enable_thinking）重试一次；仍 400 且错误
        # 指向 enable_thinking，再去掉 enable_thinking 重试一次。每次降级记 warning。
        _REJECT_KEYS = ("reasoning", "enable_thinking", "unknown parameter", "unsupported")
        try:
            return await self._qwen_responses_stream(client, base, body, model, timeout_s, summary_grace_s)
        except _QwenResponsesHttpError as e:
            msg = e.message.lower()
            if e.status != 400 or "reasoning" not in body or not any(k in msg for k in _REJECT_KEYS):
                raise
            logger.warning(
                f"[image_search] qwen {model} 400 拒绝思考参数({e.message[:120]})，"
                f"去掉 reasoning 降级重试"
            )
            body.pop("reasoning", None)
        try:
            return await self._qwen_responses_stream(client, base, body, model, timeout_s, summary_grace_s)
        except _QwenResponsesHttpError as e:
            msg = e.message.lower()
            if e.status != 400 or "enable_thinking" not in body or "enable_thinking" not in msg:
                raise
            logger.warning(
                f"[image_search] qwen {model} 400 拒绝 enable_thinking 参数({e.message[:120]})，"
                f"去掉 enable_thinking 降级重试"
            )
            body.pop("enable_thinking", None)
            return await self._qwen_responses_stream(client, base, body, model, timeout_s, summary_grace_s)

    async def _qwen_responses_stream(self, client, base: str, body: dict, model: str,
                                     timeout_s: float, summary_grace_s: float = 15.0) -> dict:
        """发起流式 Responses 请求，收 response.completed 事件拿完整响应。
        同时监听 response.output_item.done 提前截获 image_search_call/web_search_call 结果。

        静默期处理（F1 核心修复）：官方确认 image_search 工具执行期间 SSE 流完全静默
        （output_item.added → done 之间无任何事件），写在 async for 循环体内的"提前
        断开"检查在静默期永不触发。这里改为手动迭代器 + 带超时的 anext 逐行读取：
        拿到图搜结果前，行静默上限 = 剩余预算；拿到后 = summary_grace_s（模型总结静默
        超时则提前返回已截获的相似图，不再陪跑到预算耗尽）。
        延迟上界 = 图搜图耗时 + summary_grace_s。
        关键事件记 info 日志（带耗时 t=），便于诊断卡在思考/工具/总结哪一段。"""
        headers = {"Authorization": f"Bearer {self.qwen_key}"}
        t0 = time.time()
        deadline = t0 + max(float(timeout_s), 5.0)
        completed: dict | None = None
        early_matches: list[tuple[str, str]] = []   # 图搜图相似图 (title, url)
        early_sources: list[tuple[str, str]] = []   # 联网搜索来源 (title, url)
        early_queries: list[str] = []
        first_image_result_ts: float | None = None
        unknown_ev = 0
        # per-request timeout 整体覆盖 client 级 timeout（单引擎超时 20s 不影响千问）；
        # 分项设置：connect 10s（不再被抬到 50s），read 给足本次尝试预算，
        # 容忍 image_search 工具执行期间的 SSE 静默（F5）
        req_timeout = httpx.Timeout(connect=10.0, read=float(timeout_s), write=30.0, pool=5.0)
        async with client.stream(
            "POST", f"{base}/responses",
            headers=headers,
            json=body, timeout=req_timeout,
        ) as resp:
            if not resp.is_success:
                raw = (await resp.aread()).decode("utf-8", "replace")
                err = raw[:200]
                try:
                    err = (json.loads(raw).get("error") or {}).get("message", err)
                except Exception:
                    pass
                raise _QwenResponsesHttpError(resp.status_code, err)
            aiter = resp.aiter_lines()
            while True:
                # 拿到图搜结果后：行静默超时 summary_grace_s（可调）；拿到前：用剩余预算
                idle = summary_grace_s if first_image_result_ts else max(deadline - time.time(), 5.0)
                try:
                    line = await asyncio.wait_for(aiter.__anext__(), timeout=idle)
                except asyncio.TimeoutError:
                    # 图搜结果已到、模型总结静默超时：提前返回已截获的相似图，不再陪跑
                    if first_image_result_ts and early_matches:
                        logger.info(
                            f"[image_search] 千问模型总结静默超时({idle:.0f}s)，提前返回 "
                            f"{len(early_matches)} 条相似图标题(t={time.time() - t0:.1f}s)"
                        )
                        break
                    # 尚未拿到图搜结果就静默超时：抛 TimeoutError，与外层 wait_for 语义一致
                    raise TimeoutError(f"千问流式响应静默超时({idle:.0f}s)")
                except StopAsyncIteration:
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    ev = json.loads(data)
                except Exception:
                    logger.warning(
                        f"[image_search] qwen SSE 行 JSON 解析失败: len={len(data)}, 前100字符={data[:100]!r}"
                    )
                    continue
                ev_type = ev.get("type")
                if ev_type == "response.created":
                    logger.info(f"[image_search] qwen 流已建立(t={time.time() - t0:.1f}s)")
                elif ev_type == "response.output_item.added":
                    logger.info(
                        f"[image_search] qwen output_item.added: "
                        f"{(ev.get('item') or {}).get('type')}(t={time.time() - t0:.1f}s)"
                    )
                elif ev_type == "response.completed":
                    completed = ev.get("response") or {}
                elif ev_type == "response.failed":
                    msg = ((ev.get("response") or {}).get("error") or {}).get("message", "unknown")
                    raise RuntimeError(f"qwen responses failed: {msg}")
                elif ev_type == "response.output_item.done":
                    item = ev.get("item") or {}
                    itype = item.get("type")
                    if itype == "image_search_call":
                        out = item.get("output")
                        if isinstance(out, str):
                            try:
                                out = json.loads(out)
                            except Exception:
                                out = []
                        for m in (out or []):
                            if isinstance(m, dict):
                                t, u = str(m.get("title") or ""), str(m.get("url") or "")
                                if t or u:
                                    early_matches.append((t, u))
                        if early_matches and first_image_result_ts is None:
                            first_image_result_ts = time.time()
                        logger.info(
                            f"[image_search] qwen output_item.done: image_search_call "
                            f"相似图 {len(early_matches)} 条(t={time.time() - t0:.1f}s)"
                        )
                    elif itype == "web_search_call":
                        action = item.get("action") or {}
                        if action.get("query"):
                            early_queries.append(str(action["query"]))
                        for s in (action.get("sources") or []):
                            u = s.get("url", "")
                            if u:
                                early_sources.append((s.get("title") or u, u))
                        logger.info(
                            f"[image_search] qwen output_item.done: web_search_call "
                            f"来源 {len(early_sources)} 条(t={time.time() - t0:.1f}s)"
                        )
                    else:
                        logger.info(
                            f"[image_search] qwen output_item.done: {itype}(t={time.time() - t0:.1f}s)"
                        )
                else:
                    unknown_ev += 1
                    if unknown_ev <= 3:
                        logger.debug(f"[image_search] qwen 未识别事件: {ev_type}(t={time.time() - t0:.1f}s)")
        if completed is None:
            # completed 未到但已截获图搜结果：用截获数据构造返回，不视为失败
            if early_matches:
                early_matches = list(dict.fromkeys(early_matches))
                early_sources = list(dict.fromkeys(early_sources))
                return {
                    "engine": "qwen", "model_used": model, "text": "",
                    "sources": early_sources[: self.max_results],
                    "image_matches": early_matches[: self.max_results * 2],
                    "queries": early_queries[:5],
                }
            raise RuntimeError("qwen responses 流式响应未收到 completed 事件")
        return self._parse_qwen_responses(completed, model)

    def _parse_qwen_responses(self, j: dict, model: str) -> dict:
        """解析 Responses API 输出：message 文本 / web_search_call 来源 / image_search_call 相似图。"""
        text_parts: list[str] = []
        sources: list[tuple[str, str]] = []       # 联网搜索来源 (title, url)
        image_matches: list[tuple[str, str]] = []  # 图搜图相似图 (title, url)
        queries: list[str] = []
        for item in (j.get("output") or []):
            itype = item.get("type")
            if itype == "message":
                for c in (item.get("content") or []):
                    t = c.get("text", "")
                    if t and "image_pad" not in t.lower():
                        text_parts.append(t)
                    for ann in (c.get("annotations") or []):
                        u = ann.get("url", "")
                        if ann.get("type") == "url_citation" and u:
                            sources.append((ann.get("title") or u, u))
            elif itype == "web_search_call":
                action = item.get("action") or {}
                if action.get("query"):
                    queries.append(str(action["query"]))
                for s in (action.get("sources") or []):
                    u = s.get("url", "")
                    if u:
                        sources.append((s.get("title") or u, u))
            elif itype == "image_search_call":
                out = item.get("output")
                if isinstance(out, str):
                    try:
                        out = json.loads(out)
                    except Exception:
                        out = []
                for m in (out or []):
                    if isinstance(m, dict):
                        t, u = str(m.get("title") or ""), str(m.get("url") or "")
                        if t or u:
                            image_matches.append((t, u))
        # 去重保序
        sources = list(dict.fromkeys(sources))
        image_matches = list(dict.fromkeys(image_matches))
        text = "".join(text_parts).strip()
        logger.info(
            f"[image_search] qwen {model}: text={len(text)}字, web来源={len(sources)}, "
            f"相似图={len(image_matches)}, queries={queries[:3]}"
        )
        # 拒答检测：称"无法查看图片"的套话不能当有效文本（防污染主 LLM）
        if text and ImageSearchPlugin._qwen_is_refusal(text):
            if not image_matches:
                raise RuntimeError("千问拒答：称无法查看图片")
            logger.warning("[image_search] 千问拒答(称无法查看图片)，丢弃文本")
            text = ""
        if not text:
            text = "以图搜图完成" + (f"，找到{len(image_matches)}张相似图" if image_matches else "，未找到明确匹配")
        return {
            "engine": "qwen", "model_used": model, "text": text,
            "sources": sources[: self.max_results],
            "image_matches": image_matches[: self.max_results * 2],
            "queries": queries[:5],
        }

    async def _qwen_chat_simple(self, client, base: str, model: str, data_url: str,
                                timeout_s: float | None = None) -> dict:
        """Chat Completions + enable_search。qwen-vl-max/qwen-plus 等旧模型的路径。
        qwen-vl 系可能不支持联网搜索参数：400 且错误信息涉及 search/enable_search 时，
        去掉 enable_search 重试一次（记 warning）；仍失败原样抛。"""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": self._QWEN_CHAT_PROMPT},
            ]}],
            "max_tokens": 500, "enable_search": True,
        }
        dropped_search = False
        while True:
            resp = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {self.qwen_key}"},
                json=body, timeout=timeout_s or max(float(self.timeout), 30.0),
            )
            if resp.is_success:
                break
            err = ""
            try:
                err = (resp.json().get("error") or {}).get("message", resp.text[:200])
            except Exception:
                err = resp.text[:200]
            # qwen-vl 系可能不支持 enable_search 参数：去掉后重试一次
            if (resp.status_code == 400 and not dropped_search
                    and "enable_search" in body and "search" in err.lower()):
                logger.warning(
                    f"[image_search] qwen {model} 400 拒绝联网搜索参数({err[:120]})，"
                    f"去掉 enable_search 重试一次"
                )
                body.pop("enable_search", None)
                dropped_search = True
                continue
            raise RuntimeError(f"qwen chat {resp.status_code}: {err}")
        j = resp.json()
        choices = j.get("choices") or [{}]
        content = str((choices[0].get("message") or {}).get("content") or "").strip()
        # 拒答检测：称无法查看图片的套话按失败处理（让外层继续降级），不进入 _format_llm
        if content and ImageSearchPlugin._qwen_is_refusal(content):
            raise RuntimeError("千问拒答：称无法查看图片")
        sr = (j.get("search_info") or {}).get("search_results") or []
        sources = []
        for s in sr[:self.max_results]:
            u = s.get("url", "")
            if u:
                sources.append((s.get("title", ""), u))
        return {"engine": "qwen", "model_used": model, "text": content, "sources": sources}

# ---------------- 引擎：DeepSeek 官方图搜（deepseek-flash + deepseek_web_search） ----------------

    _DEEPSEEK_PROMPT = (
        "请先用图搜图/联网搜索查证，再回答：这张图里是什么？"
        "如果是角色/人物，给出名字和出处作品；如果是物品/场景/梗图，说明它是什么、来自哪里。"
        "请给出判断依据。用简体中文，200字以内；不确定就直说不确定，不要编造。"
    )

    async def _deepseek_lens(self, client, img_bytes: bytes, filename: str,
                             budget: float | None = None) -> dict:
        """DeepSeek 官方图搜：deepseek-flash 视觉 + deepseek_web_search 服务端图搜。

        DeepSeek 是国内服务，请求走直连 client；图片按千问同款策略预压缩。
        服务端会自动发起图片搜索并在 annotations 里返回引用来源。
        """
        if not self.deepseek_key:
            raise RuntimeError("未配置 DeepSeek API Key")
        compressed = self._compress_for_qwen(img_bytes)
        if compressed is not img_bytes:
            img_bytes, mime = compressed, "image/jpeg"
        else:
            mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode()
        model = (self.deepseek_model or "deepseek-flash").strip()
        body = {
            "model": model,
            "thinking": {"type": "disabled"},
            "tools": [{"type": "deepseek_web_search"}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._DEEPSEEK_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": 1200,
        }
        timeout = float(budget if budget is not None else self.deepseek_timeout)
        # 合并第三方端点需要的额外请求体（同名键覆盖插件默认值）
        if self.deepseek_extra_body:
            body = {**body, **self.deepseek_extra_body}
        headers = {"Authorization": f"Bearer {self.deepseek_key}"}
        if self.deepseek_extra_headers:
            headers.update(self.deepseek_extra_headers)
        resp = await client.post(
            self._deepseek_chat_url(),
            headers=headers,
            json=body,
            timeout=httpx.Timeout(connect=15.0, read=timeout, write=30.0, pool=5.0),
        )
        if not resp.is_success:
            err = resp.text[:200]
            try:
                err = (resp.json().get("error") or {}).get("message", err)
            except Exception:
                pass
            raise RuntimeError(f"deepseek {resp.status_code}: {err}")
        j = resp.json()
        msg = ((j.get("choices") or [{}])[0].get("message") or {})
        text = str(msg.get("content") or "").strip()

        # server_tool_calls 里有完整的图搜来源（url/title/snippet/site_name/image_url），
        # annotations 只是回答里引用到的子集；两者合并去重。
        sources: list[tuple[str, str]] = []
        images: list[tuple[str, str]] = []
        image_search_count = 0
        for group in msg.get("server_tool_calls") or []:
            for call in group.get("calls") or []:
                if call.get("kind") != "image_search":
                    continue
                if call.get("status") not in (None, "completed"):
                    continue
                image_search_count += 1
                for s in call.get("sources") or []:
                    url = s.get("url") or ""
                    title = s.get("title") or s.get("site_name") or url
                    site = s.get("site_name") or ""
                    if site and site not in title:
                        title = f"{title} ({site})"
                    if url:
                        sources.append((title, url))
                    iu = s.get("image_url") or ""
                    if iu:
                        images.append((title, iu))
        for a in msg.get("annotations") or []:
            cite = a.get("url_citation") or {}
            url = cite.get("url") or ""
            if url:
                sources.append((cite.get("title") or url, url))
        sources = list(dict.fromkeys(sources))
        images = list(dict.fromkeys(images))

        tool_use = ((j.get("usage") or {}).get("server_tool_use") or {})
        usage_count = int(tool_use.get("tool_image_search") or 0)
        image_search_count = max(image_search_count, usage_count)
        verified = image_search_count > 0 and bool(sources)

        logger.info(
            f"[image_search] deepseek {model}: text={len(text)}字, sources={len(sources)}, "
            f"images={len(images)}, image_search={image_search_count}, verified={verified}"
        )
        if not text:
            raise RuntimeError("DeepSeek 返回空结果（可能触发安全审核、额度不足或模型不支持视觉）")
        if not verified and self.deepseek_require_search:
            raise RuntimeError(
                "DeepSeek 未真正触发图搜（纯识图结果不可信，已按配置丢弃）"
            )
        return {
            "engine": "deepseek", "model_used": model, "text": text,
            "sources": sources[: self.max_results],
            "images": images[: self.max_results],
            "image_search_count": image_search_count,
            "verified": verified,
        }

# ---------------- 引擎：SauceNAO ----------------

    @staticmethod
    def _sauce_base_name(text: str) -> str:
        """去掉章节/汉化标记，得到用于聚合的作品名。"""
        if not text:
            return ""
        s = str(text)
        s = re.sub(r"\[[^\]]*\]", " ", s)  # [Fue] [Chinese] [丧尸汉化]
        s = re.sub(r"\b(?:Ch(?:apter)?\.?)\s*\d+(?:\s*[-~]\s*\d+)?", " ", s, flags=re.I)
        s = re.sub(r"第\s*\d+(?:\s*[-~]\s*\d+)?\s*[話话集]", " ", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    @staticmethod
    def _sauce_chapter_end(text: str) -> int:
        """取章节范围的结束数字（Ch. 1-5 / 第1-5話 -> 5），无章节返回 0。"""
        m = re.search(r"(?:Ch(?:apter)?\.?|第)\s*(\d+)(?:\s*[-~]\s*(\d+))?", str(text or ""), re.I)
        if not m:
            return 0
        return int(m.group(2) or m.group(1))

    def _merge_sauce_results(self, results: list[dict]) -> list[dict]:
        """按 (画师, 作品名) 聚合：保留章节最全的一条，其余折叠为计数并合并链接。"""
        groups: dict[tuple, list[dict]] = {}
        for r in results:
            base = self._sauce_base_name(r.get("source") or r.get("title"))
            if not base:
                base = str(r.get("title") or "").strip().lower()
            key = (str(r.get("member") or "").strip().lower(), base, r.get("index_id"))
            groups.setdefault(key, []).append(r)
        merged: list[dict] = []
        for items in groups.values():
            items.sort(key=lambda x: (-float(x.get("sim") or 0),
                                      -self._sauce_chapter_end(x.get("source") or x.get("title"))))
            main = dict(items[0])
            urls: list[str] = []
            for it in items:
                for u in (it.get("urls") or ([it.get("url")] if it.get("url") else [])):
                    if u and str(u) not in urls:
                        urls.append(str(u))
            main["urls"] = urls[:2]
            main["url"] = urls[0] if urls else ""
            main["merged_count"] = len(items) - 1
            merged.append(main)
        merged.sort(key=lambda x: -float(x.get("sim") or 0))
        return merged

    async def _saucenao(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        await self._rate_limit("saucenao", _SAUCENAO_MIN_INTERVAL, "_saucenao_last")
        resp = await client.post(
            "https://saucenao.com/search.php",
            data={
                "output_type": "2",
                "api_key": self.saucenao_key,
                "numres": str(max(self.max_results, 10)),
                "db": "999",
            },
            files={"file": (filename, img_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        j = resp.json()
        header = j.get("header") or {}
        if header.get("status") not in (0, None):
            raise RuntimeError(f"saucenao status={header.get('status')}: {header.get('message', '')}")
        results = []
        raw = j.get("results") or []
        numres = max(self.max_results, 10)
        for item in raw[:numres]:
            h = item.get("header") or {}
            d = item.get("data") or {}
            try:
                sim = float(h.get("similarity") or 0)
            except (TypeError, ValueError):
                sim = 0.0
            title = d.get("title") or d.get("jp_name") or d.get("eng_name") or d.get("source") or ""
            member = (d.get("member_name") or d.get("creator")
                      or d.get("artist") or d.get("author") or "")
            if isinstance(member, list):
                member = ", ".join(str(m) for m in member)
            urls = d.get("ext_urls") or []
            source = d.get("source") or ""
            # booru 系结果直接带角色/作品字段（这是 SauceNAO 最有价值的部分）
            characters = d.get("characters") or ""
            if isinstance(characters, list):
                characters = ", ".join(str(c) for c in characters)
            material = d.get("material") or ""
            if isinstance(material, list):
                material = ", ".join(str(c) for c in material)
            part = str(d.get("part") or "").strip(" -")
            results.append(
                {
                    "sim": sim,
                    "title": title,
                    "member": str(member),
                    "url": urls[0] if urls else "",
                    "urls": [str(u) for u in urls[:2]],
                    "part": part,
                    "index_id": h.get("index_id"),
                    "source": source,
                    "characters": str(characters),
                    "material": str(material),
                    "thumb": h.get("thumbnail") or "",
                }
            )
        # 前 N 条可能全是没有外链的 H-Manga 条目；若都没有 ext_urls，
        # 从后续结果补一条带链接的（不挤掉原有条目），保证能给出漫画网址。
        selected = results[: self.max_results]
        if not any(r.get("urls") for r in selected):
            extra = next((r for r in results[self.max_results:] if r.get("urls")), None)
            if extra:
                selected.append(extra)
        results = selected
        if self.saucenao_merge_versions and len(results) > 1:
            results = self._merge_sauce_results(results)
        top_sim = 0.0
        try:
            top_sim = max(float((x.get("header") or {}).get("similarity") or 0) for x in raw) if raw else 0.0
        except (TypeError, ValueError):
            pass
        logger.info(f"[image_search] saucenao: {len(raw)} raw results, top_sim={top_sim}")
        return {"engine": "saucenao", "results": results, "raw_count": len(raw), "top_sim": top_sim}

    # ---------------- 引擎：百度识图（免 key 免验证，实测可用） ----------------

    async def _baidu(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        await self._rate_limit("baidu", _BAIDU_MIN_INTERVAL, "_baidu_last")
        # 1) 拿 cookie
        await client.get("https://graph.baidu.com/pcpage/index?tpl_from=pc")

        async def _upload() -> str:
            # Acs-Token: 601 是官方前端 SDK 加载失败时的降级值，实测免验证可用
            resp = await client.post(
                f"https://graph.baidu.com/upload?uptime={int(time.time() * 1000)}",
                data={
                    "tn": "pc",
                    "from": "pc",
                    "image_source": "PC_UPLOAD_SEARCH_FILE",
                    "sdkParams": "undefined",
                    "imageType": "jpeg",
                },
                files={"image": (filename, img_bytes, "image/jpeg")},
                headers={"Acs-Token": "601"},
            )
            resp.raise_for_status()
            j = resp.json()
            if j.get("status") != 0:
                raise RuntimeError(f"baidu upload: {j.get('msg', 'unknown')}")
            return j["data"]["url"]

        page_url = await _upload()
        # 3) 结果页 → window.cardData：
        #    cardHeader.pano = 识图猜测词（"明日方舟-安魂曲"式实体猜测，名字常准、作品可能张冠李戴，是最有价值的线索）
        #    same 卡片 = 图片来源（带标题），simipic 兜底
        r2 = await client.get(page_url)
        cards = self._baidu_cards(r2.text)
        guesses = self._baidu_guesses(cards)
        if not guesses:
            # 猜测词每次请求不稳定（实测约 1/3 概率为空或泛泛如"角色插画"），重传重试一次（~1.5s）
            try:
                page_url2 = await _upload()
                r2 = await client.get(page_url2)
                cards2 = self._baidu_cards(r2.text)
                retry_guesses = self._baidu_guesses(cards2)
                if retry_guesses:
                    page_url, cards, guesses = page_url2, cards2, retry_guesses
            except Exception:
                pass
        sources: list[dict] = []
        for card in cards:
            name = card.get("cardName")
            if name == "cardHeader":
                continue
            if name != "same":
                continue
            for item in ((card.get("tplData") or {}).get("list") or []):
                title = item.get("title") or ""
                if isinstance(title, list):
                    title = " ".join(str(t) for t in title)
                url = item.get("url") or ""
                if not (title or url):
                    continue
                domain = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
                sources.append(
                    {
                        "title": str(title).strip(),
                        "url": url,
                        "website": item.get("website") or (domain.group(1) if domain else ""),
                        "thumb": item.get("image_src") or "",
                        "abstract": item.get("abstract") or "",
                    }
                )
        matches: list[dict] = []
        if not sources:
            # 4) 无 same 卡片时走相似图 JSON（只有链接没有标题）
            first_url = ""
            for card in cards:
                if card.get("cardName") == "simipic":
                    first_url = (card.get("tplData") or {}).get("firstUrl") or ""
                    break
            if first_url:
                r3 = await client.get(first_url, headers={"Referer": "https://graph.baidu.com/"})
                dj = r3.json()
                for item in ((dj.get("data") or {}).get("list") or [])[: self.max_results * 2]:
                    from_url = item.get("fromUrl") or ""
                    if not from_url:
                        continue
                    domain = re.search(r"https?://(?:www\.)?([^/]+)", from_url)
                    matches.append(
                        {
                            "url": from_url,
                            "thumb": item.get("thumbUrl") or "",
                            "domain": domain.group(1) if domain else "",
                        }
                    )
                seen: set[str] = set()
                uniq = []
                for m in matches:
                    if m["domain"] in seen:
                        continue
                    seen.add(m["domain"])
                    uniq.append(m)
                matches = uniq[: self.max_results]
        return {"engine": "baidu", "guesses": guesses[:3], "sources": sources[: self.max_results], "matches": matches, "page_url": page_url}

    @staticmethod
    def _baidu_guesses(cards: list[dict]) -> list[str]:
        """从 cardHeader.pano 提取识图猜测词（如"明日方舟-安魂曲"）。"""
        guesses: list[str] = []
        for card in cards:
            if card.get("cardName") != "cardHeader":
                continue
            pano = (card.get("tplData") or {}).get("pano") or {}
            for item in pano.get("list") or []:
                st = str(item.get("subTitle") or "").strip()
                if st and st not in guesses:
                    guesses.append(st)
        return guesses

    @staticmethod
    def _baidu_cards(html: str) -> list[dict]:
        """从结果页提取 window.cardData（JSON 数组）。"""
        idx = html.find("window.cardData")
        if idx == -1:
            return []
        start = html.find("[", idx)
        if start == -1:
            return []
        depth, in_str, esc, end = 0, False, False, -1
        for i in range(start, len(html)):
            c = html[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end <= start:
            return []
        try:
            return json.loads(html[start:end])
        except Exception:
            return []

    # ---------------- 引擎：Yandex 识图（免 key，俄语区索引强，人物/角色识别补充） ----------------

    async def _yandex(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        await self._rate_limit("yandex", _YANDEX_MIN_INTERVAL, "_yandex_last")
        last_err: Exception | None = None
        result: dict | None = None
        for host in ("https://yandex.com", "https://yandex.ru"):
            try:
                resp = await client.post(
                    f"{host}/images/search?rpt=imageview&cbir_page=sites",
                    data={"prg": 1},
                    files={"upfile": (filename, img_bytes, "image/jpeg")},
                )
                resp.raise_for_status()
                result = self._parse_yandex(resp.text, str(resp.url))
                break
            except Exception as e:
                last_err = e
                logger.warning(f"[image_search] yandex {host} 失败: {type(e).__name__}: {e}")
        if result is None:
            raise RuntimeError(f"yandex 两个域名均失败（最后错误: {type(last_err).__name__}: {last_err}）")
        # 标题/标签多为俄语等外文，不再翻译（交由 LLM 自行翻译理解）
        return result

    @staticmethod
    def _parse_yandex(page: str, serp_url: str) -> dict:
        """解析 Yandex 结果页：定位 ImagesApp 节点的 data-state 属性（服务端渲染 JSON）。"""
        # 先定位 id="ImagesApp- 节点，再在其后 2000 字符内找 data-state="，不假设属性顺序
        idx = page.find('id="ImagesApp-')
        if idx == -1:
            # 找不到 ImagesApp 节点视为触发验证码/结构变更
            raise RuntimeError("结果页未找到 data-state（可能触发验证码或页面结构变更）")
        ds = page.find('data-state="', idx, idx + 2000)
        if ds == -1:
            raise RuntimeError("结果页未找到 data-state（可能触发验证码或页面结构变更）")
        start = ds + len('data-state="')
        end = page.find('"', start)
        if end == -1:
            raise RuntimeError("data-state 属性值不完整")
        try:
            state = json.loads(html.unescape(page[start:end]))
        except Exception as e:
            raise RuntimeError(f"data-state JSON 解析失败: {e}")
        init = state.get("initialState") or {}
        tags = []
        for t in ((init.get("cbirTags") or {}).get("tags") or []):
            text = str(t.get("text") or "").strip()
            if text:
                tags.append(text)
        sites = []
        for s in ((init.get("cbirSites") or {}).get("sites") or []):
            thumb = (s.get("thumb") or {}).get("url") or ""
            if thumb.startswith("//"):
                thumb = "https:" + thumb  # 协议相对地址补 https:
            sites.append({
                "title": str(s.get("title") or "").strip(),
                "url": s.get("url") or "",
                "domain": s.get("domain") or "",
                "thumb": thumb,
            })
        # 按归一化标题（lower+strip）去重，保留首次出现（解决 2ch.su/2ch.life/2ch.org 等镜像站重复）
        seen_titles: set[str] = set()
        uniq_sites = []
        for site in sites:
            key = (site.get("title") or "").strip().lower()
            if key:
                if key in seen_titles:
                    continue
                seen_titles.add(key)
            uniq_sites.append(site)
        return {"engine": "yandex", "tags": tags, "sites": uniq_sites, "serp_url": serp_url}

    # ---------------- 引擎：trace.moe ----------------

    async def _tracemoe(self, client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> dict:
        resp = await client.post(
            "https://api.trace.moe/search?anilistInfo&cutBorders",
            files={"image": (filename, img_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        j = resp.json()
        if j.get("error"):
            raise RuntimeError(str(j["error"]))
        results = []
        seen = set()
        for r in (j.get("result") or [])[: self.max_results]:
            sim = (r.get("similarity") or 0) * 100
            # 官方称相似度 <90% 基本为错误结果，cutoff 提到 80，80-89 标注疑似
            if sim < 80:
                continue
            anilist = r.get("anilist") or {}
            title_obj = anilist.get("title") or {}
            title = title_obj.get("native") or title_obj.get("romaji") or title_obj.get("english") or ""
            # 中文名在 title.chinese（synonyms_chinese 常为空）
            title_cn = str(title_obj.get("chinese") or "").strip()
            if title_cn == title:
                title_cn = ""
            key = (title, r.get("episode"), int(float(r.get("from") or 0)))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "sim": round(sim, 1),
                "title": title,
                "title_cn": title_cn,
                "episode": r.get("episode"),
                "from": r.get("from"),
                "to": r.get("to"),
                "site_url": anilist.get("siteUrl") or "",
                "uncertain": sim < 90,
            })
        return {"engine": "tracemoe", "results": results}

    # ---------------- 图床（SerpApi 需要公网 URL） ----------------

    @staticmethod
    async def _upload_litterbox(client: httpx.AsyncClient, img_bytes: bytes, filename: str) -> str:
        resp = await client.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "1h"},
            files={"fileToUpload": (filename, img_bytes)},
        )
        resp.raise_for_status()
        url = resp.text.strip()
        if not url.startswith("http"):
            raise RuntimeError(f"图床上传失败: {url[:100]}")
        return url

    # ---------------- 搜索核心 ----------------

    async def _image_bytes(self, image: Image) -> tuple[bytes, str]:
        # URL 型图片优先绕开系统代理直连下载（qpic 等国内地址经海外代理会 TLS 失败）
        if getattr(image, "image_type", None) == "url" and image.image:
            try:
                async with httpx.AsyncClient(
                    trust_env=False,
                    timeout=httpx.Timeout(15.0),
                    follow_redirects=True,
                    headers={"User-Agent": _UA},
                ) as direct:
                    r = await direct.get(image.image)
                    r.raise_for_status()
                    name = image.image.split("?")[0].rstrip("/").split("/")[-1] or "image.jpg"
                    if "." not in name:
                        name += ".jpg"
                    return r.content, name
            except Exception as e:
                logger.warning(f"[image_search] 直连下载图片失败，回退 to_path: {type(e).__name__}: {e}")
        try:
            path = await image.to_path()
        except Exception as e:
            raise RuntimeError(f"图片下载失败({type(e).__name__})，可能是原图链接已过期，请重新发图再试") from e
        p = Path(path)
        return p.read_bytes(), p.name

    async def _run_serpapi(self, client, upload_client, img_bytes: bytes, filename: str,
                          source_url: str | None) -> dict:
        if not self.serpapi_key:
            raise RuntimeError("未配置 SerpApi Key")
        # 图床上传走直连：litterbox 经部分海外代理会返回 412 Invalid uploader，
        # 而 SerpApi 本身仍用海外 client（走代理）。
        url = source_url or await self._upload_litterbox(upload_client, img_bytes, filename)
        return await self._lens_serpapi(client, url)

    def _engine_status(self) -> list[str]:
        """各引擎启用状态（用于结果展示与调试）。"""
        st = []
        st.append("SauceNAO: " + ("启用" if (self.enable_saucenao and self.saucenao_key) else "未启用" + ("(缺key)" if self.enable_saucenao else "")))
        st.append("IQDB: " + ("启用" if self.enable_iqdb else "未启用"))
        st.append("trace.moe: " + ("启用" if self.enable_tracemoe else "未启用"))
        st.append("Gemini谷歌AI搜索: " + ("启用" if (self.enable_gemini and self.gemini_key) else "未启用" + ("(缺key)" if self.enable_gemini else "")))
        st.append("百度识图: " + ("启用" if self.enable_baidu else "未启用"))
        st.append("Yandex识图: " + ("启用" if self.enable_yandex else "未启用"))
        st.append("千问图搜图: " + ("启用" if (self.enable_qwen and self.qwen_key) else "未启用" + ("(缺key)" if self.enable_qwen else "")))
        st.append("DeepSeek图搜: " + ("启用" if (self.enable_deepseek and self.deepseek_key) else "未启用" + ("(缺key)" if self.enable_deepseek else "")))
        st.append("GoogleLens(SerpApi): " + ("启用" if (self.enable_serpapi and self.serpapi_key) else "未启用" + ("(缺key)" if self.enable_serpapi else "")))
        return st

    async def _do_search(self, image: Image, only: set[str] | None = None, channel: str = "tool") -> dict:
        """返回 {llm_text, user_text, thumb}；带 md5 缓存。only=限定引擎子集（如 {'tracemoe'}）。
        channel: "tool"(LLM 工具，受 60s 框架超时约束，追求 <20s) / "command"(命令直达，无超时约束)。
        千问图搜图(约10-30s)按次收费，工具通道默认不跑（enable_qwen_in_tool 可开）。"""
        md5 = None
        # 只有开启结果缓存时才计算 md5：URL 型图片的 hash_image() 需要完整下载一次，
        # 缓存关闭时算了也用不上，等于白白多下载一次（速度损耗）。
        if self.cache_days:
            try:
                md5 = await image.hash_image()
            except Exception:
                pass
        cache_key = ""
        if md5:
            cache_key = md5 + ("|" + ",".join(sorted(only)) if only else "")
            # 工具通道默认跳过千问/DeepSeek，命令通道会跑：两者结果不同，缓存键必须区分，
            # 否则工具通道先缓存后，命令通道会命中并永远看不到完整结果。
            skipped = []
            if channel == "tool" and not self.enable_qwen_in_tool:
                skipped.append("noqwen")
            if channel == "tool" and not self.deepseek_in_tool:
                skipped.append("nodeepseek")
            if channel == "tool" and not self.iqdb_in_tool:
                skipped.append("noiqdb")
            if skipped:
                cache_key += "|" + ",".join(skipped)
        if cache_key:
            cached = self._cache_get(cache_key)
            if cached:
                return cached

        img_bytes, filename = await self._image_bytes(image)
        source_url = image.image if image.image and (image.image.startswith("http://") or image.image.startswith("https://")) else None

        # 网络分流：海外引擎（Gemini/SauceNAO/IQDB/trace.moe/Yandex/GoogleLens）走代理，
        # 国内引擎（千问/百度）默认直连（domestic_direct），避免系统代理绕行海外导致超时/断连。
        proxy_url = (self.proxy or "").strip() or None
        common = dict(timeout=httpx.Timeout(self.timeout), follow_redirects=True, headers={"User-Agent": _UA})
        # 海外 client：显式代理或回退环境变量
        proxy_kwargs = dict(common, trust_env=False, proxy=proxy_url) if proxy_url else dict(common, trust_env=True)
        # 国内直连 client：永不走代理
        direct_kwargs = dict(common, trust_env=False)
        async with httpx.AsyncClient(**proxy_kwargs) as client, httpx.AsyncClient(**direct_kwargs) as direct_client:
            def want(n: str) -> bool:
                return only is None or n in only

            # 引擎分流：国内引擎（千问/百度）默认走直连 client，其余引擎仍用海外 client
            baidu_client = direct_client if self.domestic_direct else client
            qwen_client = direct_client if self.domestic_direct else client

            tasks: dict[str, asyncio.Task] = {}
            if want("saucenao") and self.enable_saucenao and self.saucenao_key:
                tasks["saucenao"] = asyncio.create_task(self._saucenao(client, img_bytes, filename))
            if want("iqdb") and self.enable_iqdb:
                # IQDB 服务端反爬、首次常 45–130s：工具通道默认不跑，命令通道照跑
                if channel != "tool" or self.iqdb_in_tool:
                    tasks["iqdb"] = asyncio.create_task(self._iqdb(client, img_bytes, filename))
            # trace.moe 只对番剧截图有效（官方称 <90% 基本为错误结果）：
            # only 显式包含（搜番命令/工具）时总是跑；only is None（搜图/工具）时默认不跑
            if want("tracemoe") and self.enable_tracemoe and (only is not None or self.tracemoe_in_image_search):
                tasks["tracemoe"] = asyncio.create_task(self._tracemoe(client, img_bytes, filename))
            if want("gemini") and self.enable_gemini and self.gemini_key:
                tasks["gemini"] = asyncio.create_task(self._gemini_lens(client, img_bytes, filename))
            if want("baidu") and self.enable_baidu:
                tasks["baidu"] = asyncio.create_task(self._baidu(baidu_client, img_bytes, filename))
            if want("yandex") and self.enable_yandex:
                tasks["yandex"] = asyncio.create_task(self._yandex(client, img_bytes, filename))
            if want("qwen") and self.enable_qwen and self.qwen_key:
                # 工具通道默认跳过千问（慢且按次收费），保证 <20s 返回；命令通道或显式开启才跑。
                # 工具通道受框架 60s 工具超时约束，时限钳到 50s
                if channel != "tool" or self.enable_qwen_in_tool:
                    budget = float(self.qwen_timeout) if channel != "tool" else min(float(self.qwen_timeout), 50.0)
                    tasks["qwen"] = asyncio.create_task(
                        self._qwen_lens(qwen_client, img_bytes, filename, source_url, budget)
                    )
            if want("deepseek") and self.enable_deepseek and self.deepseek_key:
                # DeepSeek 是国内服务，走直连 client；工具通道默认不跑（含联网图搜，较慢且计费）
                if channel != "tool" or self.deepseek_in_tool:
                    budget = (float(self.deepseek_timeout) if channel != "tool"
                              else min(float(self.deepseek_timeout), 50.0))
                    tasks["deepseek"] = asyncio.create_task(
                        self._deepseek_lens(direct_client, img_bytes, filename, budget)
                    )
            if want("serpapi") and self.enable_serpapi and self.serpapi_key:
                tasks["serpapi"] = asyncio.create_task(
                    self._run_serpapi(client, direct_client, img_bytes, filename, source_url)
                )

            if not tasks:
                return {
                    "llm_text": "以图搜图不可用：所有引擎都未启用或未配置 key（请在插件配置中填写 SauceNAO key）。",
                    "user_text": "⚠️ 以图搜图暂不可用：还没有配置任何搜索引擎，请联系管理员配置 SauceNAO key（saucenao.com 免费注册）。",
                    "thumb": "",
                }

            results: dict[str, dict | Exception] = {}
            try:
                gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
                for name, res in zip(tasks.keys(), gathered):
                    if isinstance(res, Exception):
                        # 包一层，避免 TimeoutException 等 str(e) 为空导致日志/结果无信息
                        results[name] = RuntimeError(f"{type(res).__name__}: {res}")
                        logger.warning(f"[image_search] 引擎 {name} 失败: {type(res).__name__}: {res}")
                    else:
                        results[name] = res
            finally:
                # 框架工具超时取消本协程时，后台任务还在用已关闭的 client 跑
                # （"Cannot send a request, as the client has been closed" / Task exception was never retrieved），
                # 必须取消并回收所有未完成任务
                pending = [t for t in tasks.values() if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    try:
                        await asyncio.gather(*pending, return_exceptions=True)
                    except BaseException:
                        pass

        # 相似度阈值过滤（低于阈值的结果不返回不显示）
        if self.enable_min_sim and self.min_similarity > 0:
            th = float(self.min_similarity)
            sau = results.get("saucenao")
            if isinstance(sau, dict) and sau.get("results"):
                sau["results"] = [r for r in sau["results"] if r["sim"] >= th]
            iq = results.get("iqdb")
            if isinstance(iq, dict) and iq.get("matches"):
                kept = [r for r in iq["matches"] if r["sim"] >= th]
                iq["filtered_out"] = len(iq["matches"]) - len(kept)
                iq["matches"] = kept
            tm = results.get("tracemoe")
            if isinstance(tm, dict) and tm.get("results"):
                kept = [r for r in tm["results"] if r["sim"] >= th]
                tm["filtered_out"] = len(tm["results"]) - len(kept)
                tm["results"] = kept

        # 工具通道跳过千问时在结果里留标记，让主 LLM 知道不是引擎故障
        if (channel == "tool" and not self.enable_qwen_in_tool
                and self.enable_qwen and self.qwen_key
                and (only is None or "qwen" in only)):
            results["qwen_skipped"] = True

        llm_text = self._format_llm(results)
        user_text = self._format_user(results)
        thumb = ""
        sau = results.get("saucenao")
        if isinstance(sau, dict) and sau.get("results"):
            thumb = sau["results"][0].get("thumb", "")
        if not thumb:
            sp = results.get("serpapi")
            if isinstance(sp, dict) and sp.get("thumb"):
                thumb = sp["thumb"]
        if not thumb:
            bd = results.get("baidu")
            if isinstance(bd, dict):
                if bd.get("sources"):
                    thumb = bd["sources"][0].get("thumb", "")
                elif bd.get("matches"):
                    thumb = bd["matches"][0].get("thumb", "")

        out = {"llm_text": llm_text, "user_text": user_text, "thumb": thumb}
        # 空结果不缓存（避免一次失败被缓存 7 天）
        if cache_key and "未返回有效结果" not in llm_text and "未解析到具体条目" not in llm_text:
            self._cache_set(cache_key, out)
        return out

    # ---------------- 命令结果发送（合并转发 / 普通消息） ----------------

    @staticmethod
    def _split_blocks(text: str) -> list[str]:
        """按【分节标题】把结果文本切成转发节点。"""
        blocks: list[str] = []
        cur: list[str] = []
        for line in text.split("\n"):
            if line.startswith("【") and cur:
                blocks.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            blocks.append("\n".join(cur))
        return blocks

    def _supports_qq_forward(self, adapter_info) -> bool:
        """判断当前适配器是否支持 NapCat/OneBot 的合并转发接口。

        只有 QQ 系（QQ / NapCat / OneBot）且真的提供 get_client().send_action()
        时才走合并转发；qq_official 等官方 API 适配器、Telegram / Discord /
        Webchat 等非 QQ 平台一律返回 False，直接走普通消息，避免无谓异常与降级。
        adapter_info 可以是 AdapterInfo 元数据，也可以直接是适配器实例。
        """
        if adapter_info is None:
            return False
        platform = str(getattr(adapter_info, "platform", "") or "").lower()
        name = str(getattr(adapter_info, "name", "") or "").lower()
        # qq_official 走官方 API，没有 NapCat 的 send_action
        if "official" in platform or "official" in name:
            return False
        if not any(k in platform or k in name for k in ("qq", "napcat", "onebot")):
            return False
        adapter = adapter_info
        if not callable(getattr(adapter, "get_client", None)):
            # event.adapter 是 AdapterInfo 元数据，需要通过 adapter_mgr 换真实实例
            mgr = getattr(self.ctx, "adapter_mgr", None)
            get_adapter = getattr(mgr, "get_adapter", None)
            if not callable(get_adapter):
                return False
            try:
                adapter = get_adapter(getattr(adapter_info, "name", "") or "")
            except Exception:
                return False
        get_client = getattr(adapter, "get_client", None)
        if not callable(get_client):
            return False
        try:
            client = get_client()
        except Exception:
            return False
        return client is not None and callable(getattr(client, "send_action", None))

    async def _send_qq_forward(self, event: KiraMessageEvent, blocks: list[str], thumb: str) -> bool:
        """用 NapCat 节点接口发合并转发（对齐 deepwiki 的可用格式）。返回是否成功。"""
        adapter = self.ctx.adapter_mgr.get_adapter(event.adapter.name)
        if adapter is None:
            return False
        get_client = getattr(adapter, "get_client", None)
        client = get_client() if callable(get_client) else None
        if client is None or not callable(getattr(client, "send_action", None)):
            return False
        self_id = str(event.message.self_id or "")
        nick = "以图搜图"

        def make_nodes(fmt: int) -> list[dict]:
            nodes = []
            for b in blocks:
                if fmt == 0:
                    nodes.append({"type": "node", "data": {"name": nick, "uin": self_id, "content": b}})
                else:
                    nodes.append(
                        {"type": "node", "data": {"name": nick, "uin": self_id,
                         "content": [{"type": "text", "data": {"text": b}}]}}
                    )
            if thumb:
                nodes.append(
                    {"type": "node", "data": {"name": nick, "uin": self_id,
                     "content": [{"type": "image", "data": {"file": thumb}}]}}
                )
            return nodes

        sid_parts = event.session.sid.split(":", 2)
        is_group = len(sid_parts) == 3 and sid_parts[1] == "gm"
        try:
            target = int(sid_parts[2]) if len(sid_parts) == 3 and sid_parts[2].isdigit() else int(sid_parts[-1])
        except (ValueError, IndexError) as e:
            logger.warning(f"[image_search] 合并转发目标ID转换失败(sid={event.session.sid}): {e}")
            return False
        if is_group:
            actions = ["send_group_forward_msg", "send_forward_msg"]
            params_base = {"group_id": target}
        else:
            actions = ["send_private_forward_msg", "send_forward_msg"]
            params_base = {"user_id": target}

        last_err: Exception | None = None
        for fmt in (0, 1):
            nodes = make_nodes(fmt)
            for action in actions:
                try:
                    resp = await client.send_action(action, {**params_base, "messages": nodes})
                    if isinstance(resp, dict) and resp.get("status") == "failed":
                        raise RuntimeError(str(resp.get("message") or resp.get("wording") or resp))
                    return True
                except Exception as e:
                    last_err = e
                    logger.warning(f"[image_search] 转发尝试 fmt{fmt} {action} 失败: {e}")
        if last_err:
            logger.warning(f"[image_search] 合并转发全部失败: {last_err}")
        return False

    async def _send_command_result(self, event: KiraMessageEvent, result: dict):
        sid = event.session.sid
        text = result["user_text"]
        thumb = result.get("thumb") if self.send_thumbnail else ""

        # 合并转发是 QQ 系（NapCat/OneBot）专有能力：先做能力探测，
        # 非 QQ 平台/官方 API 适配器直接走普通消息，保证全平台可用。
        if self.send_forward and self._supports_qq_forward(event.adapter):
            try:
                blocks = self._split_blocks(text)
                if await self._send_qq_forward(event, blocks, thumb):
                    return
            except Exception as e:
                logger.warning(f"[image_search] 合并转发异常，降级普通消息: {e}")

        await self._send_plain_text(sid, text, thumb)

    async def _send_plain_text(self, sid: str, text: str, thumb: str):
        """普通消息发送（全平台通用）。

        按 1800 字符在行边界分段：Discord 单条上限 2000、Telegram 4096，
        长结果（多引擎 × 多条）直接发整段可能被平台拒收或截断。
        """
        limit = 1800
        chunks: list[str] = []
        cur = ""
        for line in text.split("\n"):
            # 单行本身超过上限（例如超长标题）也要硬切，保证每段都能发出去
            while len(line) > limit:
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.append(line[:limit])
                line = line[limit:]
            if cur and len(cur) + 1 + len(line) > limit:
                chunks.append(cur)
                cur = line
            else:
                cur = f"{cur}\n{line}" if cur else line
        if cur:
            chunks.append(cur)
        for i, chunk in enumerate(chunks or [""]):
            elements: list = [Text(chunk)]
            if thumb and i == 0:
                elements.append(Image(thumb))
            try:
                await self.ctx.message_processor.send_message_chain(sid, MessageChain(elements))
            except Exception as e:
                # 缩略图 URL 失效不应该让整条文字结果一起丢掉
                if thumb and i == 0:
                    logger.warning(f"[image_search] 带缩略图发送失败，回退纯文本: {type(e).__name__}: {e}")
                    await self.ctx.message_processor.send_message_chain(
                        sid, MessageChain([Text(chunk)])
                    )
                else:
                    raise

    @staticmethod
    def _sau_conf(sim: float) -> str:
        if sim >= 80:
            return ""
        if sim >= 60:
            return "(中置信)"
        return "(低置信,仅供参考)"

    def _format_llm(self, results: dict[str, dict | Exception]) -> str:
        lines = ["以图搜图结果（真实网页标注信息，可信度高于图片caption）:"]
        links: list[str] = []
        has_any = False

        # 优先级：文字结论（千问/Gemini/百度有标题的）→ 番剧 → 百分比匹配（SauceNAO/IQDB）
        # 放在前面是因为这些是 LLM 最应该参考的判断
        qw = results.get("qwen")
        if isinstance(qw, dict) and (qw.get("text") or qw.get("image_matches")):
            has_any = True
            qwtext = qw.get("text") or ""
            if len(qwtext) > 450:
                qwtext = qwtext[:450] + "…"
            if qwtext:
                lines.append(f"[千问图搜图] {qwtext}")
            for title, url in (qw.get("image_matches") or [])[: self.max_results]:
                if title:
                    lines.append(f"[千问图搜图] 相似图网页标题:「{title}」")
                if url:
                    links.append(url)
            for title, url in (qw.get("sources") or [])[: self.max_results]:
                links.append(url)
        elif isinstance(qw, Exception):
            lines.append(f"[千问图搜图] 查询失败: {qw}")
        if results.get("qwen_skipped"):
            lines.append("[千问图搜图] 本次未运行（LLM 工具通道默认关闭千问以保证响应速度，并非故障）")

        dsr = results.get("deepseek")
        if isinstance(dsr, dict) and dsr.get("text"):
            has_any = True
            dtext = dsr["text"]
            if len(dtext) > 450:
                dtext = dtext[:450] + "…"
            lines.append(f"[DeepSeek图搜] {dtext}")
            if not dsr.get("verified", True):
                lines.append("（未使用在线搜图，可能不准确，仅供参考）")
            for title, url in (dsr.get("sources") or [])[: self.max_results]:
                links.append(url)
            for title, image_url in (dsr.get("images") or [])[:2]:
                links.append(image_url)
        elif isinstance(dsr, Exception):
            lines.append(f"[DeepSeek图搜] 查询失败: {dsr}")

        gem = results.get("gemini")
        if isinstance(gem, dict) and gem.get("text"):
            has_any = True
            gtext = gem["text"]
            if len(gtext) > 350:
                gtext = gtext[:350] + "…"
            lines.append(f"[Gemini谷歌AI搜索] {gtext}")
            for title, url in (gem.get("sources") or [])[: self.max_results]:
                links.append(url)
        elif isinstance(gem, Exception):
            lines.append(f"[Gemini谷歌AI搜索] 查询失败: {gem}")

        bd = results.get("baidu")
        if isinstance(bd, dict):
            for g in (bd.get("guesses") or [])[:3]:
                has_any = True
                lines.append(
                    f"[百度识图] 识图猜测图中主体是:「{g}」"
                    "（这是识图引擎的实体猜测，名字通常较准但作品归属可能张冠李戴，请把名字和作品拆开看待）"
                )
            bd_sources, _bd_seen = [], set()
            for s in bd.get("sources") or []:
                key = (s.get("url") or "", (s.get("title") or "").strip())
                if key in _bd_seen:
                    continue
                _bd_seen.add(key)
                bd_sources.append(s)
            for s in bd_sources[: self.max_results]:
                has_any = True
                seg = f"[百度识图] 来源页标题:「{s['title']}」"
                if s["website"]:
                    seg += f" ({s['website']})"
                if s.get("abstract"):
                    seg += f" 摘要:{str(s['abstract'])[:80]}"
                lines.append(seg)
                if s["url"]:
                    links.append(s["url"])
            for m in (bd.get("matches") or [])[: self.max_results]:
                has_any = True
                lines.append(f"[百度识图] 相似图出现于 {m['domain']} 的页面")
                links.append(m["url"])
            if not (bd.get("sources")) and not (bd.get("matches")):
                lines.append("[百度识图] 未找到该图的来源页或相似图")
        elif isinstance(bd, Exception):
            lines.append(f"[百度识图] 查询失败: {bd}")

        yx = results.get("yandex")
        if isinstance(yx, dict):
            tags = yx.get("tags") or []
            # 站点列表在前（包含此图的网页是可靠证据），tags 放最后（AI 猜测，仅作旁证）
            yx_sites, _yx_seen = [], set()
            for s in yx.get("sites") or []:
                title = s.get("title") or ""
                if not title:
                    continue
                key = title.strip().lower()
                if key in _yx_seen:
                    continue
                _yx_seen.add(key)
                yx_sites.append(s)
            for s in yx_sites[: self.max_results]:
                title = s.get("title") or ""
                has_any = True
                seg = f"[Yandex识图] 包含此图的网页:「{title}」"
                if s.get("domain"):
                    seg += f"({s['domain']})"
                lines.append(seg)
                if s.get("url"):
                    links.append(s["url"])
            if tags:
                has_any = True
                lines.append(f"[Yandex识图] 图片内容标签（AI 猜测，可能认错角色，仅作旁证）: 「{'、'.join(tags[:8])}」")
            if tags or yx.get("sites"):
                lines.append("[Yandex识图] 以上标题/标签多为俄语等外文，未翻译，请自行翻译理解，并结合其他引擎结果交叉印证")
        elif isinstance(yx, Exception):
            lines.append(f"[Yandex识图] 查询失败: {yx}")

        sp = results.get("serpapi")
        if isinstance(sp, dict):
            for kg in sp.get("knowledge") or []:
                has_any = True
                lines.append(f"[Google Lens] 识别为: {kg}")
            for title, url in (sp.get("matches") or [])[: self.max_results]:
                has_any = True
                lines.append(f"[Google Lens] 相似页面: {title}")
                links.append(url)
        elif isinstance(sp, Exception):
            lines.append(f"[Google Lens] 查询失败: {sp}")

        tm = results.get("tracemoe")
        if isinstance(tm, dict):
            for r in tm.get("results") or []:
                has_any = True
                ep = f" 第{r['episode']}集" if r.get("episode") else ""
                note = "（疑似，仅供参考）" if r.get("uncertain") else ""
                cn = f"（{r['title_cn']}）" if r.get("title_cn") else ""
                ts = ""
                if r.get("from") is not None:
                    sec = int(r["from"])
                    ts = f" 时间 {sec // 60:02d}:{sec % 60:02d}"
                lines.append(
                    f"[trace.moe] 动画截图出自《{r['title']}》{cn}{ep} "
                    f"(相似度{r['sim']}%){ts}{note}"
                )
        elif isinstance(tm, Exception):
            lines.append(f"[trace.moe] 查询失败: {tm}")

        sau = results.get("saucenao")
        if isinstance(sau, dict):
            for r in sau.get("results") or []:
                has_any = True
                seg = f"[SauceNAO] 相似度{r['sim']:.0f}%"
                if r["characters"]:
                    seg += f" 角色:{r['characters']}"
                if r["material"]:
                    seg += f" 作品:《{r['material']}》"
                if r["title"]:
                    seg += f" 「{r['title']}」"
                if r["member"]:
                    seg += f" 画师:{r['member']}"
                if r["source"]:
                    seg += f" 条目:{r['source']}"
                if r.get("part"):
                    seg += f" 章节:{r['part']}"
                if r.get("merged_count"):
                    seg += f"（同作品另有 {r['merged_count']} 个章节版本）"
                lines.append(seg)
                if r["url"]:
                    links.append(r["url"])
                for u in (r.get("urls") or [])[1:]:
                    links.append(u)
            if not (sau.get("results")):
                raw_count = sau.get("raw_count", 0)
                top = sau.get("top_sim", 0)
                if raw_count and self.enable_min_sim and top < self.min_similarity:
                    lines.append(f"[SauceNAO] 有{raw_count}条匹配但最高仅{top:.0f}%，低于显示阈值{self.min_similarity}%")
                elif raw_count:
                    lines.append(f"[SauceNAO] 有{raw_count}条低置信匹配（最高{top:.0f}%），图中可能不是其库收录的插画")
                else:
                    lines.append("[SauceNAO] 无匹配（图中可能不是插画，或被大幅裁剪/加工）")
        elif isinstance(sau, Exception):
            lines.append(f"[SauceNAO] 查询失败: {sau}")

        iq = results.get("iqdb")
        if isinstance(iq, dict):
            for r in (iq.get("matches") or [])[: self.max_results]:
                has_any = True
                lines.append(f"[IQDB] {r['site']} 图库匹配度{r['sim']:.0f}%")
                links.append(r["url"])
            if not (iq.get("matches")) and iq.get("filtered_out"):
                lines.append(f"[IQDB] 匹配度均低于阈值{self.min_similarity}%，未显示")
        elif isinstance(iq, Exception):
            lines.append(f"[IQDB] 查询失败: {iq}")

        if not has_any:
            lines.append("各引擎均未返回有效结果（可能不是动漫图，或图片太模糊/被裁剪过）。")
        lines.append("(引擎状态: " + "; ".join(self._engine_status()) + ")")
        if links:
            uniq = list(dict.fromkeys(links))[:6]
            lines.append(
                "请综合以上各引擎结果回答用户：外文（俄语/日语/英语）标题自行翻译理解，"
                "多引擎交叉一致的结论最可信；引擎间结论冲突时说明分歧并给出你判断最可能的答案，"
                "都不确定就直说不确定。不要主动罗列链接；"
                "若用户需要来源链接，可从备查链接提供: " + " , ".join(uniq)
            )
        else:
            lines.append(
                "请综合以上各引擎结果回答用户：外文标题自行翻译理解，多引擎交叉印证，"
                "冲突时给出最可能的判断，不确定就直说不确定。"
            )
        return "\n".join(lines)

    def _format_user(self, results: dict[str, dict | Exception]) -> str:
        lines = ["🔍 以图搜图结果："]

        sau = results.get("saucenao")
        if isinstance(sau, dict) and sau.get("results"):
            lines.append("【SauceNAO】")
            for r in sau["results"]:
                seg = f"· {r['sim']:.0f}%{self._sau_conf(r['sim'])}"
                if r["characters"]:
                    seg += f" 角色:{r['characters']}"
                if r["material"]:
                    seg += f" 《{r['material']}》"
                if r["title"]:
                    seg += f" 「{r['title']}」"
                if r["member"]:
                    seg += f" 画师:{r['member']}"
                if r["source"]:
                    seg += f"\n  条目:{r['source']}"
                if r.get("part"):
                    seg += f" 章节:{r['part']}"
                if r.get("merged_count"):
                    seg += f"（同作品另有 {r['merged_count']} 个章节版本）"
                lines.append(seg)
                if r["url"]:
                    lines.append(f"  {r['url']}")
                for u in (r.get("urls") or [])[1:]:
                    lines.append(f"  {u}")

        iq = results.get("iqdb")
        if isinstance(iq, dict) and iq.get("matches"):
            lines.append("【IQDB 匹配】")
            for r in iq["matches"][: self.max_results]:
                lines.append(f"· {r['site']} {r['sim']:.0f}%\n  {r['url']}")

        tm = results.get("tracemoe")
        if isinstance(tm, dict) and tm.get("results"):
            for r in tm["results"]:
                ep = f" 第{r['episode']}集" if r.get("episode") else ""
                note = "（疑似，仅供参考）" if r.get("uncertain") else ""
                cn = f"（{r['title_cn']}）" if r.get("title_cn") else ""
                ts = ""
                if r.get("from") is not None:
                    sec = int(r["from"])
                    ts = f" {sec // 60:02d}:{sec % 60:02d}"
                lines.append(f"【番剧截图】《{r['title']}》{cn}{ep} ({r['sim']}%){ts}{note}")

        sp = results.get("serpapi")
        if isinstance(sp, dict):
            for kg in sp.get("knowledge") or []:
                lines.append(f"【Google】可能是：{kg}")
            for title, url in (sp.get("matches") or [])[: self.max_results]:
                lines.append(f"· {title}\n  {url}")
        elif isinstance(sp, Exception):
            lines.append("【Google Lens】查询失败")

        gem = results.get("gemini")
        if isinstance(gem, dict) and gem.get("text"):
            gtext = gem["text"]
            if len(gtext) > 500:
                gtext = gtext[:500] + "…"
            lines.append(f"【谷歌AI搜索】{gtext}")
            for title, url in (gem.get("sources") or [])[: self.max_results]:
                lines.append(f"  {title}: {url}")

        qw = results.get("qwen")
        if isinstance(qw, dict) and (qw.get("text") or qw.get("image_matches")):
            qwtext = qw.get("text") or ""
            if len(qwtext) > 500:
                qwtext = qwtext[:500] + "…"
            if qwtext:
                lines.append(f"【千问图搜图】{qwtext}")
            for title, url in (qw.get("image_matches") or [])[: self.max_results]:
                if title:
                    lines.append(f"· 相似图:「{title}」")
                    if url:
                        lines.append(f"  {url}")
            for title, url in (qw.get("sources") or [])[: self.max_results]:
                lines.append(f"  {title}: {url}")
        elif isinstance(qw, Exception):
            lines.append(f"【千问图搜图】查询失败: {qw}")

        dsr = results.get("deepseek")
        if isinstance(dsr, dict) and dsr.get("text"):
            dtext = dsr["text"]
            if len(dtext) > 500:
                dtext = dtext[:500] + "…"
            lines.append(f"【DeepSeek图搜】{dtext}")
            if not dsr.get("verified", True):
                lines.append("（未使用在线搜图，可能不准确，仅供参考）")
            for title, url in (dsr.get("sources") or [])[: self.max_results]:
                lines.append(f"  {title}: {url}")
            for title, image_url in (dsr.get("images") or [])[:2]:
                lines.append(f"  [图搜图片] {title}: {image_url}")
        elif isinstance(dsr, Exception):
            lines.append(f"【DeepSeek图搜】查询失败: {dsr}")

        bd = results.get("baidu")
        if isinstance(bd, dict):
            if bd.get("guesses"):
                lines.append("【百度识图】猜测：" + "、".join(f"「{g}」" for g in bd["guesses"][:3]))
            if bd.get("sources"):
                lines.append("【百度识图】")
                for s in bd["sources"][: self.max_results]:
                    seg = f"·「{s['title']}」"
                    if s["website"]:
                        seg += f" — {s['website']}"
                    lines.append(seg)
                    if s["url"]:
                        lines.append(f"  {s['url']}")
            elif bd.get("matches"):
                lines.append("【百度识图】")
                for m in bd["matches"][: self.max_results]:
                    lines.append(f"· 相似图来源({m['domain']})\n  {m['url']}")

        yx = results.get("yandex")
        if isinstance(yx, dict):
            tags = yx.get("tags") or []
            sites = [s for s in (yx.get("sites") or []) if s.get("title")][: self.max_results]
            if tags or sites:
                lines.append("【Yandex识图】")
                for s in sites:
                    seg = f"·「{s.get('title')}」"
                    if s.get("domain"):
                        seg += f" — {s['domain']}"
                    lines.append(seg)
                    if s.get("url"):
                        lines.append(f"  {s['url']}")
                if tags:
                    lines.append("· 标签（仅供参考）：" + "、".join(tags[:8]))

        if len(lines) == 1:
            lines.append("没有搜到有效结果，可以换更清晰的图试试。")
        return "\n".join(lines)

    # ---------------- 图片来源解析 ----------------

    @staticmethod
    def _images_in_chain(chain) -> list[Image]:
        if not chain:
            return []
        imgs = [e for e in chain if isinstance(e, Image)]
        for e in chain:
            if isinstance(e, Reply) and getattr(e, "chain", None):
                imgs.extend(x for x in e.chain if isinstance(x, Image))
        return imgs

    def _record_recent(self, sid: str, chain):
        imgs = [e for e in chain if isinstance(e, Image)] if chain else []
        if not imgs:
            return
        now = time.time()
        lst = [(t, img) for t, img in self._recent.get(sid, []) if now - t < _RECENT_TTL]
        for img in imgs:
            lst.append((now, img))
        self._recent[sid] = lst[-_RECENT_MAX:]
        # 顺带清理完全过期的 sid（list 空则 pop），防止 _recent 无限增长
        for k in [k for k, v in self._recent.items() if not v or now - v[-1][0] >= _RECENT_TTL]:
            self._recent.pop(k, None)

    def _recent_image(self, sid: str) -> Image | None:
        now = time.time()
        lst = [(t, img) for t, img in self._recent.get(sid, []) if now - t < _RECENT_TTL]
        if lst:
            self._recent[sid] = lst
        else:
            self._recent.pop(sid, None)
        return lst[-1][1] if lst else None

    # ---------------- 命令直达通道（0 token） ----------------

    def _is_allowed(self, user_id: str) -> bool:
        return not self.allowed_users or user_id in self.allowed_users

    @staticmethod
    def _match_command(text: str, word: str) -> bool:
        """命令匹配：允许命令后跟任意空白（含全角空格/换行），
        但 /搜图xxx 这种粘连文本不算命中。"""
        if not word or not text:
            return False
        if text == word:
            return True
        return text.startswith(word) and text[len(word):len(word) + 1].isspace()

    @on.im_message(priority=Priority.HIGH)
    async def on_im_message(self, event: KiraMessageEvent, *_):
        try:
            await self._handle_im_message(event)
        except Exception:
            logger.exception("[image_search] im_message 钩子异常")
            # 不 discard，让消息走正常流程

    async def _handle_im_message(self, event: KiraMessageEvent):
        if not self.enabled:
            return
        sid = event.session.sid
        # 轻量记录最近图片，供命令/工具兜底
        try:
            self._record_recent(sid, event.message.chain)
        except Exception:
            pass

        text = "".join(e.text for e in (event.message.chain or []) if isinstance(e, Text)).strip()
        if not text:
            return

        # 匹配命令：搜图（全引擎）/ 搜番（仅 trace.moe）
        only: set[str] | None = None
        if self.enable_command and any(self._match_command(text, w) for w in self.command_words):
            only = None
        elif self.enable_command_anime and any(self._match_command(text, w) for w in self.command_words_anime):
            only = {"tracemoe"}
        else:
            return

        sender = event.message.sender
        user_id = str(sender.user_id) if sender else ""
        if not self._is_allowed(user_id):
            await self.ctx.message_processor.send_message_chain(
                sid, MessageChain([Text("抱歉，你没有权限使用以图搜图。")])
            )
            event.discard(force=True)
            event.stop()
            return

        # 解析目标图片：本条消息自带/回复引用 → 最近图片缓存
        image = None
        imgs = self._images_in_chain(event.message.chain)
        if imgs:
            image = imgs[-1]
        if image is None:
            image = self._recent_image(sid)
        if image is None:
            await self.ctx.message_processor.send_message_chain(
                sid, MessageChain([Text("没找到最近的图片，请把图和命令一起发，或回复图片消息使用。")])
            )
            event.discard(force=True)
            event.stop()
            return

        # 两段式发送：快速引擎结果先回（通常 <20s），千问图搜图（约10-30秒）随后补发，
        # 兼顾速度与准确度。仅 /搜图（全引擎）且千问图搜图可用时启用。
        qwen_deep = (
            only is None
            and self.enable_qwen and self.qwen_key and self.enable_qwen_image_search
        )
        fast_only = {"saucenao", "iqdb", "tracemoe", "gemini", "baidu", "serpapi", "yandex"} if qwen_deep else only
        # qwen_deep 快速通道显式含 tracemoe 会绕过 tracemoe_in_image_search 配置
        # （_do_search 里 only 非 None 即放行），按配置剔除；搜番命令的 {"tracemoe"} 分支不动
        if qwen_deep and fast_only is not None and not self.tracemoe_in_image_search:
            fast_only.discard("tracemoe")
        # 千问深搜与快速搜索并行起跑（不再串行等快速结果），快速结果发送后直接 await 补发
        qwen_task: asyncio.Task | None = None
        if qwen_deep:
            qwen_task = asyncio.create_task(self._do_search(image, only={"qwen"}, channel="command"))
        try:
            result = await self._do_search(image, only=fast_only, channel="command")
            if qwen_deep:
                result = dict(result)
                result["user_text"] += "\n⏳ 千问图搜图深搜中（约10-30秒），结果稍后补发…"
            await self._send_command_result(event, result)
        except Exception as e:
            if qwen_task is not None:
                qwen_task.cancel()
                # 回收被取消的深搜任务，避免 "Task exception was never retrieved"
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await qwen_task
            logger.exception("[image_search] 命令搜索失败")
            await self.ctx.message_processor.send_message_chain(
                sid, MessageChain([Text(f"以图搜图失败了：{e}")])
            )
            event.discard(force=True)
            event.stop()
            return
        if qwen_task is not None:
            try:
                qr = await qwen_task
                qtext = qr.get("user_text", "")
                if "查询失败" in qtext or "未启用" in qtext:
                    logger.warning(f"[image_search] 千问深搜未获结果: {qtext[:100]}")
                    await self.ctx.message_processor.send_message_chain(
                        sid, MessageChain([Text("千问图搜图深搜超时或失败，以上快速引擎结果仅供参考。")])
                    )
                else:
                    await self.ctx.message_processor.send_message_chain(sid, MessageChain([Text(qtext)]))
            except Exception as e:
                logger.warning(f"[image_search] 千问深搜异常: {type(e).__name__}: {e}")
        event.discard(force=True)
        event.stop()

    # ---------------- LLM 工具通道 ----------------

    async def _resolve_tool_image(self, event: KiraMessageBatchEvent, image_index: int, file_path: str):
        """按 file_path → 批量消息第 N 张 → 最近图片缓存 的顺序解析目标图片。"""
        # 1) file_path 优先
        if file_path:
            p = Path(file_path)
            if not p.exists():
                alt = Path(get_data_path()).parent / file_path
                p = alt if alt.exists() else p
            if p.is_file():
                return Image(str(p)), ""
            if p.exists():
                return None, f"不是图片文件（路径是目录）: {file_path}"
            return None, f"找不到图片文件: {file_path}"

        # 2) 批量消息里倒序找第 N 张
        imgs: list[Image] = []
        for msg in event.messages or []:
            imgs.extend(self._images_in_chain(msg.chain))
        idx = max(1, int(image_index or 1))
        if len(imgs) >= idx:
            return imgs[-idx], ""

        # 3) 最近图片缓存兜底
        image = self._recent_image(event.sid)
        if image is not None:
            return image, ""
        return None, "当前会话没有找到可搜索的图片，请让用户先发送图片。"

    @register.tool(
        name="search_image",
        description=(
            "以图搜图：用 SauceNAO、DeepSeek、百度、Yandex、IQDB 等 9 个引擎搜索该图或相似图的网页，"
            "返回角色名、作品、画师、出处链接等真实标注，比图片 caption 准确。"
            "用户问图片里是谁/什么角色/原图/画师/来源时必须调用；番剧截图问第几集改用 search_anime。"
            "即使 caption 看起来确定也要调用（caption 可能编错角色名）。"
            "仅问画面内容（几个人、在干嘛）时不需要调用。"
        ),
        params={
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "搜第几张图：1=最近一张（默认），2=倒数第二张，依此类推",
                    "default": 1,
                },
                "file_path": {
                    "type": "string",
                    "description": "可选。上下文中 [Image ..., file_path: xxx] 里的路径，指定后优先使用该文件",
                    "default": "",
                },
            },
            "required": [],
        },
    )
    async def search_image(self, event: KiraMessageBatchEvent, image_index: int = 1, file_path: str = "") -> str:
        if not self.enabled or not self.enable_tool:
            return "以图搜图插件未启用。"

        image, err = await self._resolve_tool_image(event, image_index, file_path)
        if image is None:
            return err

        try:
            result = await self._do_search(image)
            return result["llm_text"]
        except Exception as e:
            logger.exception("[image_search] 工具搜索失败")
            return f"以图搜图失败: {e}"

    @register.tool(
        name="search_anime",
        description=(
            "识别动画截图出自哪部番剧、第几集、第几秒（trace.moe 专用）。"
            "用户发番剧/动画截图问出处、哪部番、第几集时调用。"
            "同人图、立绘、插画、表情包不要调用（改用 search_image）。"
        ),
        params={
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "搜第几张图：1=最近一张（默认），2=倒数第二张，依此类推",
                    "default": 1,
                },
                "file_path": {
                    "type": "string",
                    "description": "可选。上下文中 [Image ..., file_path: xxx] 里的路径，指定后优先使用该文件",
                    "default": "",
                },
            },
            "required": [],
        },
    )
    async def search_anime(self, event: KiraMessageBatchEvent, image_index: int = 1, file_path: str = "") -> str:
        if not self.enabled or not self.enable_tool_anime:
            return "搜番工具未启用。"

        image, err = await self._resolve_tool_image(event, image_index, file_path)
        if image is None:
            return err

        try:
            result = await self._do_search(image, only={"tracemoe"})
            return result["llm_text"]
        except Exception as e:
            logger.exception("[image_search] 搜番失败")
            return f"搜番失败: {e}"

    # ---------------- Prompt 注入 ----------------

    @on.llm_request()
    async def inject(self, event, req: LLMRequest, tag_set, *_):
        if not self.enabled:
            return
        hints = []
        if self.enable_tool:
            hints.append(
                "图片身份/出处类问题（谁、什么角色、哪部番、原图、画师、来源）必须调用 search_image 以图搜图，"
                "不要只信图片 caption——caption 可能认错角色。画面内容类问题不必调用。"
            )
        if self.enable_tool_anime:
            hints.append("动画截图问出处/第几集时调用 search_anime。")
        if not hints:
            return
        hint = "\n" + "\n".join(hints)
        for p in req.system_prompt:
            if p.name == "tools":
                p.content += hint
                return
        # 找不到 tools 段时兜底追加到最后一个 system_prompt，避免静默丢弃
        if req.system_prompt:
            req.system_prompt[-1].content += hint

