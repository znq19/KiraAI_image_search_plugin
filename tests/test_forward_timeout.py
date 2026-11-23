"""image_search：合并转发可配置超时 + 超时降级为普通消息。

    python3 tests/test_forward_timeout.py
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# 框架日志按 cwd/data 落盘，切到临时目录避免在仓库留下产物
_TMP = tempfile.mkdtemp(prefix="is_test_")
os.chdir(_TMP)
os.makedirs(os.path.join(_TMP, "data"), exist_ok=True)

CORE = "/var/minis/workspace/com3d_check/kira"
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import asyncio
import time
import types

src = (ROOT / "main.py").read_text(encoding="utf-8")
mod = types.ModuleType("is_mod")
exec(compile(src, "main.py", "exec"), mod.__dict__)

FAILS = []


def check(label, cond, detail=""):
    print(("  ✓ " if cond else "  ✗ ") + label + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(label)
    return cond


class FakeClient:
    def __init__(self, mode="ok"):
        self.calls = []
        self.timeouts = []
        self.mode = mode

    async def send_action(self, action, params, timeout=10.0):
        self.calls.append((action, params))
        self.timeouts.append(timeout)
        if self.mode == "timeout":
            raise asyncio.TimeoutError(f"请求 {action} 超时")
        if self.mode == "conn":
            raise ConnectionError(f"NapCat WebSocket 未连接，无法执行 {action}")
        return {"status": "ok", "retcode": 0, "data": {"message_id": 1}}


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _Sess:
    sid = "qq:gm:12345"


class FakeEvent:
    session_type = "group"
    group_id = 12345
    user_id = 999
    message_id = "m1"
    session = _Sess()
    message = type("M", (), {"self_id": "10001", "message_id": "m1"})()
    reply = None
    adapter = type("A", (), {"name": "qq"})()
    sid = "group_12345"


class FakeProcessor:
    def __init__(self):
        self.sent = []

    async def send_message_chain(self, sid, chain):
        self.sent.append(chain)
        return {"status": "ok", "retcode": 0}


class _Adapter:
    def __init__(self, client):
        self._client = client

    def get_client(self):
        return self._client


class _AdapterMgr:
    def __init__(self, client):
        self._adapter = _Adapter(client)

    def get_adapter(self, name):
        return self._adapter


class FakeCtx:
    def __init__(self, client):
        self.message_processor = FakeProcessor()
        self.adapter_mgr = _AdapterMgr(client)
        self._client = client

    def get_client(self, *a, **kw):
        return self._client


def make_plugin(client, forward_timeout=60.0):
    p = mod.ImageSearchPlugin.__new__(mod.ImageSearchPlugin)
    p.ctx = FakeCtx(client)
    p.forward_timeout = forward_timeout
    p.send_thumbnail = False
    p.send_forward = True
    return p


def make_items():
    return [
        {"title": "测试标题一", "url": "https://example.com/1",
         "source": "pixiv", "thumb": ""},
        {"title": "测试标题二", "url": "https://example.com/2",
         "source": "danbooru", "thumb": ""},
    ]


print("═══ 1) 正常：转发成功，只调一次，带 60s")
c = FakeClient("ok")
p = make_plugin(c)
ok = run(p._send_qq_forward(FakeEvent(), make_items(), ""))
check("返回 True", ok is True, repr(ok))
check("只调用 1 次", len(c.calls) == 1, str([a for a, _ in c.calls]))
check("★ 显式传 60s", c.timeouts == [60.0], str(c.timeouts))

print()
print("═══ 2) ★ 超时 ⇒ 换 action 后仍失败，最终返回 False 供降级")
c2 = FakeClient("timeout")
p2 = make_plugin(c2)
ok2 = run(p2._send_qq_forward(FakeEvent(), make_items(), ""))
check("★ 超时最终返回 False（调用方降级）", ok2 is False, repr(ok2))
check("尝试了多个 action（原设计保留）", len(c2.calls) > 1, str([a for a, _ in c2.calls]))

print()
print("═══ 3) 明确故障（连接断开）⇒ 同样返回 False")
c3 = FakeClient("conn")
p3 = make_plugin(c3)
ok3 = run(p3._send_qq_forward(FakeEvent(), make_items(), ""))
check("返回 False", ok3 is False, repr(ok3))

print()
print("═══ 4) forward_timeout 可配置")
c4 = FakeClient("ok")
p4 = make_plugin(c4, forward_timeout=25)
run(p4._send_qq_forward(FakeEvent(), make_items(), ""))
check("传入 25s", c4.timeouts == [25.0], str(c4.timeouts))

c5 = FakeClient("ok")
p5 = make_plugin(c5, forward_timeout=0)
run(p5._send_qq_forward(FakeEvent(), make_items(), ""))
check("0 ⇒ 用框架默认 10s", c5.timeouts == [10.0], str(c5.timeouts))

print()
print("═══ 5) 老核心（无 timeout 形参）自动退回")
class OldClient:
    def __init__(self):
        self.calls = []

    async def send_action(self, action, params):
        self.calls.append(action)
        return {"status": "ok", "retcode": 0}


c6 = OldClient()
p6 = make_plugin(c6)
ok6 = run(p6._send_qq_forward(FakeEvent(), make_items(), ""))
check("老核心下仍发送成功", ok6 is True, repr(ok6))

print()
if FAILS:
    print(f"✗ 失败 {len(FAILS)} 项: {FAILS}")
    sys.exit(1)
print("✓ 全部通过")
