"""Feishu/Lark platform adapter for tyagent.

Supports:
- WebSocket long connection
- Direct-message and group @mention-gated text receive/send
- Inbound image/file/audio/video caching with correct extensions
- Outbound image/file upload (send_photo, send_document)
- QR scan-to-create onboarding
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tyagent.config import default_home
from tyagent.config_field import ConfigField
from tyagent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

# QR onboarding constants
_ONBOARD_ACCOUNTS_URLS = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
_ONBOARD_OPEN_URLS = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}
_REGISTRATION_PATH = "/oauth/v1/app/registration"
_ONBOARD_REQUEST_TIMEOUT_S = 10

# Feishu message type constants
_MSG_TYPE_TEXT = "text"
_MSG_TYPE_IMAGE = "image"
_MSG_TYPE_FILE = "file"
_MSG_TYPE_AUDIO = "audio"
_MSG_TYPE_MEDIA = "media"
_MSG_TYPE_POST = "post"

# Dedup TTL
_DEDUP_TTL_SECONDS = 24 * 60 * 60

# Feishu processing reaction constants
_FEISHU_REACTION_IN_PROGRESS = "Typing"        # ⌨️ badge — processing
_FEISHU_PROCESSING_REACTION_CACHE_SIZE = 100   # LRU bound for reaction handles

# Content-Type to extension mapping (supplements mimetypes)
_CT_EXT_OVERRIDES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/aac": ".aac",
    "audio/m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/ogg": ".ogv",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/zip": ".zip",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
}

# ---------------------------------------------------------------------------
# Markdown rendering helpers (inspired by hermes-agent)
# ---------------------------------------------------------------------------

# Detect markdown syntax hints to decide whether to send as post (rendered)
# or plain text. Matches: headers, lists, code blocks, inline code, bold,
# italic, strikethrough, underline, links, blockquotes.
_MARKDOWN_HINT_RE = re.compile(
    r"(^#{1,6}\s)|(^\s*[-*]\s)|(^\s*\d+\.\s)|(^\s*---+\s*$)|(```)|(`[^`\n]+`)|"
    r"(\*\*[^*\n].+?\*\*)|(~~[^~\n].+?~~)|(<u>.+?</u>)|(\*[^*\n]+\*)|"
    r"(\[[^\]]+\]\([^)]+\))|(^>\s)",
    re.MULTILINE,
)
# Detect markdown tables: a line starting with | followed by a separator line.
# Feishu post-type 'md' elements do not render tables, so we force text mode.
_MARKDOWN_TABLE_RE = re.compile(r"^\|.*\|\n\|[-|: ]+\|", re.MULTILINE)
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")

# Match a complete markdown table: header row + separator row + optional data rows.
# A table ends at a blank line, a non-table line, or end of text.
_TABLE_BLOCK_RE = re.compile(
    r"^(\|(?:[^|\n]*\|)+\n\|[-|: ]+\|\n?)(?:\|(?:[^|\n]*\|)+\n?)*",
    re.MULTILINE,
)


def _escape_markdown_text(text: str) -> str:
    """Escape markdown special characters so they render literally."""
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _build_markdown_post_payload(content: str) -> str:
    """Build a Feishu post payload with markdown rendering enabled."""
    rows = _build_markdown_post_rows(content)
    return json.dumps({"zh_cn": {"content": rows}}, ensure_ascii=False)


def _build_markdown_post_rows(content: str) -> List[List[Dict[str, str]]]:
    """Build Feishu post rows while isolating fenced code blocks.

    Feishu's `md` renderer can swallow trailing content when a fenced code
    block appears inside one large markdown element. Split the reply at real
    fence lines so prose before/after the code block remains visible while
    code stays in a dedicated row.

    Uses fence char + length matching to avoid mis-interpreting inner fences
    (e.g. a ``` line inside a `````` block is content, not a close fence).
    """
    if not content:
        return [[{"tag": "md", "text": ""}]]
    if "```" not in content and "~~~" not in content:
        return [[{"tag": "md", "text": content}]]

    rows: List[List[Dict[str, str]]] = []
    current: List[str] = []
    in_code_block = False
    fence_char = ""  # '`' or '~'
    fence_len = 0    # length of opening fence

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        segment = "\n".join(current)
        if segment.strip():
            rows.append([{"tag": "md", "text": segment}])
        current = []

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        leading_spaces = len(raw_line) - len(raw_line.lstrip(" "))

        is_fence = False
        if in_code_block:
            # Close fence: same char, same or longer length, indent <= 3
            if leading_spaces <= 3:
                m = re.match(rf"^({re.escape(fence_char)}{{3,}})\s*$", stripped)
                if m and len(m.group(1)) >= fence_len:
                    is_fence = True
        else:
            # Open fence: 3+ backticks or tildes, optional lang tag
            m = re.match(r"^(```+|~~~+)([^`\n]*)\s*$", stripped)
            if m:
                is_fence = True
                fence_char = m.group(1)[0]  # '`' or '~'
                fence_len = len(m.group(1))

        if is_fence:
            if not in_code_block:
                _flush()
            current.append(raw_line)
            in_code_block = not in_code_block
            if not in_code_block:
                fence_char = ""
                fence_len = 0
                _flush()
            continue

        current.append(raw_line)

    _flush()
    return rows or [[{"tag": "md", "text": content}]]


def _convert_tables_to_code_blocks(text: str) -> str:
    """Replace complete markdown table blocks with fenced code blocks.

    Feishu post 'md' tags do not render markdown tables (they show as blank).
    Wrapping tables in a code fence preserves the visual structure as monospace
    text while keeping the message as 'post' type so other markdown (bold, lists,
    code blocks) renders correctly.
    """
    def _wrap(m: re.Match) -> str:
        table = m.group(0).rstrip("\n")
        return f"```\n{table}\n```"

    return _TABLE_BLOCK_RE.sub(_wrap, text)


def _build_outbound_payload(text: str) -> tuple[str, str]:
    """Determine msg_type and payload for sending text to Feishu.

    If the text contains markdown syntax, send as 'post' with md tag
    so Feishu renders it (bold, italic, code blocks, etc.).
    Otherwise send as plain 'text' for efficiency.
    """
    # Feishu post-type 'md' elements do not render markdown tables; convert
    # table blocks to fenced code blocks so they render as monospace text
    # while keeping the message as 'post' for other markdown formatting.
    if _MARKDOWN_TABLE_RE.search(text):
        text = _convert_tables_to_code_blocks(text)
        return "post", _build_markdown_post_payload(text)
    if _MARKDOWN_HINT_RE.search(text):
        return "post", _build_markdown_post_payload(text)
    return "text", json.dumps({"text": text}, ensure_ascii=False)


try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        GetFileRequest,
        GetImageRequest,
        GetMessageResourceRequest,
        ReplyMessageRequest,
        ReplyMessageRequestBody,
    )
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    from lark_oapi.ws import Client as FeishuWSClient

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None  # type: ignore[assignment]
    FeishuWSClient = None  # type: ignore[assignment, misc]
    FEISHU_DOMAIN = None  # type: ignore[assignment]
    LARK_DOMAIN = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# QR scan-to-create onboarding
# ---------------------------------------------------------------------------

def _accounts_base_url(domain: str) -> str:
    return _ONBOARD_ACCOUNTS_URLS.get(domain, _ONBOARD_ACCOUNTS_URLS["feishu"])


def _onboard_open_base_url(domain: str) -> str:
    return _ONBOARD_OPEN_URLS.get(domain, _ONBOARD_OPEN_URLS["feishu"])


def _post_registration(base_url: str, body: Dict[str, str]) -> dict:
    """POST form-encoded data to the registration endpoint, return parsed JSON."""
    url = f"{base_url}{_REGISTRATION_PATH}"
    data = urlencode(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_bytes = exc.read()
        if body_bytes:
            try:
                return json.loads(body_bytes.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                raise exc from None
        raise


def _init_registration(domain: str = "feishu") -> None:
    """Verify the environment supports client_secret auth."""
    base_url = _accounts_base_url(domain)
    res = _post_registration(base_url, {"action": "init"})
    methods = res.get("supported_auth_methods") or []
    if "client_secret" not in methods:
        raise RuntimeError(
            f"Feishu / Lark registration environment does not support client_secret auth. "
            f"Supported: {methods}"
        )


def _begin_registration(domain: str = "feishu") -> dict:
    """Start the device-code flow."""
    base_url = _accounts_base_url(domain)
    res = _post_registration(base_url, {
        "action": "begin",
        "archetype": "PersonalAgent",
        "auth_method": "client_secret",
        "request_user_info": "open_id",
    })
    device_code = res.get("device_code")
    if not device_code:
        raise RuntimeError("Feishu / Lark registration did not return a device_code")
    qr_url = res.get("verification_uri_complete", "")
    if "?" in qr_url:
        qr_url += "&from=tyagent&tp=tyagent"
    else:
        qr_url += "?from=tyagent&tp=tyagent"
    return {
        "device_code": device_code,
        "qr_url": qr_url,
        "user_code": res.get("user_code", ""),
        "interval": res.get("interval") or 5,
        "expire_in": res.get("expire_in") or 600,
    }


def _poll_registration(
    *,
    device_code: str,
    interval: int,
    expire_in: int,
    domain: str = "feishu",
) -> Optional[dict]:
    """Poll until the user scans the QR code, or timeout/denial."""
    deadline = time.time() + expire_in
    current_domain = domain
    domain_switched = False
    poll_count = 0

    while time.time() < deadline:
        base_url = _accounts_base_url(current_domain)
        try:
            res = _post_registration(base_url, {
                "action": "poll",
                "device_code": device_code,
                "tp": "ob_app",
            })
        except (URLError, OSError, json.JSONDecodeError):
            time.sleep(interval)
            continue

        poll_count += 1
        if poll_count == 1:
            print("  Fetching configuration results...", end="", flush=True)
        elif poll_count % 6 == 0:
            print(".", end="", flush=True)

        # Domain auto-detection
        user_info = res.get("user_info") or {}
        tenant_brand = user_info.get("tenant_brand")
        if tenant_brand == "lark" and not domain_switched:
            current_domain = "lark"
            domain_switched = True

        # Success
        if res.get("client_id") and res.get("client_secret"):
            if poll_count > 0:
                print()
            return {
                "app_id": res["client_id"],
                "app_secret": res["client_secret"],
                "domain": current_domain,
                "open_id": user_info.get("open_id"),
            }

        # Terminal errors
        error = res.get("error", "")
        if error in ("access_denied", "expired_token"):
            if poll_count > 0:
                print()
            logger.warning("[Feishu onboard] Registration %s", error)
            return None

        time.sleep(interval)

    if poll_count > 0:
        print()
    logger.warning("[Feishu onboard] Poll timed out after %ds", expire_in)
    return None


try:
    import qrcode as _qrcode_mod
except ImportError:
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Try to render a QR code in the terminal."""
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


def probe_bot(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Verify bot connectivity via /open-apis/bot/v3/info."""
    base_url = _onboard_open_base_url(domain)
    try:
        token_data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
        token_req = Request(
            f"{base_url}/open-apis/auth/v3/tenant_access_token/internal",
            data=token_data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(token_req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            token_res = json.loads(resp.read().decode("utf-8"))

        access_token = token_res.get("tenant_access_token")
        if not access_token:
            return None

        bot_req = Request(
            f"{base_url}/open-apis/bot/v3/info",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(bot_req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            bot_res = json.loads(resp.read().decode("utf-8"))

        if bot_res.get("code") != 0:
            return None
        bot = bot_res.get("bot") or bot_res.get("data", {}).get("bot") or {}
        return {
            "bot_name": bot.get("bot_name") or bot.get("app_name"),
            "bot_open_id": bot.get("open_id"),
        }
    except (URLError, OSError, KeyError, json.JSONDecodeError) as exc:
        logger.debug("[Feishu onboard] HTTP probe failed: %s", exc)
        return None


def qr_register(
    *,
    initial_domain: str = "feishu",
    timeout_seconds: int = 600,
) -> Optional[dict]:
    """Run the Feishu / Lark scan-to-create QR registration flow.

    Returns on success:
        {
            "app_id": str,
            "app_secret": str,
            "domain": "feishu" | "lark",
            "open_id": str | None,
            "bot_name": str | None,
            "bot_open_id": str | None,
        }
    """
    try:
        return _qr_register_inner(initial_domain=initial_domain, timeout_seconds=timeout_seconds)
    except (RuntimeError, URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("[Feishu onboard] Registration failed: %s", exc)
        return None


def _qr_register_inner(
    *,
    initial_domain: str,
    timeout_seconds: int,
) -> Optional[dict]:
    """Run init -> begin -> poll -> probe."""
    print("  Connecting to Feishu / Lark...", end="", flush=True)
    _init_registration(initial_domain)
    begin = _begin_registration(initial_domain)
    print(" done.")

    print()
    qr_url = begin["qr_url"]
    if _render_qr(qr_url):
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {qr_url}")
    else:
        print(f"  Open this URL in Feishu / Lark on your phone:\n\n  {qr_url}\n")

    print("  Waiting for you to scan and authorize...")
    result = _poll_registration(
        device_code=begin["device_code"],
        interval=begin["interval"],
        expire_in=min(begin["expire_in"], timeout_seconds),
        domain=initial_domain,
    )
    if not result:
        return None

    print("  Bot created successfully! Probing connectivity...")
    probe = probe_bot(result["app_id"], result["app_secret"], result["domain"])
    if probe:
        print(f"  Bot name: {probe.get('bot_name', 'unknown')}")
    else:
        print("  Warning: Could not verify bot connectivity.")

    return {
        **result,
        **(probe or {}),
    }


# ---------------------------------------------------------------------------
# WS client thread runner
# ---------------------------------------------------------------------------

def _run_ws_client(ws_client: Any, adapter: Any) -> None:
    """Run the Lark WS client in its own thread-local event loop."""
    import lark_oapi.ws.client as ws_client_module

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    original_loop = getattr(ws_client_module, "loop", None)
    ws_client_module.loop = loop
    adapter._ws_thread_loop = loop

    logger.info("WS client thread started")
    try:
        ws_client.start()
        logger.info("WS client.start() returned normally (unexpected)")
    except Exception:
        logger.warning("Feishu WS client exited with error", exc_info=True)
    finally:
        logger.info("WS client thread cleaning up")
        ws_client_module.loop = original_loop
        adapter._ws_thread_loop = None
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------

def _guess_extension_from_content_type(content_type: Optional[str]) -> str:
    """Guess file extension from Content-Type header."""
    if not content_type:
        return ""
    # Clean up content type (remove charset, etc.)
    ct = content_type.split(";")[0].strip().lower()
    # Check overrides first
    if ct in _CT_EXT_OVERRIDES:
        return _CT_EXT_OVERRIDES[ct]
    # Fall back to mimetypes
    ext = mimetypes.guess_extension(ct)
    return ext or ""


def _guess_extension_from_filename(filename: Optional[str]) -> str:
    """Guess extension from filename."""
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _resolve_extension(
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    default: str = ".bin",
) -> str:
    """Resolve the best file extension from Content-Type and/or filename."""
    # Prefer filename extension if available
    from_filename = _guess_extension_from_filename(filename)
    if from_filename:
        return from_filename
    # Fall back to Content-Type
    from_ct = _guess_extension_from_content_type(content_type)
    if from_ct:
        return from_ct
    return default


# ---------------------------------------------------------------------------
# Feishu Platform Adapter
# ---------------------------------------------------------------------------

class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark platform adapter."""

    config_schema = {
        "enabled": ConfigField(bool, default=False, doc="启用飞书平台"),
        "extra": {
            "connection": {
                "app_id": ConfigField(str, required=True, doc="飞书开放平台应用的 App ID"),
                "app_secret": ConfigField(str, required=True, secret=True,
                                           doc="飞书开放平台应用的 App Secret"),
                "domain": ConfigField(str, default="feishu", choices=["feishu", "lark"],
                                      doc="API 域名（feishu=国内, lark=海外）"),
            },
            "event_subscription": {
                "encrypt_key": ConfigField(str, default="",
                                           doc="飞书事件订阅的加密密钥（不需要则留空）"),
                "verification_token": ConfigField(str, default="",
                                                  doc="飞书事件订阅的验证令牌（不需要则留空）"),
            },
            "behavior": {
                "group_policy": ConfigField(str, default="mention",
                                            choices=["open", "mention", "disabled"],
                                            doc="群聊消息处理策略（open=全部, mention=仅@, disabled=关闭）"),
            },
        },
    }

    def __init__(self, config: Any, home_dir: Optional[Path] = None):
        super().__init__(config, "feishu", home_dir=home_dir)
        if not FEISHU_AVAILABLE:
            raise ImportError(
                "lark_oapi is required for Feishu support. "
                "Install it with: uv add lark-oapi"
            )

        self.app_id: str = (
            (config.extra.get("connection", {}) or {}).get("app_id", "")
            or config.extra.get("app_id", "")
        )
        self.app_secret: str = (
            (config.extra.get("connection", {}) or {}).get("app_secret", "")
            or config.extra.get("app_secret", "")
        )
        self.domain: str = (
            (config.extra.get("connection", {}) or {}).get("domain", "feishu")
            or config.extra.get("domain", "feishu")
        )
        self.encrypt_key: str = (
            (config.extra.get("event_subscription", {}) or {}).get("encrypt_key", "")
            or config.extra.get("encrypt_key", "")
        )
        self.verification_token: str = (
            (config.extra.get("event_subscription", {}) or {}).get("verification_token", "")
            or config.extra.get("verification_token", "")
        )
        self.group_policy: str = (
            (config.extra.get("behavior", {}) or {}).get("group_policy", "mention")
            or config.extra.get("group_policy", "mention")
        )  # "open" | "mention" | "disabled"

        if not self.app_id or not self.app_secret:
            raise ValueError("Feishu adapter requires app_id and app_secret in config.extra")

        self._client: Any = None
        self._ws_client: Any = None
        self._ws_future: Any = None
        self._ws_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot_open_id: Optional[str] = None
        self._bot_name: str = ""
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Cache dir for media and dedup state
        if self.home_dir is None:
            logger.warning("FeishuAdapter created without home_dir — using default %s. Profile isolation may be broken.", default_home)
            _base = default_home
        else:
            _base = self.home_dir
        self._cache_dir = _base / "cache" / "feishu"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Persistent message dedup state
        self._dedup_path = self._cache_dir / "seen_message_ids.json"
        self._dedup_lock = threading.Lock()
        self._dedup: Dict[str, float] = self._load_dedup()

        # Processing reaction tracking (message_id → reaction_id)
        self._pending_processing_reactions: "OrderedDict[str, str]" = OrderedDict()

    def _load_dedup(self) -> Dict[str, float]:
        """Load deduplication state from disk, pruning expired entries."""
        if not self._dedup_path.exists():
            logger.info("No dedup state file found, starting fresh")
            return {}
        try:
            with open(self._dedup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Dedup state file corrupted, starting fresh")
                return {}
            cutoff = time.time() - _DEDUP_TTL_SECONDS
            # Prune expired entries on load
            pruned = {k: float(v) for k, v in data.items() if float(v) > cutoff}
            dropped = len(data) - len(pruned)
            if dropped:
                logger.info("Loaded %d dedup entries, pruned %d expired", len(pruned), dropped)
            else:
                logger.info("Loaded %d dedup entries", len(pruned))
            return pruned
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to load dedup state: %s", exc)
            return {}

    def _save_dedup(self) -> None:
        """Save deduplication state to disk."""
        try:
            with open(self._dedup_path, "w", encoding="utf-8") as f:
                json.dump(self._dedup, f, ensure_ascii=False)
            logger.debug("Saved %d dedup entries to %s", len(self._dedup), self._dedup_path)
        except OSError as exc:
            logger.warning("Failed to save dedup state: %s", exc)

    def _is_duplicate(self, message_id: str) -> bool:
        """Check and record message ID for deduplication (persisted to disk)."""
        now = time.time()
        cutoff = now - _DEDUP_TTL_SECONDS

        with self._dedup_lock:
            # Prune expired entries
            before = len(self._dedup)
            self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}
            pruned = before - len(self._dedup)

            if message_id in self._dedup:
                if pruned:
                    logger.debug("Duplicate message %s (pruned %d expired)", message_id, pruned)
                else:
                    logger.debug("Duplicate message %s", message_id)
                return True
            self._dedup[message_id] = now
            self._save_dedup()
            if pruned:
                logger.debug("New message %s recorded (pruned %d expired, total %d)",
                             message_id, pruned, len(self._dedup))
            else:
                logger.debug("New message %s recorded (total %d)", message_id, len(self._dedup))
            return False

    def _build_client(self) -> Any:
        sdk_domain = LARK_DOMAIN if self.domain == "lark" else FEISHU_DOMAIN
        return (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .domain(sdk_domain)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    # ---- Processing status reactions ----

    def _reactions_enabled(self) -> bool:
        """Whether Feishu processing reactions are enabled (default: on)."""
        return os.getenv("FEISHU_REACTIONS", "true").strip().lower() not in ("false", "0", "no")

    async def _add_reaction(self, message_id: str, emoji_type: str) -> Optional[str]:
        """Add a reaction emoji to a message. Returns reaction_id on success."""
        if not self._client or not message_id or not emoji_type:
            return None
        try:
            from lark_oapi.api.im.v1 import (
                CreateMessageReactionRequest,
                CreateMessageReactionRequestBody,
            )
            body = (
                CreateMessageReactionRequestBody.builder()
                .reaction_type({"emoji_type": emoji_type})
                .build()
            )
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(body)
                .build()
            )
            response = await asyncio.to_thread(
                self._client.im.v1.message_reaction.create, request
            )
            if response and getattr(response, "code", None) == 0:
                data = getattr(response, "data", None)
                return getattr(data, "reaction_id", None)
            logger.debug(
                "[Feishu] Add reaction %s on %s rejected: code=%s msg=%s",
                emoji_type, message_id,
                getattr(response, "code", None),
                getattr(response, "msg", None),
            )
        except Exception:
            logger.warning(
                "[Feishu] Add reaction %s on %s raised", emoji_type, message_id,
                exc_info=True,
            )
        return None

    async def _remove_reaction(self, message_id: str, reaction_id: str) -> bool:
        """Remove a reaction from a message by its reaction_id."""
        if not self._client or not message_id or not reaction_id:
            return False
        try:
            from lark_oapi.api.im.v1 import DeleteMessageReactionRequest
            request = (
                DeleteMessageReactionRequest.builder()
                .message_id(message_id)
                .reaction_id(reaction_id)
                .build()
            )
            response = await asyncio.to_thread(
                self._client.im.v1.message_reaction.delete, request
            )
            if response and getattr(response, "code", None) == 0:
                return True
            logger.debug(
                "[Feishu] Remove reaction %s on %s rejected: code=%s msg=%s",
                reaction_id, message_id,
                getattr(response, "code", None),
                getattr(response, "msg", None),
            )
        except Exception:
            logger.warning(
                "[Feishu] Remove reaction %s on %s raised", reaction_id, message_id,
                exc_info=True,
            )
        return False

    def _remember_processing_reaction(self, message_id: str, reaction_id: str) -> None:
        """Store a processing reaction handle in the LRU cache."""
        cache = self._pending_processing_reactions
        cache[message_id] = reaction_id
        cache.move_to_end(message_id)
        while len(cache) > _FEISHU_PROCESSING_REACTION_CACHE_SIZE:
            evicted, _ = cache.popitem(last=False)
            logger.warning(
                "[Feishu] Evicted processing reaction for %s from cache (limit %d) — "
                "⌨️ badge will be permanent on that message",
                evicted, _FEISHU_PROCESSING_REACTION_CACHE_SIZE,
            )

    def _pop_processing_reaction(self, message_id: str) -> Optional[str]:
        """Retrieve and remove a stored processing reaction handle."""
        return self._pending_processing_reactions.pop(message_id, None)

    async def _add_processing_reaction(self, message_id: Optional[str]) -> None:
        """Add the ⌨️ processing reaction and track its handle."""
        if not message_id or message_id in self._pending_processing_reactions:
            return
        try:
            reaction_id = await self._add_reaction(message_id, _FEISHU_REACTION_IN_PROGRESS)
            if reaction_id:
                self._remember_processing_reaction(message_id, reaction_id)
        except asyncio.CancelledError:
            # The reaction may have been created server-side before
            # cancellation, but we lost the reaction_id (opaque handle
            # from the Create API).  Feishu's Delete Reaction API
            # requires this ID — we cannot remove by emoji_type alone.
            # The ⌨️ badge may remain permanently on this message.
            # This is an inherent API limitation, accepted as a rare
            # cosmetic edge case during forced cancellation (shutdown).
            logger.warning(
                "[Feishu] Cancelled while adding processing reaction on %s — "
                "⌨️ badge may be permanent (API requires reaction_id for deletion)",
                message_id,
            )
            raise

    async def _remove_processing_reaction(self, message_id: Optional[str]) -> None:
        """Remove the ⌨️ processing reaction from a message."""
        if not message_id:
            return
        reaction_id = self._pop_processing_reaction(message_id)
        if reaction_id:
            await self._remove_reaction(message_id, reaction_id)

    # ---- Processing status reactions ----

    def _on_message(self, data: Any) -> None:
        """Handle incoming WebSocket message event.

        This callback runs in the WS client's background thread.
        We must not call asyncio.create_task() here directly.
        """
        logger.info("_on_message called")
        if self._loop is None or self._loop.is_closed():
            logger.warning("No event loop available for Feishu message handling")
            return

        try:
            event = self._parse_event(data)
        except Exception as exc:
            logger.warning("Failed to parse Feishu event: %s", exc, exc_info=True)
            return

        if event is None:
            logger.info("_parse_event returned None, skipping")
            return

        logger.info("Received message from %s in %s chat %s: %r",
                    event.sender_id, event.chat_type, event.chat_id, event.text)

        # Schedule a single orchestrator coroutine that adds the ⌨️ reaction,
        # then runs the handler, then cleans up — all in one atomic sequence.
        # This eliminates the race where a fast handler could complete before
        # _add_processing_reaction's API call had returned its reaction_id.
        async def _handle_with_reaction(ev: MessageEvent) -> Optional[str]:
            """Add ⌨️, run handler, clean up — single coroutine, no race.

            Uses try/finally so the ⌨️ badge is always removed even if the
            coroutine is cancelled (gateway shutdown).

            Note: base.py's _handle_message already catches all Exception and
            returns None, so exceptions from the message handler never propagate
            here. The handler's result (str or None) is returned as-is; reaction
            cleanup happens regardless.
            """
            do_reaction = (
                self._reactions_enabled()
                and bool(ev.message_id)
                and not ev.is_command()
            )
            try:
                if do_reaction:
                    await self._add_processing_reaction(ev.message_id)
                return await self._handle_message(ev)
            finally:
                if do_reaction:
                    await self._remove_processing_reaction(ev.message_id)

        future = asyncio.run_coroutine_threadsafe(
            _handle_with_reaction(event), self._loop
        )
        # Log any unhandled exception from the orchestrator (shouldn't happen
        # with internal try/except, but defensive).
        future.add_done_callback(lambda f: (
            logger.error("[Feishu] _handle_with_reaction failed: %s", f.exception())
            if not f.cancelled() and f.exception() else None
        ))

    def _parse_event(self, data: Any) -> Optional[MessageEvent]:
        """Parse a Feishu event (dict or P2ImMessageReceiveV1 object) into a MessageEvent."""
        # The lark_oapi SDK passes P2ImMessageReceiveV1 objects, not dicts.
        # Use getattr for safe attribute access.
        header = getattr(data, "header", None)
        event_type = getattr(header, "event_type", "") if header else ""

        logger.info("Parsing event, type=%s", event_type)

        if event_type != "im.message.receive_v1":
            logger.info("Ignoring non-message event: %s", event_type)
            return None

        event_data = getattr(data, "event", None)
        if event_data is None:
            logger.warning("Event has no event data")
            return None

        message = getattr(event_data, "message", None)
        sender = getattr(event_data, "sender", None)
        if message is None or sender is None:
            logger.warning("Event missing message or sender")
            return None

        sender_id_info = getattr(sender, "sender_id", None) or {}
        sender_type = getattr(sender, "sender_type", "")

        message_id = getattr(message, "message_id", "") or ""
        if self._is_duplicate(message_id):
            logger.info("Duplicate message %s, skipping", message_id)
            return None

        msg_type = getattr(message, "message_type", "")
        content_str = getattr(message, "content", "{}") or "{}"
        chat_id = getattr(message, "chat_id", "") or ""
        chat_type_raw = getattr(message, "chat_type", "p2p")
        chat_type = "group" if chat_type_raw == "group" else "private"
        sender_id = getattr(sender_id_info, "open_id", "") or ""

        logger.info("Raw message: msg_type=%s, chat_type=%s, sender_type=%s, sender_id=%s, bot_open_id=%s",
                    msg_type, chat_type, sender_type, sender_id, self._bot_open_id)

        try:
            content = json.loads(content_str) if isinstance(content_str, str) else content_str
        except json.JSONDecodeError:
            content = {}

        # Skip messages sent by the bot itself
        if sender_type == "bot" or sender_id == self._bot_open_id:
            logger.info("Skipping self-message: sender_type=%s, sender_id=%s, bot_open_id=%s",
                        sender_type, sender_id, self._bot_open_id)
            return None

        # Parse text content
        text = ""
        if msg_type == _MSG_TYPE_TEXT:
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            # Remove @bot mentions and normalize
            if self._bot_open_id:
                text = re.sub(rf"<at user_id=\"{self._bot_open_id}\">.*?</at>", "", text).strip()
        elif msg_type == _MSG_TYPE_IMAGE:
            text = "[Image]"
        elif msg_type == _MSG_TYPE_FILE:
            text = "[File]"
        elif msg_type == _MSG_TYPE_AUDIO:
            text = "[Audio]"
        elif msg_type == _MSG_TYPE_MEDIA:
            text = "[Media]"
        elif msg_type == _MSG_TYPE_POST:
            text = _extract_post_text(content) if isinstance(content, dict) else "[Post]"
        else:
            text = f"[{msg_type}]"

        if not text:
            return None

        # Group chat gating: require @mention unless policy is "open"
        if chat_type == "group" and self.group_policy != "open":
            raw_content = content_str if isinstance(content_str, str) else json.dumps(content_str)
            if self._bot_open_id and f'user_id="{self._bot_open_id}"' not in raw_content:
                logger.debug("Ignoring group message without @mention: %s", message_id)
                return None

        event = MessageEvent(
            text=text,
            message_type=_map_msg_type(msg_type),
            platform="feishu",
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
            message_id=message_id,
            raw_message=data,
        )

        # Handle media — dict-driven dispatch replaces 4 near-identical blocks
        _MEDIA_KEYS = {
            _MSG_TYPE_IMAGE: ("image_key", "image"),
            _MSG_TYPE_FILE:  ("file_key",  "file"),
            _MSG_TYPE_AUDIO: ("file_key",  "audio"),
            _MSG_TYPE_MEDIA: ("file_key",  "media"),
        }
        if msg_type in _MEDIA_KEYS:
            key_name, media_type = _MEDIA_KEYS[msg_type]
            key_val = content.get(key_name, "") if isinstance(content, dict) else ""
            if key_val:
                event.media_urls = [key_val]
                event.media_types = [media_type]

        return event

    # ---- Media download ----

    async def _download_media(
        self,
        message_id: str,
        file_key: str,
        media_type: str,
    ) -> Optional[str]:
        """Download media (image/file/audio/media) from Feishu with correct extension.

        Uses the appropriate API based on media_type:
        - image: GetMessageResourceRequest (or GetImageRequest)
        - file: GetMessageResourceRequest (or GetFileRequest)
        - audio/media: GetMessageResourceRequest
        """
        if not self._client:
            return None
        loop = asyncio.get_running_loop()

        def _sync_download() -> Optional[str]:
            try:
                if media_type == "image":
                    # Try GetImageRequest first (dedicated image API)
                    req = GetImageRequest.builder().image_key(file_key).build()
                    resp = self._client.im.v1.image.get(req)
                elif media_type == "file":
                    # Try GetFileRequest first (dedicated file API)
                    req = GetFileRequest.builder().file_key(file_key).build()
                    resp = self._client.im.v1.file.get(req)
                else:
                    # Fallback to GetMessageResourceRequest for audio/media
                    req = GetMessageResourceRequest.builder() \
                        .message_id(message_id) \
                        .file_key(file_key) \
                        .build()
                    resp = self._client.im.v1.message_resource.get(req)

                if resp.code != 0:
                    logger.warning("Failed to download %s %s: %s", media_type, file_key, resp.msg)
                    return None

                # Determine extension from Content-Type and/or filename
                content_type = None
                filename = None
                if resp.raw and resp.raw.headers:
                    content_type = resp.raw.headers.get("Content-Type")
                if hasattr(resp, "file_name") and resp.file_name:
                    filename = resp.file_name

                ext = _resolve_extension(content_type, filename)
                if not ext:
                    # Fallback based on media type
                    ext = {
                        "image": ".png",
                        "file": ".bin",
                        "audio": ".mp3",
                        "media": ".mp4",
                    }.get(media_type, ".bin")

                prefix = media_type[:3]  # img, fil, aud, med
                filepath = self._cache_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"

                # Read content from response
                if hasattr(resp, "file") and resp.file:
                    filepath.write_bytes(resp.file.read())
                elif resp.raw and resp.raw.content:
                    filepath.write_bytes(resp.raw.content)
                else:
                    logger.warning("No content in %s download response", media_type)
                    return None

                logger.info("Downloaded %s to %s (Content-Type: %s, filename: %s)",
                            media_type, filepath, content_type, filename)
                return str(filepath)
            except Exception as exc:
                logger.warning("Error downloading %s %s: %s", media_type, file_key, exc)
                return None

        return await loop.run_in_executor(None, _sync_download)

    # Backward-compatible alias
    async def _download_image(self, message_id: str, image_key: str) -> Optional[str]:
        """Download an image (backward-compatible alias)."""
        return await self._download_media(message_id, image_key, "image")

    # ---- Lifecycle ----

    async def start(self) -> None:
        self._client = self._build_client()
        self._loop = asyncio.get_running_loop()

        # Probe bot info
        loop = asyncio.get_running_loop()
        probe = await loop.run_in_executor(
            None, probe_bot, self.app_id, self.app_secret, self.domain
        )
        if probe:
            self._bot_name = probe.get("bot_name", "")
            self._bot_open_id = probe.get("bot_open_id", "")
            logger.info("Feishu bot connected: %s", self._bot_name)
        else:
            # Fallback to configured bot_open_id if probe failed
            self._bot_open_id = self.config.extra.get("bot_open_id") or None
            logger.warning(
                "Could not probe Feishu bot info. "
                "Self-message filtering and @mention gating may be impaired."
            )

        # Build WS client
        event_handler = lark.EventDispatcherHandler.builder(
            verification_token=self.verification_token or None,
            encrypt_key=self.encrypt_key or None,
        ).register_p2_im_message_receive_v1(self._on_message).build()

        sdk_domain = LARK_DOMAIN if self.domain == "lark" else FEISHU_DOMAIN
        self._ws_client = FeishuWSClient(
            app_id=self.app_id,
            app_secret=self.app_secret,
            log_level=lark.LogLevel.INFO,
            event_handler=event_handler,
            domain=sdk_domain,
        )

        self._running = True
        logger.info("Starting Feishu WebSocket client...")
        # Run WS client in a dedicated thread with its own event loop
        loop = asyncio.get_running_loop()
        self._ws_future = loop.run_in_executor(
            None, _run_ws_client, self._ws_client, self
        )
        try:
            await self._ws_future
        except asyncio.CancelledError:
            logger.info("WS client task cancelled")
            raise

    async def stop(self) -> None:
        self._running = False
        ws_thread_loop = self._ws_thread_loop
        if ws_thread_loop is not None and not ws_thread_loop.is_closed():
            logger.debug("Cancelling Feishu websocket tasks and stopping loop")

            def cancel_all_tasks() -> None:
                for task in asyncio.all_tasks(ws_thread_loop):
                    if not task.done():
                        task.cancel()

            ws_thread_loop.call_soon_threadsafe(cancel_all_tasks)

        ws_future = self._ws_future
        if ws_future is not None:
            try:
                await asyncio.wait_for(ws_future, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Feishu WS thread did not exit within 10s")
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Feishu WS thread exited with error: %s", exc, exc_info=True)

        self._ws_future = None
        self._ws_thread_loop = None
        self._loop = None
        self._ws_client = None
        logger.info("Feishu adapter stopped")

    # ---- Send methods ----

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Client not initialized")

        msg_type, payload = _build_outbound_payload(text)
        result = await self._sync_send(chat_id, msg_type, payload, reply_to_message_id)

        # Fallback: if post was rejected, try plain text to avoid losing the message
        if not result.success and msg_type == "post":
            logger.warning(
                "[Feishu] Post send failed (%s), falling back to plain text", result.error
            )
            fallback_payload = json.dumps({"text": text}, ensure_ascii=False)
            result = await self._sync_send(chat_id, "text", fallback_payload, reply_to_message_id)

        return result

    async def send_photo(
        self,
        chat_id: str,
        photo_path: str,
        *,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a photo by uploading it to Feishu first, then sending as image message."""
        if not self._client:
            return SendResult(success=False, error="Client not initialized")

        # Upload image to Feishu
        image_key = await self._upload_image(photo_path)
        if not image_key:
            # Fallback to text with path
            text = f"[Photo: {photo_path}]"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send_message(chat_id, text, **kwargs)

        # Build image message payload
        payload = json.dumps({"image_key": image_key}, ensure_ascii=False)
        return await self._sync_send(chat_id, "image", payload, kwargs.get("reply_to_message_id"))

    async def send_document(
        self,
        chat_id: str,
        document_path: str,
        *,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a document by uploading it to Feishu first, then sending as file message."""
        if not self._client:
            return SendResult(success=False, error="Client not initialized")

        # Upload file to Feishu
        file_key = await self._upload_file(document_path)
        if not file_key:
            # Fallback to text with path
            text = f"[Document: {document_path}]"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send_message(chat_id, text, **kwargs)

        # Build file message payload
        filename = os.path.basename(document_path)
        payload = json.dumps({
            "file_key": file_key,
            "file_name": filename,
        }, ensure_ascii=False)
        return await self._sync_send(chat_id, "file", payload, kwargs.get("reply_to_message_id"))

    async def _upload_image(self, image_path: str) -> Optional[str]:
        """Upload an image to Feishu and return the image_key."""
        path = Path(image_path)
        if not path.exists():
            logger.warning("Image file not found: %s", image_path)
            return None

        loop = asyncio.get_running_loop()

        def _do_upload() -> Optional[str]:
            try:
                image_bytes = path.read_bytes()
                # Guess image type from extension
                ext = path.suffix.lower()
                image_type = {
                    ".png": "png",
                    ".jpg": "jpeg",
                    ".jpeg": "jpeg",
                    ".gif": "gif",
                    ".bmp": "bmp",
                    ".webp": "webp",
                }.get(ext, "png")

                req = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image(image_bytes)
                        .image_type(image_type)
                        .build()
                    )
                    .build()
                )
                resp = self._client.im.v1.image.create(req)
                if resp.code == 0 and resp.data and resp.data.image_key:
                    logger.info("Uploaded image %s -> key %s", image_path, resp.data.image_key)
                    return resp.data.image_key
                else:
                    logger.warning("Failed to upload image %s: %s", image_path, resp.msg)
                    return None
            except Exception as exc:
                logger.warning("Error uploading image %s: %s", image_path, exc)
                return None

        return await loop.run_in_executor(None, _do_upload)

    async def _upload_file(self, file_path: str) -> Optional[str]:
        """Upload a file to Feishu and return the file_key."""
        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found: %s", file_path)
            return None

        loop = asyncio.get_running_loop()

        def _do_upload() -> Optional[str]:
            try:
                file_bytes = path.read_bytes()
                filename = path.name
                ext = path.suffix.lower()
                # Map extension to Feishu file_type
                file_type = {
                    ".pdf": "pdf",
                    ".doc": "doc",
                    ".docx": "docx",
                    ".xls": "xls",
                    ".xlsx": "xlsx",
                    ".ppt": "ppt",
                    ".pptx": "pptx",
                    ".txt": "txt",
                    ".md": "txt",  # markdown as text
                    ".csv": "csv",
                    ".zip": "zip",
                    ".mp4": "mp4",
                    ".mov": "mov",
                    ".mp3": "mp3",
                    ".wav": "wav",
                }.get(ext, "stream")

                req = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file(file_bytes)
                        .file_name(filename)
                        .file_type(file_type)
                        .build()
                    )
                    .build()
                )
                resp = self._client.im.v1.file.create(req)
                if resp.code == 0 and resp.data and resp.data.file_key:
                    logger.info("Uploaded file %s -> key %s", file_path, resp.data.file_key)
                    return resp.data.file_key
                else:
                    logger.warning("Failed to upload file %s: %s", file_path, resp.msg)
                    return None
            except Exception as exc:
                logger.warning("Error uploading file %s: %s", file_path, exc)
                return None

        return await loop.run_in_executor(None, _do_upload)

    async def _sync_send(
        self, chat_id: str, msg_type: str, payload: str, reply_to_message_id: Optional[str]
    ) -> SendResult:
        """Synchronous send wrapped in executor."""
        loop = asyncio.get_running_loop()

        def _do_send() -> SendResult:
            try:
                if reply_to_message_id:
                    req = (
                        ReplyMessageRequest.builder()
                        .message_id(reply_to_message_id)
                        .request_body(
                            ReplyMessageRequestBody.builder()
                            .content(payload)
                            .msg_type(msg_type)
                            .build()
                        )
                        .build()
                    )
                    resp = self._client.im.v1.message.reply(req)
                else:
                    req = (
                        CreateMessageRequest.builder()
                        .receive_id_type("chat_id")
                        .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(chat_id)
                            .content(payload)
                            .msg_type(msg_type)
                            .build()
                        )
                        .build()
                    )
                    resp = self._client.im.v1.message.create(req)

                if resp.code == 0:
                    msg_id = resp.data.message_id if resp.data else ""
                    logger.info("[Feishu] Message sent successfully (type=%s, msg_id=%s)", msg_type, msg_id)
                    return SendResult(success=True, message_id=msg_id, msg_type=msg_type)
                else:
                    logger.warning("[Feishu] Message send failed (type=%s): %s: %s", msg_type, resp.code, resp.msg)
                    return SendResult(success=False, error=f"{resp.code}: {resp.msg}")
            except Exception as exc:
                logger.exception("Failed to send Feishu message")
                return SendResult(success=False, error=str(exc), retryable=True)

        return await loop.run_in_executor(None, _do_send)

    def build_session_key(self, event: MessageEvent) -> str:
        # Group chats: platform:chat_id:sender_id
        # Private chats: platform:chat_id
        if event.chat_type == "group":
            return f"feishu:{event.chat_id}:{event.sender_id}"
        return f"feishu:{event.chat_id}"

    # ---- Edit message (progressive streaming update) ----

    MAX_MESSAGE_LENGTH: int = 20000

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        msg_type: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Edit an existing message (progressive update for streaming).

        Uses Feishu PATCH /im/v1/messages/{message_id} API.

        ⚠️ The PATCH API does NOT support switching msg_type. The caller
        (StreamConsumer) is responsible for passing the correct msg_type
        that matches the initial message's type.

        If no msg_type is provided, falls back to _build_outbound_payload
        to determine it. If the type doesn't match the original message,
        the edit will fail and the caller should fall back to text editing.
        """
        if not self._client:
            return SendResult(success=False, error="Client not initialized")

        if msg_type is None:
            msg_type, payload = _build_outbound_payload(text)
        else:
            if msg_type == "text":
                payload = json.dumps({"text": text}, ensure_ascii=False)
            else:
                # Always build post-compatible payload (never trust auto-detect
                # when caller explicitly requests post — text w/o markdown would
                # produce a text payload that mismatches the requested msg_type).
                # Also convert tables to code blocks since Feishu post md does
                # not render them — this mirrors _build_outbound_payload's logic.
                if _MARKDOWN_TABLE_RE.search(text):
                    text = _convert_tables_to_code_blocks(text)
                payload = _build_markdown_post_payload(text)

        loop = asyncio.get_running_loop()

        def _do_edit() -> SendResult:
            try:
                from lark_oapi.api.im.v1.model import (
                    UpdateMessageRequest,
                    UpdateMessageRequestBody,
                )

                body = UpdateMessageRequestBody()
                body.content = payload
                body.msg_type = msg_type

                req = (
                    UpdateMessageRequest.builder()
                    .message_id(message_id)
                    .request_body(body)
                    .build()
                )
                resp = self._client.im.v1.message.update(req)
                if resp.code == 0:
                    logger.info("[Feishu] Message edited successfully (msg_id=%s)", message_id)
                    return SendResult(success=True, message_id=message_id)
                logger.warning("[Feishu] Message edit failed (msg_id=%s, type=%s): %s: %s",
                               message_id, msg_type, resp.code, resp.msg)
                return SendResult(success=False, error=f"{resp.code}: {resp.msg}")
            except Exception as exc:
                logger.exception("Failed to edit Feishu message")
                return SendResult(success=False, error=str(exc), retryable=True)

        return await loop.run_in_executor(None, _do_edit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_msg_type(msg_type: str) -> MessageType:
    mapping = {
        _MSG_TYPE_TEXT: MessageType.TEXT,
        _MSG_TYPE_IMAGE: MessageType.PHOTO,
        _MSG_TYPE_FILE: MessageType.DOCUMENT,
        _MSG_TYPE_AUDIO: MessageType.AUDIO,
        _MSG_TYPE_MEDIA: MessageType.VIDEO,
    }
    return mapping.get(msg_type, MessageType.TEXT)


def _extract_post_text(content: dict) -> str:
    """Extract text from a Feishu post message, preserving markdown styles.

    Handles text elements with style attributes (bold, italic, underline,
    strikethrough, code) and converts them to markdown syntax.
    Also handles media tags: img, media, file, audio, video.
    """
    texts = []
    try:
        post = content.get("post", {})
        for locale in ("zh_cn", "en_us", "ja_jp"):
            locale_content = post.get(locale, {})
            for row in locale_content.get("content", []):
                row_parts = []
                for elem in row:
                    tag = elem.get("tag", "")
                    if tag == "text":
                        row_parts.append(_render_post_text_element(elem))
                    elif tag == "a":
                        href = elem.get("href", "")
                        label = elem.get("text", "")
                        escaped = _escape_markdown_text(label) if label else ""
                        row_parts.append(f"[{escaped}]({href})" if href and escaped else escaped or label)
                    elif tag == "at":
                        name = elem.get("user_name", elem.get("user_id", ""))
                        row_parts.append(f"@{name}")
                    elif tag == "img":
                        row_parts.append("[Image]")
                    elif tag == "media":
                        row_parts.append("[Media]")
                    elif tag == "file":
                        row_parts.append("[File]")
                    elif tag == "audio":
                        row_parts.append("[Audio]")
                    elif tag == "video":
                        row_parts.append("[Video]")
                    elif tag == "code":
                        code = elem.get("text", "")
                        row_parts.append(_wrap_inline_code(code) if code else "")
                    elif tag in ("code_block", "pre"):
                        lang = str(elem.get("language", "") or elem.get("lang", "")).strip()
                        code = str(elem.get("text", "") or elem.get("content", "")).replace("\r\n", "\n")
                        trailing = "" if code.endswith("\n") else "\n"
                        row_parts.append(f"```{lang}\n{code}{trailing}```")
                    elif tag in ("br",):
                        row_parts.append("\n")
                    elif tag in ("hr", "divider"):
                        row_parts.append("\n\n---\n\n")
                if row_parts:
                    texts.append("".join(row_parts))
    except Exception:
        pass
    return "\n".join(texts) or "[Post]"


def _render_post_text_element(elem: dict) -> str:
    """Render a single post text element with style to markdown."""
    text = str(elem.get("text", "") or "")
    style = elem.get("style")
    style_dict = style if isinstance(style, dict) else None

    # Inline code takes precedence over other styles
    if _is_style_enabled(style_dict, "code"):
        return _wrap_inline_code(text)

    rendered = _escape_markdown_text(text)
    if not rendered:
        return ""
    if _is_style_enabled(style_dict, "bold"):
        rendered = f"**{rendered}**"
    if _is_style_enabled(style_dict, "italic"):
        rendered = f"*{rendered}*"
    if _is_style_enabled(style_dict, "underline"):
        rendered = f"<u>{rendered}</u>"
    if _is_style_enabled(style_dict, "strikethrough"):
        rendered = f"~~{rendered}~~"
    return rendered


def _is_style_enabled(style: dict | None, key: str) -> bool:
    if not style:
        return False
    value = style.get(key)
    return value is True or value == 1 or value == "true"


def _wrap_inline_code(text: str) -> str:
    """Wrap text in backticks, handling existing backtick runs."""
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"
