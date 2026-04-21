"""Feishu/Lark platform adapter for ty-agent.

Supports:
- WebSocket long connection
- Direct-message and group @mention-gated text receive/send
- Inbound image caching
- QR scan-to-create onboarding
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ty_agent.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

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
_MSG_TYPE_POST = "post"

# Dedup TTL
_DEDUP_TTL_SECONDS = 24 * 60 * 60


try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
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
        qr_url += "&from=ty-agent&tp=ty-agent"
    else:
        qr_url += "?from=ty-agent&tp=ty-agent"
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
# Feishu Platform Adapter
# ---------------------------------------------------------------------------

class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark platform adapter."""

    def __init__(self, config: Any):
        super().__init__(config, "feishu")
        if not FEISHU_AVAILABLE:
            raise ImportError(
                "lark_oapi is required for Feishu support. "
                "Install it with: uv add lark-oapi"
            )

        self.app_id: str = config.extra.get("app_id", "")
        self.app_secret: str = config.extra.get("app_secret", "")
        self.domain: str = config.extra.get("domain", "feishu")
        self.encrypt_key: str = config.extra.get("encrypt_key", "")
        self.verification_token: str = config.extra.get("verification_token", "")
        self.group_policy: str = config.extra.get("group_policy", "mention")  # "open" | "mention" | "disabled"

        if not self.app_id or not self.app_secret:
            raise ValueError("Feishu adapter requires app_id and app_secret in config.extra")

        self._client: Any = None
        self._ws_client: Any = None
        self._ws_future: Any = None
        self._ws_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bot_open_id: Optional[str] = None
        self._bot_name: str = ""
        self._running = False
        self._dedup: Dict[str, float] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Cache dir for media
        self._cache_dir = Path.home() / ".ty_agent" / "cache" / "feishu"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

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

    # ---- WebSocket handlers ----

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

        # Schedule coroutine on the main event loop thread-safely
        future = asyncio.run_coroutine_threadsafe(
            self._handle_message(event), self._loop
        )
        # Attach error callback so exceptions aren't lost
        future.add_done_callback(self._on_task_done)

    def _on_task_done(self, future: Any) -> None:
        """Callback for background tasks to log errors."""
        try:
            result = future.result()
            logger.info("Message handler completed, result=%s", result)
        except Exception as exc:
            logger.exception("Feishu message handler task failed: %s", exc)

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

        # Handle media
        if msg_type == _MSG_TYPE_IMAGE:
            image_key = content.get("image_key", "") if isinstance(content, dict) else ""
            if image_key:
                event.media_urls = [image_key]
                event.media_types = ["image"]
        elif msg_type == _MSG_TYPE_FILE:
            file_key = content.get("file_key", "") if isinstance(content, dict) else ""
            if file_key:
                event.media_urls = [file_key]
                event.media_types = ["file"]

        return event

    def _is_duplicate(self, message_id: str) -> bool:
        """Check and record message ID for deduplication."""
        now = time.time()
        cutoff = now - _DEDUP_TTL_SECONDS
        self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}

        if message_id in self._dedup:
            return True
        self._dedup[message_id] = now
        return False

    async def _download_image(self, message_id: str, image_key: str) -> Optional[str]:
        """Download an image from Feishu and save to cache (async-safe)."""
        if not self._client:
            return None
        loop = asyncio.get_running_loop()

        def _sync_download() -> Optional[str]:
            try:
                req = GetMessageResourceRequest.builder() \
                    .message_id(message_id) \
                    .file_key(image_key) \
                    .build()
                resp = self._client.im.v1.message_resource.get(req)
                if resp.code != 0:
                    logger.warning("Failed to download image %s: %s", image_key, resp.msg)
                    return None
                filepath = self._cache_dir / f"img_{uuid.uuid4().hex[:12]}.png"
                filepath.write_bytes(resp.raw.content)
                return str(filepath)
            except Exception as exc:
                logger.warning("Error downloading image %s: %s", image_key, exc)
                return None

        return await loop.run_in_executor(None, _sync_download)

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

        content = json.dumps({"text": text}, ensure_ascii=False)
        loop = asyncio.get_running_loop()

        def _sync_send() -> SendResult:
            try:
                if reply_to_message_id:
                    req = ReplyMessageRequest.builder() \
                        .message_id(reply_to_message_id) \
                        .request_body(ReplyMessageRequestBody.builder()
                              .content(content)
                              .msg_type("text")
                              .build()) \
                        .build()
                    resp = self._client.im.v1.message.reply(req)
                else:
                    req = CreateMessageRequest.builder() \
                        .receive_id_type("chat_id") \
                        .request_body(CreateMessageRequestBody.builder()
                              .receive_id(chat_id)
                              .content(content)
                              .msg_type("text")
                              .build()) \
                        .build()
                    resp = self._client.im.v1.message.create(req)

                if resp.code == 0:
                    data = json.loads(resp.raw.content) if hasattr(resp, "raw") else {}
                    msg_id = data.get("data", {}).get("message_id", "")
                    return SendResult(success=True, message_id=msg_id)
                else:
                    return SendResult(success=False, error=f"{resp.code}: {resp.msg}")
            except Exception as exc:
                logger.exception("Failed to send Feishu message")
                return SendResult(success=False, error=str(exc), retryable=True)

        return await loop.run_in_executor(None, _sync_send)

    def build_session_key(self, event: MessageEvent) -> str:
        # Group chats: platform:chat_id:sender_id
        # Private chats: platform:chat_id
        if event.chat_type == "group":
            return f"feishu:{event.chat_id}:{event.sender_id}"
        return f"feishu:{event.chat_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_msg_type(msg_type: str) -> MessageType:
    mapping = {
        _MSG_TYPE_TEXT: MessageType.TEXT,
        _MSG_TYPE_IMAGE: MessageType.PHOTO,
        _MSG_TYPE_FILE: MessageType.DOCUMENT,
    }
    return mapping.get(msg_type, MessageType.TEXT)


def _extract_post_text(content: dict) -> str:
    """Extract plain text from a Feishu post message."""
    texts = []
    try:
        post = content.get("post", {})
        for locale in ("zh_cn", "en_us", "ja_jp"):
            locale_content = post.get(locale, {})
            for row in locale_content.get("content", []):
                for elem in row:
                    tag = elem.get("tag", "")
                    if tag == "text":
                        texts.append(elem.get("text", ""))
                    elif tag == "a":
                        texts.append(f"{elem.get('text', '')} ({elem.get('href', '')})")
                    elif tag == "at":
                        texts.append(f"@{elem.get('user_name', elem.get('user_id', ''))}")
                    elif tag == "img":
                        texts.append("[Image]")
    except Exception:
        pass
    return "\n".join(texts) or "[Post]"
