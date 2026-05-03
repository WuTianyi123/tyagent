"""Gateway lifecycle management — signals, restart, drain, recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tyagent.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


class GatewaySupervisor:
    """Manages gateway lifecycle: signals, graceful restart, drain, recovery.

    Does NOT inherit from Gateway — it is a helper that holds a
    reference to the gateway and manipulates its state.  This
    keeps lifecycle concerns out of the main Gateway class.
    """

    def __init__(self, gateway: Gateway) -> None:
        self._gateway = gateway

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM/SIGUSR1."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.shutdown)
            except NotImplementedError:
                pass  # Windows
        try:
            loop.add_signal_handler(signal.SIGUSR1, self._on_sigusr1)
        except NotImplementedError:
            pass

    def shutdown(self) -> None:
        """Signal the gateway to shut down."""
        gw = self._gateway
        gw._running = False
        gw._shutdown_event.set()

    def _on_sigusr1(self) -> None:
        """Handle SIGUSR1 — start graceful restart."""
        gw = self._gateway
        if gw._restart_requested:
            logger.warning("SIGUSR1 already received, ignoring duplicate")
            return
        logger.info("SIGUSR1 received — initiating graceful restart")
        gw._restart_requested = True
        gw._draining = True
        loop = asyncio.get_running_loop()
        loop.create_task(self._do_graceful_restart())

    # ------------------------------------------------------------------
    # Graceful restart
    # ------------------------------------------------------------------

    async def _do_graceful_restart(self) -> None:
        """Perform graceful restart: notify, drain, mark sessions, exit."""
        gw = self._gateway
        try:
            logger.info("Graceful restart: notifying active sessions...")
            await self._notify_active_sessions()

            logger.info(
                "Graceful restart: draining up to %.0f seconds (%d active sessions)",
                gw._restart_drain_timeout,
                len(gw._sessions),
            )
            drained = await self._drain_active_agents(gw._restart_drain_timeout)

            # Stop all session agents (actor model loops)
            for session_key in list(gw._sessions.keys()):
                try:
                    await gw._stop_session_agent(session_key)
                except Exception:
                    logger.exception(
                        "Failed to stop session agent for %s", session_key
                    )

            if not drained:
                logger.warning(
                    "Drain timeout reached — forcing restart with %d active session(s)",
                    len(gw._sessions),
                )

            # Mark pending sessions for resume, so the new process can pick them up
            for session_key in list(gw._sessions):
                try:
                    gw.session_store.mark_resume_pending(
                        session_key, reason="restart_timeout"
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark resume_pending for %s", session_key
                    )

            # Write .clean_shutdown marker
            self._write_clean_shutdown_marker()
        except Exception:
            logger.exception(
                "Graceful restart failed unexpectedly — proceeding with restart anyway"
            )

        logger.info(
            "Graceful restart complete — spawning systemctl restart and shutting down"
        )
        import subprocess

        # Spawn the restart command in the background.  If it fails we
        # still shut down gracefully; the process manager (systemd) will
        # restart us via its own Restart= policy.
        try:
            subprocess.Popen(
                [
                    "systemd-run",
                    "--user",
                    "--scope",
                    "--unit=tyagent-restart-helper",
                    "systemctl",
                    "--user",
                    "restart",
                    "tyagent-gateway",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.error(
                "Failed to spawn systemctl restart: %s — relying on Restart= policy",
                exc,
            )

        # Trigger the normal gateway shutdown path so that all cleanup
        # (adapters, session agents, cached agents) runs before exit.
        self.shutdown()

    def _write_clean_shutdown_marker(self) -> None:
        """Write .clean_shutdown marker with restart requestor info."""
        gw = self._gateway
        try:
            marker_path = gw.config.home_dir / ".clean_shutdown"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_data: dict = {"reason": "restart"}
            req = getattr(gw, "_restart_requestor", None)
            if req:
                marker_data["requestor_platform"] = req["platform"]
                marker_data["requestor_chat_id"] = req["chat_id"]
            marker_data["initiated_at"] = time.time()
            marker_path.write_text(json.dumps(marker_data), encoding="utf-8")
            logger.info("Wrote .clean_shutdown marker at %s", marker_path)
        except Exception as exc:
            logger.error("Failed to write .clean_shutdown marker: %s", exc)

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    async def _drain_active_agents(self, timeout: float) -> bool:
        """Wait for all active sessions to complete, up to timeout seconds.

        Returns True if all sessions drained, False on timeout.
        """
        gw = self._gateway
        if not gw._sessions:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while gw._sessions:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(0.5)
        return True

    async def _notify_active_sessions(self) -> None:
        """Send restart notification to all active sessions."""
        gw = self._gateway
        message = (
            "⚠️ Gateway is restarting for an update. "
            "Active requests will complete before restart."
        )
        for session_key in list(gw._sessions):
            try:
                ctx = gw._sessions.get(session_key)
                adapter_name = ctx.platform_name if ctx else ""
                chat_id = ctx.chat_id if ctx else ""
                adapter = gw.adapters.get(adapter_name) if adapter_name else None
                if adapter is not None and chat_id:
                    await adapter.send_message(chat_id, message)
            except Exception:
                logger.exception("Failed to notify session %s", session_key)

    # ------------------------------------------------------------------
    # Recovery on startup
    # ------------------------------------------------------------------

    def check_recovery_on_startup(self) -> None:
        """Check for .clean_shutdown marker and handle session recovery.

        If the marker exists with restart requestor info, save it as a
        pending notification to be sent once adapters are connected.

        If the marker is absent, assume unclean shutdown and suspend
        recently-active sessions.
        """
        gw = self._gateway
        gw._restart_notification_pending = None  # type: ignore[assignment]
        marker_path = gw.config.home_dir / ".clean_shutdown"

        if not marker_path.exists():
            logger.info(
                "No clean shutdown marker — assuming unclean shutdown, "
                "suspending recent sessions"
            )
            self._suspend_recent_sessions()
            return

        # Parse marker — new format is JSON, old format is plain "clean"
        try:
            raw = marker_path.read_text(encoding="utf-8").strip()
            marker = json.loads(raw) if raw.startswith("{") else {"reason": raw}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse .clean_shutdown marker: %s", exc)
            marker_path.unlink(missing_ok=True)
            self._suspend_recent_sessions()
            return

        logger.info(
            "Clean shutdown marker found — previous restart was intentional (reason=%s)",
            marker.get("reason", "unknown"),
        )

        # Save restart notification for later (adapters aren't connected yet)
        platform_name = marker.get("requestor_platform")
        chat_id = marker.get("requestor_chat_id")
        if platform_name and chat_id:
            gw._restart_notification_pending = {
                "platform": platform_name,
                "chat_id": chat_id,
                "initiated_at": marker.get("initiated_at"),
            }
            logger.info(
                "Saved restart notification for %s via %s (will send after adapter connects)",
                chat_id,
                platform_name,
            )

        # Clean up marker
        try:
            marker_path.unlink(missing_ok=True)
            logger.debug("Removed .clean_shutdown marker")
        except OSError as exc:
            logger.warning("Failed to remove .clean_shutdown marker: %s", exc)

    def _suspend_recent_sessions(self) -> None:
        """Suspend sessions that were recently active (unclean shutdown recovery)."""
        gw = self._gateway
        try:
            suspended = gw.session_store.suspend_recently_active(max_age_seconds=120)
            if suspended:
                logger.warning(
                    "Suspended %d recently-active session(s) due to unclean shutdown",
                    suspended,
                )
            else:
                logger.debug("No recently-active sessions to suspend")
        except Exception:
            logger.exception("Failed to suspend recent sessions on startup")

    # ------------------------------------------------------------------
    # Restart notification (called from Gateway.start after adapters connect)
    # ------------------------------------------------------------------

    @staticmethod
    def schedule_restart_notification(gateway: Gateway) -> None:
        """If there is a pending restart notification, schedule it.

        Called from Gateway.start() after adapters have been launched.
        """
        notif = getattr(gateway, "_restart_notification_pending", None)
        if not notif:
            return

        async def _send() -> None:
            adapter = gateway.adapters.get(notif["platform"])
            if not adapter:
                return
            t0 = time.monotonic()
            logger.info(
                "Restart notification: waiting for adapter %s to connect...",
                notif["platform"],
            )
            # Poll until adapter is running (WebSocket connected), up to 10 s
            for _ in range(20):
                if adapter.running:
                    break
                await asyncio.sleep(0.5)
            connect_time = time.monotonic() - t0
            if adapter.running:
                t1 = time.monotonic()
                initiated_at = notif.get("initiated_at")
                if initiated_at:
                    total_elapsed = time.time() - initiated_at
                    msg = f"✅ Gateway 已重启完成（总耗时 {total_elapsed:.0f}s）"
                else:
                    msg = "✅ Gateway 已重启完成"
                await adapter.send_message(notif["chat_id"], msg)
                send_time = time.monotonic() - t1
                logger.info(
                    "Sent restart notification to %s via %s "
                    "(connect=%.1fs, send=%.1fs, total=%.1fs)",
                    notif["chat_id"],
                    notif["platform"],
                    connect_time,
                    send_time,
                    time.monotonic() - t0,
                )
            else:
                logger.warning(
                    "Adapter %s not running within 10s — skipping restart notification",
                    notif["platform"],
                )

        asyncio.create_task(_send())
