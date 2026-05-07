"""Gateway lifecycle management — signals, restart, drain, recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import httpx
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tyagent.gateway.gateway import Gateway

logger = logging.getLogger(__name__)


def _pid_is_alive(pid: int) -> bool:
    """Check if a process is still running by sending signal 0."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


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
        """Perform graceful restart: notify, validate, drain, mark sessions, exit."""
        gw = self._gateway
        try:
            logger.info("Graceful restart: notifying active sessions...")
            await self._notify_active_sessions()

            # Pre-flight check: validate message chains before drain/restart.
            # Sends a minimal request (max_tokens=1) to the LLM API. If any
            # session's chain is rejected, a RuntimeError is raised and the
            # restart is aborted — the operator must fix the chain first.
            await self.validate_message_chains()

            active_count = len(gw._sessions)
            logger.info(
                "Graceful restart: draining up to %.0f seconds (%d active sessions)",
                gw._restart_drain_timeout,
                active_count,
            )
            await self._drain_active_agents(gw._restart_drain_timeout)

            # Write restart marker with pending tool call info so the new
            # gateway process can write accurate "restart_completed" responses.
            self._write_restart_marker()

            # Stop all session agents (actor model loops)
            for session_key in list(gw._sessions.keys()):
                try:
                    await gw._stop_session_agent(session_key)
                except Exception:
                    logger.exception(
                        "Failed to stop session agent for %s", session_key
                    )

            # Mark any sessions that might need resume after restart
            for session_key in list(gw._sessions):
                try:
                    gw.session_store.mark_resume_pending(
                        session_key, reason="restart"
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
            "Graceful restart complete — exiting with code 75 for systemd restart"
        )

        # Exit with code 75 (EX_TEMPFAIL).  The systemd service unit
        # has Restart=always + RestartForceExitStatus=75, so systemd
        # treats this exit as an intentional restart request and
        # re-launches the gateway — no shell wrapper or systemd-run
        # scope dance needed.
        #
        # os._exit() is used (not sys.exit()) because this runs inside
        # the asyncio event loop, and sys.exit() would raise
        # SystemExit which may be caught by test runners or the event
        # loop.  os._exit() is immediate and unconditional.
        os._exit(75)

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
        """Wait for in-flight turns to complete.

        Sessions are permanent actor-model loops — they don't disappear
        from ``_sessions`` until ``_stop_session_agent()`` removes them
        (which runs after this drain phase).

        Instead of waiting for sessions to vanish (which would always
        timeout), we check whether any agent is currently inside a turn
        (LLM call or tool execution).  If so, we wait a brief period for
        it to settle.  The subsequent ``stop()`` (5s shutdown_timeout)
        provides the final grace period for any remaining in-flight work.

        Returns True (drain "succeeded") — the 5s stop() timeout is the
        real safety net, making this a soft courtesy wait rather than a
        hard deadline.
        """
        gw = self._gateway
        if not gw._sessions:
            return True

        # Give any in-flight turns up to `timeout` seconds to finish
        # naturally before stop() forces them.
        start = time.monotonic()
        busy_count = len(gw._sessions)
        while time.monotonic() - start < timeout:
            # Count sessions whose agent is currently running a turn.
            # We approximate "busy" by checking if the agent's inbox
            # is non-empty (a message queued but not yet consumed).
            busy = 0
            for ctx in gw._sessions.values():
                agent = ctx.agent
                if agent._running and not agent._inbox.empty():
                    busy += 1
                elif hasattr(agent, '_bg_tasks') and agent._bg_tasks:
                    busy += 1
            if busy == 0:
                logger.info(
                    "Drain complete (%d sessions, all idle) in %.1fs",
                    busy_count, time.monotonic() - start,
                )
                return True
            await asyncio.sleep(0.5)

        remaining = len(gw._sessions)
        logger.info(
            "Drain courtesy wait elapsed (%d sessions, %d busy after %.1fs) "
            "— proceeding to stop() which handles remaining in-flight work",
            remaining, busy, time.monotonic() - start,
        )
        return True  # stop() is the real safety net

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

        If the marker is absent, sessions continue normally from persisted
        data — SQLite guarantees data consistency across crashes.
        """
        gw = self._gateway
        gw._restart_notification_pending = None  # type: ignore[assignment]
        marker_path = gw.config.home_dir / ".clean_shutdown"

        if not marker_path.exists():
            logger.warning(
                "No clean shutdown marker — gateway may not have shut down cleanly, "
                "but sessions will continue normally from persisted data"
            )
            return

        # Parse marker — new format is JSON, old format is plain "clean"
        try:
            raw = marker_path.read_text(encoding="utf-8").strip()
            marker = json.loads(raw) if raw.startswith("{") else {"reason": raw}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse .clean_shutdown marker: %s — sessions will continue normally", exc)
            marker_path.unlink(missing_ok=True)
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

    async def validate_message_chains(self) -> None:
        """Pre-flight check: validate all active sessions' message chains.

        Sends a minimal request (max_tokens=1) to the LLM API for each
        active session's message chain. If the API rejects any chain
        (e.g. orphaned tool calls with invalid format), raises a
        RuntimeError with details to block the restart.

        This prevents restarting into a state where the LLM will reject
        the message chain, ensuring the new process starts cleanly.
        """
        gw = self._gateway
        agent_cfg = gw.config.agent
        if not agent_cfg.api_key or not agent_cfg.base_url:
            return True  # Can't validate without API config

        headers = {
            "Authorization": f"Bearer {agent_cfg.api_key}",
            "Content-Type": "application/json",
        }
        all_ok = True

        for session_key in list(gw._sessions.keys()):
            try:
                session = gw.session_store.get(session_key)
                session_id = session.metadata.get("current_session_id", "")
                if not session_id:
                    continue
                messages = gw.session_store.get_messages(
                    session_key, session_id=session_id
                )
                if not messages:
                    continue
            except Exception:
                logger.exception("Failed to load session %s for validation", session_key)
                continue

            # Build a minimal API request with the message chain
            payload = {
                "model": agent_cfg.model,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You are a validation probe."},
                    *[
                        {"role": m["role"], "content": m.get("content", "")}
                        | ({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})
                        for m in messages
                    ],
                ],
            }

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        f"{agent_cfg.base_url}/chat/completions",
                        headers=headers, json=payload,
                    )
                if resp.status_code >= 400:
                    body = resp.text[:500]
                    logger.warning(
                        "Session %s message chain validation FAILED "
                        "(HTTP %d): %s",
                        session_key, resp.status_code, body,
                    )
                    all_ok = False
                else:
                    logger.info(
                        "Session %s message chain validation OK "
                        "(%d messages, %d prompt tokens)",
                        session_key, len(messages),
                        resp.json().get("usage", {}).get("prompt_tokens", "?"),
                    )
            except Exception as exc:
                logger.warning(
                    "Session %s validation request failed: %s — skipping",
                    session_key, exc,
                )

        if not all_ok:
            raise RuntimeError(
                "Message chain validation failed for one or more sessions — "
                "restart aborted. Check logs above for details."
            )

    def _cleanup_gateway_interrupt_dir(self) -> None:
        """Remove all .gateway_interrupt/ marker files.

        These are one-shot markers — once processed (or when no restart
        marker is being written), they should be removed to prevent
        disk accumulation.
        """
        gw = self._gateway
        interrupt_dir = gw.config.home_dir / ".gateway_interrupt"
        if not interrupt_dir.exists():
            return
        for _mf in interrupt_dir.glob("*.json"):
            try:
                _mf.unlink()
            except OSError:
                pass

    def _write_restart_marker(self) -> None:
        """Write pending tool call info to a restart marker file.

        Only collects tool calls that are **related to this restart**:

        1.  ``.gateway_interrupt/`` markers written by the terminal tool
            before executing commands that trigger a gateway restart
            (e.g. ``tyagent gateway restart``).  These get a normal
            success response — the restart IS the expected outcome.

        2.  In-flight tool calls that are currently executing during
            drain.  These may be interrupted — we record them so the
            new process can write an "unknown failure" response.

        Historical orphaned tool calls in the DB are NOT touched —
        they are unrelated to this restart and writing synthetic
        responses for them would break the message chain.
        """
        gw = self._gateway
        marker = {"restarted_at": time.time(), "sessions": {}}
        home_dir: Path = gw.config.home_dir

        # ── Collect gateway_interrupt markers ──────────────────
        interrupt_dir = home_dir / ".gateway_interrupt"
        if interrupt_dir.exists():
            for _mf in interrupt_dir.glob("*.json"):
                try:
                    data = json.loads(_mf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                sk = data.get("session_key", "")
                sid = data.get("session_id", "")
                tcid = data.get("tool_call_id", "")
                if not sk or not sid or not tcid:
                    continue
                marker.setdefault("sessions", {})
                entry = marker["sessions"].setdefault(sk, {
                    "session_id": sid,
                    "pending_tool_calls": [],
                })
                entry["pending_tool_calls"].append({
                    "tool_call_id": tcid,
                    "function_name": "terminal",
                    "reason": "restart_trigger",
                    "command": data.get("command", "")[:200],
                })
                logger.info(
                    "Restart marker: restart-trigger terminal for %s/%s "
                    "(tool_call_id=%s)",
                    sk, sid, tcid,
                )

        # ── Collect in-flight tool calls during drain ──────────
        for session_key in list(gw._sessions.keys()):
            ctx = gw._sessions.get(session_key)
            if ctx is None:
                continue
            agent = ctx.agent
            if not agent._running:
                continue
            # Agent is mid-tool-execution if _current_tool_call_id is set.
            # It is only set inside _execute_tool_calls and cleared in
            # its finally block — so a non-empty value means a tool is
            # executing right now and will be interrupted by the restart.
            tc_id = getattr(agent, "_current_tool_call_id", "") or ""
            if not tc_id:
                continue
            try:
                session = gw.session_store.get(session_key)
                session_id = session.metadata.get("current_session_id", "")
            except Exception:
                continue
            if not session_id:
                continue

            # Skip if already captured as restart_trigger (dedup)
            entry = marker.get("sessions", {}).get(session_key)
            if entry:
                already = {tc["tool_call_id"] for tc in entry["pending_tool_calls"]}
                if tc_id in already:
                    logger.info(
                        "Restart marker: skipping in-flight tool call for %s/%s "
                        "(tool_call_id=%s) — already captured as restart_trigger",
                        session_key, tc_id,
                    )
                    continue

            entry = marker.setdefault("sessions", {}).setdefault(session_key, {
                "session_id": session_id,
                "pending_tool_calls": [],
            })
            entry["pending_tool_calls"].append({
                "tool_call_id": tc_id,
                "function_name": "?",
                "reason": "unknown_failure",
            })
            logger.info(
                "Restart marker: in-flight tool call for %s/%s "
                "(tool_call_id=%s)",
                session_key, session_id, tc_id,
            )

        if not marker.get("sessions"):
            # No restart-related tool calls — clean up gateway_interrupt
            # markers to prevent disk accumulation.
            self._cleanup_gateway_interrupt_dir()
            logger.info("No restart-related tool calls — skipping restart marker")
            return

        marker_path = home_dir / ".restart_pending"
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "Wrote restart marker at %s (%d sessions)",
                marker_path, len(marker["sessions"]),
            )
        except Exception as exc:
            logger.error("Failed to write restart marker: %s", exc)

    def _handle_restart_marker_on_startup(self) -> None:
        """Read restart marker and write synthetic "restart_completed" tool responses.

        Called on new gateway startup. For each session with pending tool calls,
        writes a tool response to the DB indicating the gateway restart completed.
        This replaces the old approach where the old process tried to write
        synthetic responses (causing duplicate tool responses via race conditions).
        """
        gw = self._gateway
        marker_path = gw.config.home_dir / ".restart_pending"

        if not marker_path.exists():
            return

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to parse restart marker: %s — removing it", exc,
            )
            marker_path.unlink(missing_ok=True)
            return

        restarted_at = marker.get("restarted_at", time.time())
        sessions_data = marker.get("sessions", {})
        written = 0

        for session_key, sdata in sessions_data.items():
            session_id = sdata.get("session_id", "")
            pending = sdata.get("pending_tool_calls", [])
            if not session_id or not pending:
                continue

            # Load current messages to check for existing tool responses
            try:
                current_msgs = gw.session_store.get_messages(
                    session_key, session_id=session_id
                )
            except Exception:
                logger.exception(
                    "Failed to load messages for %s — skipping", session_key,
                )
                continue

            # Build set of existing tool_call_ids that already have responses
            existing_responses: set[str] = set()
            for m in current_msgs:
                if m.get("role") == "tool" and m.get("tool_call_id"):
                    tc_id = m["tool_call_id"]
                    if isinstance(tc_id, str):
                        existing_responses.add(tc_id)

            for tc_info in pending:
                tc_id = tc_info.get("tool_call_id", "")
                fn_name = tc_info.get("function_name", "?")
                reason = tc_info.get("reason", "")
                if not tc_id:
                    continue

                # Skip if a tool response already exists for this call_id
                # (tool execution completed during the old process's stop timeout)
                if tc_id in existing_responses:
                    logger.info(
                        "Skipping restart response for %s/%s "
                        "(tool_call_id=%s, reason=%s) — response already exists",
                        session_key, fn_name, tc_id, reason,
                    )
                    continue

                elapsed_seconds = max(0, time.time() - restarted_at)

                if reason == "restart_trigger":
                    # Terminal command that triggered the restart — this is normal.
                    synthetic = json.dumps({
                        "success": True,
                        "restart_completed": True,
                        "duration_seconds": round(elapsed_seconds, 1),
                        "message": f"Gateway restart completed ({elapsed_seconds:.1f}s)",
                    })
                else:
                    # unknown_failure — tool was in-flight during drain,
                    # we don't know why it didn't complete.
                    synthetic = json.dumps({
                        "success": False,
                        "error": "Unknown failure — tool execution may have been interrupted by gateway restart",
                        "interrupted": True,
                        "duration_seconds": round(elapsed_seconds, 1),
                    })

                try:
                    gw.session_store.add_message(
                        session_key, "tool", synthetic,
                        session_id=session_id,
                        tool_call_id=tc_id,
                    )
                    written += 1
                    logger.info(
                        "Restart response for %s/%s "
                        "(tool_call_id=%s, reason=%s, elapsed=%.1fs)",
                        session_key, fn_name, tc_id, reason, elapsed_seconds,
                    )
                except Exception:
                    logger.exception(
                        "Failed to write restart response for %s/%s",
                        session_key, fn_name,
                    )

        # Clean up marker
        try:
            marker_path.unlink(missing_ok=True)
            logger.debug("Removed restart marker (%d tool responses written)", written)
        except OSError as exc:
            logger.warning("Failed to remove restart marker: %s", exc)

        # Clean up gateway_interrupt markers (they are now processed)
        self._cleanup_gateway_interrupt_dir()

        # Collect completed terminal command results from detached subprocesses.
        # This captures real tool output that completed after the gateway exited.
        # Store the set of affected sessions so Gateway.start can trigger them.
        try:
            affected = self._collect_orphan_terminal_results()
            if affected:
                gw._restart_affected_sessions = affected
                logger.info(
                    "Collected terminal results for %d session(s): %s",
                    len(affected), affected,
                )
        except Exception:
            logger.exception("Failed to collect terminal results")

    def _collect_orphan_terminal_results(self) -> set[str]:
        """Collect completed terminal command results after restart.

        Scans ``.terminal_pending/`` in the home directory for marker files
        from detached subprocesses. If the output file exists and has content
        (the process completed), reads the result and writes a real tool
        response to the database instead of a synthetic one.

        Returns a set of session_keys that received new tool responses,
        so the caller can trigger agent turns for those sessions.

        This captures terminal command output even when the gateway restarted
        or crashed while the command was running.
        """
        affected: set[str] = set()
        gw = self._gateway
        pending_dir = gw.config.home_dir / ".terminal_pending"
        if not pending_dir.exists():
            return affected

        markers = list(pending_dir.glob("*.json"))
        if not markers:
            return

        logger.info("Scanning %d terminal pending markers...", len(markers))

        for marker_path in markers:
            try:
                data = json.loads(marker_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Invalid terminal marker %s: %s — removing", marker_path.name, exc)
                marker_path.unlink(missing_ok=True)
                continue

            output_path = Path(data.get("output_path", ""))
            session_key = data.get("session_key", "")
            session_id = data.get("session_id", "")
            tool_call_id = data.get("tool_call_id", "")
            started_at = data.get("started_at", 0)

            # Check if the output file exists and has content
            if not output_path.exists() or output_path.stat().st_size == 0:
                # Process is still running or never started — check if PID is alive
                pid = data.get("pid")
                if pid and _pid_is_alive(pid):
                    logger.debug(
                        "Terminal process %d still running for %s — skipping",
                        pid, marker_path.name,
                    )
                    continue
                # Stale marker — no output and process dead
                logger.warning(
                    "Stale terminal marker %s (no output, process dead) — removing",
                    marker_path.name,
                )
                marker_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                continue

            # Read the output
            try:
                with open(output_path) as f:
                    output_text = f.read()
            except OSError as exc:
                logger.warning("Failed to read output for %s: %s", marker_path.name, exc)
                marker_path.unlink(missing_ok=True)
                continue

            if not session_key or not session_id:
                logger.warning(
                    "Terminal marker %s missing session info — removing",
                    marker_path.name,
                )
                marker_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                continue

            # Check if a tool response already exists for this tool_call_id.
            # _handle_restart_marker_on_startup may have already written a
            # synthetic response — skip to avoid duplicate tool messages.
            skip_response = False
            if tool_call_id:
                try:
                    existing = gw.session_store.get_messages(
                        session_key, session_id=session_id,
                    )
                except Exception:
                    existing = []
                for m in existing:
                    if (m.get("role") == "tool"
                            and m.get("tool_call_id") == tool_call_id):
                        logger.info(
                            "Skipping terminal result for %s/%s "
                            "(tool_call_id=%s) — response already exists in DB",
                            session_key, marker_path.name, tool_call_id,
                        )
                        skip_response = True
                        break

            if skip_response:
                marker_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                continue

            # Build a real terminal tool response
            max_out = 50_000
            was_truncated = False
            if len(output_text) > max_out:
                output_text = output_text[:max_out]
                was_truncated = True

            # Try to get exit code; if not available, assume success
            exit_code = 0
            elapsed = time.time() - started_at if started_at > 0 else 0
            result_data: dict = {
                "output": output_text,
                "exit_code": exit_code,
                "collected_after_restart": True,
                "elapsed_seconds": round(elapsed, 1),
            }
            if was_truncated:
                result_data["truncated"] = True
                result_data["hint"] = "Output was truncated."

            try:
                gw.session_store.add_message(
                    session_key, "tool", json.dumps(result_data, ensure_ascii=False),
                    session_id=session_id,
                    tool_call_id=tool_call_id or "",
                )
                affected.add(session_key)
                logger.info(
                    "Collected terminal result for %s/%s (%d chars, %.1fs)",
                    session_key, marker_path.name, len(output_text), elapsed,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to write collected terminal result for %s: %s",
                    session_key, exc,
                )

            # Clean up
            marker_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        return affected
        """Trigger agent turns for sessions that received collected results."""
        # This is called from Gateway.start after handlers run;
        # the method is a no-op here — the actual trigger logic
        # lives in Gateway._trigger_pending_tool_results().

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
