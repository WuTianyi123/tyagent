"""Gateway subsystem — message routing, streaming consumer, and platform lifecycle."""
from tyagent.gateway.consumer import StreamConsumer
from tyagent.gateway.gateway import Gateway, register_platform, run_gateway

__all__ = ["Gateway", "StreamConsumer", "register_platform", "run_gateway"]
