"""
ThunderAgent - VLLM proxy with program state tracking.

Module structure:
- backend: Backend state management
- program: Program state management
- scheduler: Request routing and proxying
"""

from typing import TYPE_CHECKING

from .config import Config, get_config, set_config
from .backend import BackendState
from .program import ProgramState, ProgramStatus
from .scheduler import MultiBackendRouter

if TYPE_CHECKING:
    from .app import get_program_id, register_routes

# app.py builds a router at module level, so it must not be imported until
# __main__ has applied the CLI config. These names load it on first access.
_LAZY_APP_NAMES = ("get_program_id", "register_routes")


def __getattr__(name):
    if name == "app" or name in _LAZY_APP_NAMES:
        import importlib

        # import_module, not "from . import app": the latter checks the
        # package's attributes first, which would call __getattr__ again.
        app = importlib.import_module(".app", __name__)
        return app if name == "app" else getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_APP_NAMES))


__all__ = [
    "Config",
    "get_config",
    "set_config",
    "BackendState",
    "ProgramState",
    "ProgramStatus",
    "MultiBackendRouter",
    "get_program_id",
    "register_routes",
]

__version__ = "0.2.0"
