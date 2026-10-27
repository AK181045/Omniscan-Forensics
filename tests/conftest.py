"""
=============================================================================
SENTINELS OF INTEGRITY — Pytest Configuration
=============================================================================
"""

import pytest
import asyncio
import sys
import importlib
from importlib.abc import MetaPathFinder


class ModuleMirror(MetaPathFinder):
    """
    Ensures that imports of 'app.xxx' and 'api.app.xxx' always resolve to
    the exact same module instances in sys.modules, preventing Pydantic
    cross-module validation errors.
    """
    def find_spec(self, fullname, path, target=None):
        if fullname == "app" or fullname.startswith("app."):
            long_k = "api." + fullname
            if long_k in sys.modules:
                sys.modules[fullname] = sys.modules[long_k]
                return getattr(sys.modules[long_k], "__spec__", None)
            try:
                mod = importlib.import_module(long_k)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except Exception:
                pass
        elif fullname == "api.app" or fullname.startswith("api.app."):
            short_k = fullname[4:]
            if short_k in sys.modules:
                sys.modules[fullname] = sys.modules[short_k]
                return getattr(sys.modules[short_k], "__spec__", None)
        return None


if not any(isinstance(x, ModuleMirror) for x in sys.meta_path):
    sys.meta_path.insert(0, ModuleMirror())


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
