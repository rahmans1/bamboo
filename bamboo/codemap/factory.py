"""Code Map plugin registry.

Mirrors :mod:`bamboo.agents.extractors.factory` so the two plugin systems read
the same way.  Registration happens at import time; the built-in PanDA plugin
is registered lazily on first lookup so importing this module does not pull in
the AST machinery.
"""

from __future__ import annotations

import logging

from bamboo.codemap.base import CodeMapPlugin

logger = logging.getLogger(__name__)

# Registry: map_id → plugin class
_plugins: dict[str, type[CodeMapPlugin]] = {}


def register_code_map_plugin(map_id: str, plugin_class: type[CodeMapPlugin]) -> None:
    """Register *plugin_class* under *map_id*.

    Re-registering the same id replaces the previous entry, which lets tests
    inject a stub.
    """
    _plugins[map_id.lower()] = plugin_class
    logger.debug("Registered Code Map plugin: %s", map_id)


def _register_builtins() -> None:
    if "panda" not in _plugins:
        from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

        register_code_map_plugin("panda", PandaCodeMapPlugin)
    if "swf-epicprod" not in _plugins:
        from bamboo.codemap.epicprod.plugin import EpicprodCodeMapPlugin

        register_code_map_plugin("swf-epicprod", EpicprodCodeMapPlugin)


def get_code_map_plugin(map_id: str = "panda") -> CodeMapPlugin:
    """Return an instantiated plugin for *map_id*.

    Raises:
        ValueError: If no plugin is registered under that id.
    """
    _register_builtins()
    plugin_class = _plugins.get(map_id.lower())
    if plugin_class is None:
        raise ValueError(
            f"No Code Map plugin registered for {map_id!r}. "
            f"Available: {sorted(_plugins)}"
        )
    return plugin_class()


def available_map_ids() -> list[str]:
    """Return the registered map ids."""
    _register_builtins()
    return sorted(_plugins)
