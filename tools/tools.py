"""Tools dictionary for backward compatibility and easy access."""

from .registry import execute_tool

# Legacy tools dictionary for easy migration
TOOLS = {
    "web_search": lambda q: execute_tool("web_search", q),
    "summarize": lambda f: execute_tool("summarize", f),
    "note": lambda t: execute_tool("note", t, tags=["user"]),
    "analyze": lambda q: execute_tool("analyze", q),
    "extract": lambda f: execute_tool("extract", f),
    "create": lambda t: execute_tool("create", t, tags=["auto"])
}
