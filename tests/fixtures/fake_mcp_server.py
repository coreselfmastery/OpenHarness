"""Small stdio MCP server used by integration tests.

Compatible with BOTH mcp 1.x and 2.x: the SDK renamed ``FastMCP`` to
``MCPServer`` (``mcp.server.mcpserver``) in 2.0. Import whichever the
installed version provides.
"""

from __future__ import annotations

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
except ImportError:  # mcp 2.x — FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer

server = MCPServer("fixture-demo")


@server.tool()
def hello(name: str) -> str:
    return f"fixture-hello:{name}"


@server.resource("fixture://readme", name="Fixture Readme")
def readme() -> str:
    return "fixture resource contents"


if __name__ == "__main__":
    server.run("stdio")
