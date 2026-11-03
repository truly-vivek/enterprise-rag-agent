from mcp.server.fastmcp import FastMCP

from app.main import search_knowledge_base

server = FastMCP("enterprise-rag-agent")
server.tool()(search_knowledge_base)


if __name__ == "__main__":
    server.run()
