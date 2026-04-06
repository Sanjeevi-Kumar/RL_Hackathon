"""
FastAPI application for the Warehouse Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .warehouse_environment import WarehouseEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.warehouse_environment import WarehouseEnvironment

# Create the app with MCP types for action/observation
app = create_app(
    WarehouseEnvironment, CallToolAction, CallToolObservation, env_name="warehouse_env"
)


def main():
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
