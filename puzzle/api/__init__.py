from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient as _StarletteTC


class _PatchedClient(_StarletteTC):
    """Compatibility layer for older tests expecting data=file tuples."""

    def post(self, url, data=None, **kwargs):
        if isinstance(data, dict):
            files = {}
            form = {}
            for key, val in data.items():
                if isinstance(val, tuple) and len(val) >= 2 and hasattr(val[0], "read"):
                    fileobj, filename = val[0], val[1]
                    content_type = (
                        val[2] if len(val) > 2 else "application/octet-stream"
                    )
                    files[key] = (filename, fileobj, content_type)
                else:
                    form[key] = val
            if files:
                kwargs.setdefault("files", files)
                data = form
        return super().post(url, data=data, **kwargs)


# monkey patch the TestClient used in tests
import fastapi.testclient as _ftc

_ftc.TestClient = _PatchedClient

from .segmentation_api import router as segmentation_router
from .canvas_api import router as canvas_router
from .rl_api import router as rl_router
from .state import canvas_items, merge_history
from ..features import EdgeFeatures, PieceFeatures
from ..group import PieceGroup, merge_groups


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(segmentation_router)
    app.include_router(canvas_router)
    app.include_router(rl_router)
    return app


# default application instance
app = create_app()

# expose commonly used classes for tests
__all__ = [
    "create_app",
    "app",
    "canvas_items",
    "merge_history",
    "EdgeFeatures",
    "PieceFeatures",
    "PieceGroup",
    "merge_groups",
]
