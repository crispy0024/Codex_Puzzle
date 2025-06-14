from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
