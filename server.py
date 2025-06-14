from puzzle.api import create_app, canvas_items, merge_history, EdgeFeatures, PieceFeatures, PieceGroup, merge_groups

app = create_app()

__all__ = [
    "app",
    "canvas_items",
    "merge_history",
    "EdgeFeatures",
    "PieceFeatures",
    "PieceGroup",
    "merge_groups",
]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
