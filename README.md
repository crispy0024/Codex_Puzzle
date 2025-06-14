# Codex_Puzzle

This repository contains notebooks and Python modules experimenting with computer vision techniques for jigsaw puzzle pieces. The goal is to explore ways to segment images of puzzles and attempt reassembly programmatically.

## Contents

- **Kaggle_Jigsaw.ipynb** – Demonstrates thresholding and contour detection with OpenCV. The notebook walks through creating bounding boxes around individual pieces and saving them as separate images.
- **Masking_puzzle.ipynb** – A larger workflow that removes backgrounds with GrabCut segmentation, performs feature extraction and corner detection, and includes sample code to display selected puzzle pieces.
- **puzzle/** – Python package containing reusable functions for segmentation, feature extraction and puzzle assembly.

## Setup

The project requires Python 3.8+ with the packages listed in `requirements.txt`. This includes `flask-cors` for enabling cross-origin requests from the frontend. To install the dependencies run:

```bash
./setup.sh
```

For contributors who want to hack on the code, install the package in editable
mode so local changes are picked up automatically:

```bash
pip install -e .
```

## Tests

Simple unit tests are provided for the main utilities. After installing the
package in editable mode, run the tests with:

```bash
pytest
```

These tests do not require the original puzzle images and serve only as sanity checks for the library functions.

## Next.js Frontend

A minimal frontend built with [Next.js](https://nextjs.org/) is included under the
`frontend/` directory. To start the development server run:

```bash
cd frontend
npm install
npm run dev
```

The site will be available at `http://localhost:3000`.


## Flask API

A lightweight Flask application exposes puzzle utilities. The app uses
`flask-cors` so the frontend can call the API from another origin. Make sure the
`flask-cors` package is installed before launching the server. Start it with:

```bash
python server.py
```

By default it runs on port 5000. Several endpoints are available that each
perform one step of the workflow:

- `/remove_background` – segment the puzzle piece and return the result and mask. Optional `lower` and `upper` form fields provide grayscale thresholds for the background.
- `/detect_corners` – highlight detected corners on the piece. Accepts the same `lower` and `upper` fields.
- `/classify_piece` – return whether the piece is a corner, edge or middle piece. Accepts the same `lower` and `upper` fields.
- `/edge_descriptors` – compute simple metrics for each edge. Accepts the same `lower` and `upper` fields.
- `/segment_pieces` – split an image containing several pieces into
  individual crops. Optional `threshold`, `kernel_size` and `use_hull`
  fields control the binary threshold, smoothing kernel size and whether
  bounding boxes use the contour convex hull.

The included Next.js site provides buttons that call these endpoints
individually so you can inspect the output of every stage.

## Feature Extraction Pipeline

Each puzzle edge is analyzed to produce several descriptors:

- **Edge type** – labeled as `tab`, `hole` or `flat` by comparing the piece
  contour to its convex hull.
- **Color histogram** – HSV histogram sampled from a thin strip along the edge.
- **Hu moments** – shape moments of the same edge segment capturing its curve.
- **Color profile** – average HSV values across that strip.

The dataclasses `EdgeFeatures` and `PieceFeatures` store these metrics for each
piece. `extract_edge_descriptors` returns dictionaries containing the vectors for
all four edges in order.

## Piece Compatibility Scoring

The `puzzle.scoring` module implements helpers to compare edges and rank
potential matches.

* `shape_similarity` – computes a distance between Hu moments from two edges.
  A smaller value means the edge shapes are closer.
* `color_similarity` – compares edge colors either by histogram or a simple HSV
  profile.
* `compatibility_score` – combines shape and color distances and returns a
  single value, lower being a better fit.

To evaluate all edges of a set of pieces you can call `top_n_matches` which
returns the best candidate pairs per edge sorted by score.

```python
from puzzle.scoring import top_n_matches
from puzzle.features import PieceFeatures, EdgeFeatures

pieces = [PieceFeatures(...), PieceFeatures(...)]
matches = top_n_matches(pieces, n=3)
```

Each entry in `matches` maps `(piece_id, edge_index)` to a list of matching
`(other_piece_id, other_edge_index, score)` tuples.

## Additional Notes

All puzzle processing endpoints are served from `server.py` using Flask on
port `5000`. The `Segment Pieces` and `Segment Selected` buttons in the
frontend call the `/segment_pieces` route to split an image containing
multiple pieces into individual crops. The route now applies a closing
and opening step to smooth shapes and accepts optional `threshold`,
`kernel_size` and `use_hull` parameters. Be sure to start the Flask server with
`python server.py` before using the Next.js interface.

## Reinforcement Learning Trainer

Feedback from the UI is stored in `feedback.jsonl`. To update the policy run:

```bash
python train_rl.py
```

This uses Stable Baselines3 to train a PPO model and saves the weights to
`rl_model.zip`.

## Interactive Suggestion Loop

1. Start the backend with `python server.py`.
2. In another terminal run the frontend from `frontend/` using `npm run dev` and open `http://localhost:3000`.
3. Select a piece on the canvas to request matches via the `/suggest_match` endpoint.
4. Use the **Accept** or **Reject** buttons to send feedback through `/submit_feedback`.
5. Periodically execute `python train_rl.py` (or call `/train_rl`) to refine the model with the logged feedback.

## References

- [Godot puzzle demo](https://github.com/godotengine/godot-demo-projects/tree/master/2d/puzzle) – example of snapping behaviour in a dedicated engine.
- [Maxim Terleev’s article on jigsaw puzzle scoring](https://habr.com/ru/articles/197012/) – inspiration for improving edge matching heuristics.




## License
Released under the [MIT License](LICENSE).
