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
  individual crops. Optional `threshold` and `kernel_size` fields control the
  binary threshold and smoothing kernel size used during detection.

The included Next.js site provides buttons that call these endpoints
individually so you can inspect the output of every stage.

## Additional Notes

All puzzle processing endpoints are served from `server.py` using Flask on
port `5000`. The `Segment Pieces` and `Segment Selected` buttons in the
frontend call the `/segment_pieces` route to split an image containing
multiple pieces into individual crops. The route now applies a closing
and opening step to smooth shapes and accepts optional `threshold` and
`kernel_size` parameters. Be sure to start the Flask server with
`python server.py` before using the Next.js interface.




## License
Released under the [MIT License](LICENSE).
