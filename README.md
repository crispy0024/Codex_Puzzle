# Codex_Puzzle

This repository contains notebooks and Python modules experimenting with computer vision techniques for jigsaw puzzle pieces. The goal is to explore ways to segment images of puzzles and attempt reassembly programmatically.

## Contents

- **Kaggle_Jigsaw.ipynb** – Demonstrates thresholding and contour detection with OpenCV. The notebook walks through creating bounding boxes around individual pieces and saving them as separate images.
- **Masking_puzzle.ipynb** – A larger workflow that removes backgrounds with GrabCut segmentation, performs feature extraction and corner detection, and includes sample code to display selected puzzle pieces.
- **puzzle/** – Python package containing reusable functions for segmentation, feature extraction and puzzle assembly.

## Setup

The project requires Python 3.8+ with the packages listed in `requirements.txt`. To install them run:

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

A lightweight Flask application exposes puzzle utilities. Start it with:

```bash
python server.py
```

By default it runs on port 5000. Several endpoints are available that each
perform one step of the workflow:

- `/remove_background` – segment the puzzle piece and return the result and mask
- `/detect_corners` – highlight detected corners on the piece
- `/classify_piece` – return whether the piece is a corner, edge or middle piece
- `/edge_descriptors` – compute simple metrics for each edge

The included Next.js site provides buttons that call these endpoints
individually so you can inspect the output of every stage.

## Additional Notes

All puzzle processing endpoints are served from `server.py` using Flask on
port `5000`. The `Segment Pieces` button in the frontend calls the
`/segment_pieces` route to split an image containing multiple pieces into
individual crops.




## License
Released under the [MIT License](LICENSE).
