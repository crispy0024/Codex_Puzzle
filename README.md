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

## Tests

Simple unit tests are provided for the main utilities. Execute them with:

```bash
pytest
```

These tests do not require the original puzzle images and serve only as sanity checks for the library functions.

## Web Interface

A simple Flask application is provided in `webapp/` to try the puzzle utilities
from the browser. Run the server with:

```bash
python -m webapp.app
```

Open `http://localhost:5000` and upload an image of a puzzle piece. The page
shows the segmented piece and reports whether it is classified as a corner,
edge or middle piece.


