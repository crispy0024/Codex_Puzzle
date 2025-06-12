# Codex_Puzzle

This repository contains notebooks experimenting with computer vision techniques for jigsaw puzzle pieces. The goal is to explore ways to segment images of puzzles and attempt reassembly programmatically.

## Contents

- **Kaggle_Jigsaw.ipynb** – Demonstrates thresholding and contour detection with OpenCV. The notebook walks through creating bounding boxes around individual pieces and saving them as separate images.
- **Masking_puzzle.ipynb** – A larger workflow that connects to Google Drive, removes backgrounds with GrabCut segmentation, performs feature extraction and corner detection, and includes sample code to display selected puzzle pieces.

Both notebooks are rough experiments used to test Codex for a puzzle project. They are not yet cleaned up for production use, but serve as a starting point for future development.
