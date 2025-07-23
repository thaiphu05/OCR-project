# OCR Project

## Introduction
This project performs Optical Character Recognition (OCR) for invoices, tables, and document images. The system uses PaddleOCR for text recognition, Table Transformer for table detection, and optionally post-processes results with an LLM (OpenAI GPT).

## Features
- Text recognition in images (PaddleOCR)
- Table detection and extraction (Table Transformer)
- AI-powered OCR post-processing (OpenAI GPT, optional)
- Supports Vietnamese invoices, tables, and documents

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd OCR
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download PaddleOCR models:**
   - Download the models to the corresponding folders as specified in `models/infer_PaddleOCR.py` (or update the paths to match your machine):
     - `PP-LCNet_x1_0_textline_ori`
     - `PP-OCRv5_server_det`
     - `PP-OCRv5_server_rec`
   - Reference: https://github.com/PaddlePaddle/PaddleOCR

## Usage
Run a sample image through the pipeline:
```bash
python main.py
```
- Sample images are in the `data/` folder
- Results will be printed to the console: text outside tables and recognized tables

## Main Structure
- `main.py`: Runs the OCR and Table Detection pipeline
- `models/`: Contains PaddleOCR, Table Detection, and LLM postprocessing classes
- `utils/`: Image processing, bounding box, and result merging utilities
- `data/`: Sample images

## Notes
- To use LLM post-processing, you need an OpenAI API key and the `OPENAI_API_KEY` environment variable set.
- Update PaddleOCR model paths in the code to match your local setup.
