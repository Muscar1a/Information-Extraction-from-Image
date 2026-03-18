# image-information-extraction

OCR pipeline project for text detection and recognition from scene images.

## Project Structure

```text
image-information-extraction/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- main.py
|-- data/
|   |-- raw/
|   \-- processed/
|-- notebooks/
|   \-- exploration.ipynb
|-- scripts/
|   |-- prepare_detection_data.py
|   |-- train_yolo.py
|   |-- train_crnn.py
|   \-- compare_models.py
\-- src/
	|-- __init__.py
	|-- config.py
	|-- dataset.py
	|-- model_crnn.py
	|-- train_crnn.py
	|-- data/
	|   |-- __init__.py
	|   |-- loader.py
	|   \-- preprocess.py
	|-- models/
	|   |-- __init__.py
	|   \-- model.py
	|-- utils/
	|   |-- __init__.py
	|   \-- utils.py
	\-- inference.py
```

## Migration Guide From Notebook

- Put XML parsing and dataset loading logic in `src/data/loader.py`.
- Put conversion/cropping/encoding logic in `src/data/preprocess.py`.
- Keep model architecture in `src/models/model.py` (currently bridged to `src/model_crnn.py`).
- Put generic helper functions in `src/utils/utils.py`.
- Put end-to-end inference entry points in `src/inference.py`.
- Use files under `scripts/` as runnable stages.

## Quick Start

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Muscar1a/Information-Extraction-from-Image.git
cd Information-Extraction-from-Image
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/MacOS:
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Extract bounding boxes from XML files and convert them into YOLO format. Ensure your datasets are placed in the `datasets/` directory.

```bash
python scripts/prepare_detection_data.py
```

### 2. Text Detection (YOLO)
Train the YOLO model to detect text regions from scene images.

```bash
python scripts/train_yolo.py
```

### 3. Text Recognition (CRNN)
Crop the detected text regions and train the CRNN model to recognize the characters within them.

```bash
python scripts/train_crnn.py
```

### 4. Evaluation & Comparison
Compare the accuracy and performance of different pipelines (e.g., YOLO + CRNN vs YOLO + TrOCR vs EasyOCR).

```bash
python scripts/compare_models.py
```

### 5. Inference
Run the trained end-to-end pipeline on new images to extract text.

```bash
python src/inference.py --image path/to/your/image.jpg
```
