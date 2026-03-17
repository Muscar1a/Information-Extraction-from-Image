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

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run staged pipelines from `scripts/`.

