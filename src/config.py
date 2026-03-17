"""Project configuration and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "SceneTrialTrain"
YOLO_DATA_DIR = PROJECT_ROOT / "yolo_data"
CROPPED_TEXT_DIR = PROJECT_ROOT / "cropped_text"
YOLO_BEST_WEIGHT = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
CRNN_WEIGHT = PROJECT_ROOT / "ocr_crnn.pt"
