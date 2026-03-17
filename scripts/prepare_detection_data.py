"""Run detection data preparation steps."""

import argparse
import os

import yaml
from sklearn.model_selection import train_test_split

from src.data.loader import extract_data_from_xml
from src.data.preprocess import convert_to_yolo_format, save_yolo_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="datasets/SceneTrialTrain", help="Path to dataset root"
    )
    parser.add_argument(
        "--output-dir", default="yolo_data", help="Path to YOLO output directory"
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    words_xml_path = os.path.join(args.dataset_dir, "words.xml")
    image_paths, image_sizes, _, bounding_boxes = extract_data_from_xml(words_xml_path)

    yolo_data = convert_to_yolo_format(image_paths, image_sizes, bounding_boxes)
    train_yolo_data, val_yolo_data = train_test_split(
        yolo_data, test_size=args.val_size, random_state=42
    )

    save_yolo_data(train_yolo_data, "train", args.output_dir, args.dataset_dir)
    save_yolo_data(val_yolo_data, "val", args.output_dir, args.dataset_dir)

    data_yaml = {
        "path": os.path.abspath(args.output_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["text"],
    }

    yolo_yaml_path = os.path.join(args.output_dir, "data.yml")
    with open(yolo_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Prepared YOLO dataset at: {args.output_dir}")
    print(f"YOLO config: {yolo_yaml_path}")


if __name__ == "__main__":
    main()
