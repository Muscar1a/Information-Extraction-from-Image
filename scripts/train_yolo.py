"""Train and evaluate YOLO model."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="yolo_data/data.yml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default="yolo11m.pt")
    args = parser.parse_args()

    yolo_model = YOLO(args.base_model)
    yolo_model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        cache=True,
        patience=20,
        plots=True,
    )

    best_path = "runs/detect/train/weights/best.pt"
    trained = YOLO(best_path)
    metrics = trained.val()
    print(metrics)


if __name__ == "__main__":
    main()
