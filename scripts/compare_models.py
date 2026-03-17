"""Compare YOLO+CRNN, YOLO+TrOCR, and EasyOCR."""

import argparse


def _todo_message():
    return (
        "Comparison pipeline scaffold is ready. "
        "Wire your model-loading section and test_data construction from notebook "
        "before running full benchmark."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Compare script scaffold ready (conf_threshold={args.conf_threshold}).")
    print(_todo_message())


if __name__ == "__main__":
    main()
