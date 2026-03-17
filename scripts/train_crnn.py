"""Prepare recognition data and train CRNN."""

import argparse


def _todo_message():
    return (
        "CRNN full training pipeline has been modularized into src/ modules. "
        "Next step is wiring your exact augmentation, dataloader, and hyperparameters "
        "from the notebook into this script."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"CRNN script scaffold ready (epochs={args.epochs}, batch_size={args.batch_size}).")
    print(_todo_message())


if __name__ == "__main__":
    main()
