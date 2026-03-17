import os
import shutil

import cv2
import torch


def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for img_path, img_size, bbs in zip(image_paths, image_sizes, bounding_boxes):
        img_w, img_h = img_size
        yolo_bbs = []

        for bb in bbs:
            x, y, w, h = bb
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_bbs.append([0, x_center, y_center, w_norm, h_norm])

        yolo_data.append((img_path, yolo_bbs))

    return yolo_data


def save_yolo_data(data, split, save_dir, dataset_dir):
    split_dir = os.path.join(save_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for img_path, yolo_bbs in data:
        img_name = img_path.replace("/", "_")
        img_src = os.path.join(dataset_dir, img_path)
        img_dest = os.path.join(images_dir, img_name)

        shutil.copy(img_src, img_dest)

        label_name = img_name.replace(".JPG", ".txt")
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w", encoding="utf-8") as f:
            for bb in yolo_bbs:
                f.write(f"{bb[0]} {bb[1]} {bb[2]} {bb[3]} {bb[4]}\n")


def crop_text(img_paths, img_labels, bboxes, save_dir="cropped_text"):
    os.makedirs(save_dir, exist_ok=True)

    cropped_paths = []
    labels = []

    for img_path, img_label, bbox in zip(img_paths, img_labels, bboxes):
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        for i, (bb, label) in enumerate(zip(bbox, img_label)):
            x, y, w, h = bb
            x, y, w, h = int(x), int(y), int(w), int(h)

            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            x_end = min(img.shape[1], x + w)
            y_end = min(img.shape[0], y + h)

            if x_end <= x or y_end <= y:
                print(f"Warning: Invalid bounding box for {img_path}")
                continue

            cropped_img = img[y:y_end, x:x_end]

            if cropped_img.size == 0:
                print(f"Warning: Empty crop for {img_path} at bbox {bb}")
                continue

            img_name = os.path.basename(img_path).replace(".JPG", "")
            save_path = os.path.join(save_dir, f"{img_name}_{i}.jpg")

            cv2.imwrite(save_path, cropped_img)

            cropped_paths.append(save_path)
            labels.append(label)

    return cropped_paths, labels


def encode(text, char_to_idx, max_label_len):
    encoded = []
    for char in text:
        encoded.append(char_to_idx[char])

    label_len = len(encoded)
    encoded += [0] * (max_label_len - len(encoded))

    return torch.LongTensor(encoded), torch.tensor(label_len, dtype=torch.long)


def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


def decode_label(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)

        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences
