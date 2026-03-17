"""General utility helpers.

Suggested mapping from notebook:
- Metrics functions: Cell 59
- IoU matching: Cell 66
"""

import Levenshtein
import numpy as np
import random
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_char_accuracy(pred, gt):
    """Calculate character-level accuracy using Levenshtein distance."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0

    distance = Levenshtein.distance(pred.lower(), gt.lower())
    accuracy = 1 - (distance / max(len(pred), len(gt)))
    return max(0, accuracy)


def calculate_word_accuracy(pred, gt):
    """Calculate exact word match accuracy"""
    return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0


def calculate_metrics(predictions, ground_truths):
    """Calculate average metrics for a list of predictions"""
    char_accs = []
    word_accs = []
    
    for pred, gt in zip(predictions, ground_truths):
        char_accs.append(calculate_char_accuracy(pred, gt))
        word_accs.append(calculate_word_accuracy(pred, gt))
    
    return {
        'char_accuracy': np.mean(char_accs) * 100,
        'word_accuracy': np.mean(word_accs) * 100,
        'total_samples': len(predictions)
    }