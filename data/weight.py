import torch
import numpy as np
from collections import Counter
import logging
logger = logging.getLogger(__name__)

def calculate_inverse_frequency_weights(labels, num_classes, device='cpu', strategy='inverse_frequency'):
    if strategy != 'inverse_frequency':
        return None
    if labels is None or len(labels) == 0:
        pass
        return None
    if num_classes <= 0:
        pass
        return None
    label_counts = Counter(labels)
    total_samples = len(labels)
    weights = torch.zeros(num_classes, device=device)
    missing_classes = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            weights[i] = 1.0
            missing_classes.append(i)
        else:
            weights[i] = total_samples / (num_classes * count)
    if missing_classes:
        pass
    return weights

def calculate_bce_pos_weight(labels, device='cpu', strategy='inverse_frequency'):
    if strategy != 'inverse_frequency':
        return None
    if labels is None or len(labels) == 0:
        pass
        return None
    label_counts = Counter(labels)
    count_neg = label_counts.get(0, 0)
    count_pos = label_counts.get(1, 0)
    pos_weight_val = 1.0
    if count_pos == 0:
        pass
    elif count_neg == 0:
        pass
    else:
        pos_weight_val = count_neg / count_pos
    pos_weight = torch.tensor([pos_weight_val], device=device)
    return pos_weight
