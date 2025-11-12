import numpy as np
import random

def add_gaussian_noise(signal, std=0.01):
    noise = np.random.normal(0, std, signal.shape)
    return signal + noise

def add_salt_pepper_noise(signal, amount=0.01, salt_vs_pepper=0.5):
    noisy_signal = np.copy(signal)
    num_salt = np.ceil(amount * signal.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * signal.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in signal.shape if i > 1]
    if coords and len(coords) == 2:
        noisy_signal[coords[0], coords[1]] = np.max(signal)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in signal.shape if i > 1]
    if coords and len(coords) == 2:
        noisy_signal[coords[0], coords[1]] = np.min(signal)
    return noisy_signal

def adjust_abnormal_positions(patches, labels, dynamic_features, type_to_label):
    augmented_sequences = []
    num_patches = len(patches)
    if num_patches < 2:
        return []
    n_label = type_to_label.get('N', -1)
    abnormal_indices = [i for i, lbl in enumerate(labels) if lbl != n_label and n_label != -1]
    if not abnormal_indices:
        return []
    if abnormal_indices[0] != 0:
        idx_to_move = abnormal_indices[0]
        new_indices = [idx_to_move] + [i for i in range(num_patches) if i != idx_to_move]
        new_patches = [patches[i] for i in new_indices]
        new_labels = [labels[i] for i in new_indices]
        new_dyn_feats = [dynamic_features[i] for i in new_indices] if dynamic_features else None
        augmented_sequences.append((new_patches, new_labels, new_dyn_feats))
    last_abnormal_idx = abnormal_indices[-1]
    if last_abnormal_idx != num_patches - 1:
        new_indices = [i for i in range(num_patches) if i != last_abnormal_idx] + [last_abnormal_idx]
        new_patches = [patches[i] for i in new_indices]
        new_labels = [labels[i] for i in new_indices]
        new_dyn_feats = [dynamic_features[i] for i in new_indices] if dynamic_features else None
        augmented_sequences.append((new_patches, new_labels, new_dyn_feats))
    return augmented_sequences
