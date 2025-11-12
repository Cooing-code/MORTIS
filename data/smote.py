import numpy as np
from collections import Counter
import logging
import random
import torch
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)

def apply_smote(patches, labels, dynamic_features=None, k_neighbors=5, sampling_strategy='auto', smote_type='regular'):
    if isinstance(patches, np.ndarray):
        patches_list = [patches[i] for i in range(patches.shape[0])]
    else:
        patches_list = patches
    if isinstance(labels, np.ndarray):
        labels_list = labels.tolist()
    else:
        labels_list = labels
    dynamic_features_list = None
    if dynamic_features is not None:
        if isinstance(dynamic_features, np.ndarray):
            dynamic_features_list = [dynamic_features[i] for i in range(dynamic_features.shape[0])]
        else:
            dynamic_features_list = dynamic_features
    if dynamic_features is not None:
        pass
    if len(patches_list) == 0 or len(np.unique(labels_list)) <= 1:
        pass
        return (patches, labels, dynamic_features)
    class_counts = Counter(labels_list)
    target_counts = {}
    majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
    majority_count = class_counts[majority_class]
    if sampling_strategy == 'auto':
        for label in class_counts:
            target_counts[label] = majority_count
    elif sampling_strategy == 'not majority':
        for label in class_counts:
            if label == majority_class:
                target_counts[label] = class_counts[label]
            else:
                target_counts[label] = max(class_counts[label], majority_count // 2)
    elif sampling_strategy == 'minority':
        minority_class = min(class_counts.items(), key=lambda x: x[1])[0]
        for label in class_counts:
            if label == minority_class:
                target_counts[label] = majority_count // 2
            else:
                target_counts[label] = class_counts[label]
    elif isinstance(sampling_strategy, dict):
        target_counts = sampling_strategy
    else:
        raise ValueError(f'Unsupported sampling strategy: {sampling_strategy}')
    class_patches = {}
    class_features = {}
    for i, label in enumerate(labels_list):
        if label not in class_patches:
            class_patches[label] = []
            if dynamic_features_list is not None:
                class_features[label] = []
        class_patches[label].append(patches_list[i])
        if dynamic_features_list is not None:
            class_features[label].append(dynamic_features_list[i])
    oversampled_patches = []
    oversampled_labels = []
    oversampled_features = [] if dynamic_features_list is not None else None
    for label, target_count in target_counts.items():
        current_count = class_counts[label]
        if current_count >= target_count:
            oversampled_patches.extend(class_patches[label])
            oversampled_labels.extend([label] * current_count)
            if dynamic_features_list is not None:
                oversampled_features.extend(class_features[label])
            continue
        patches_to_generate = target_count - current_count
        if current_count < 2:
            for _ in range(patches_to_generate):
                idx = 0
                oversampled_patches.append(class_patches[label][idx])
                oversampled_labels.append(label)
                if dynamic_features_list is not None:
                    oversampled_features.append(class_features[label][idx])
            oversampled_patches.extend(class_patches[label])
            oversampled_labels.extend([label] * current_count)
            if dynamic_features_list is not None:
                oversampled_features.extend(class_features[label])
            continue
        X = np.array([p.flatten() for p in class_patches[label]])
        if smote_type == 'regular':
            k = min(k_neighbors + 1, len(X))
            nn = NearestNeighbors(n_neighbors=k).fit(X)
            distances, indices = nn.kneighbors(X)
        oversampled_patches.extend(class_patches[label])
        oversampled_labels.extend([label] * current_count)
        if dynamic_features_list is not None:
            oversampled_features.extend(class_features[label])
        for _ in range(patches_to_generate):
            if smote_type == 'regular':
                idx = random.randrange(len(X))
                nn_idx = random.choice(indices[idx][1:])
            else:
                idx = random.randrange(len(X))
                nn_idx = random.choice([i for i in range(len(X)) if i != idx])
            sample = class_patches[label][idx]
            nn_sample = class_patches[label][nn_idx]
            if sample.shape != nn_sample.shape:
                min_shape = tuple((min(s1, s2) for s1, s2 in zip(sample.shape, nn_sample.shape)))
                sample_temp = np.zeros(min_shape)
                nn_sample_temp = np.zeros(min_shape)
                sample_slices = tuple((slice(0, dim) for dim in min_shape))
                sample_temp = sample[sample_slices]
                nn_sample_temp = nn_sample[sample_slices]
                sample = sample_temp
                nn_sample = nn_sample_temp
            alpha = random.random()
            new_sample = sample + alpha * (nn_sample - sample)
            oversampled_patches.append(new_sample)
            oversampled_labels.append(label)
            if dynamic_features_list is not None:
                feat = class_features[label][idx]
                nn_feat = class_features[label][nn_idx]
                if feat.shape != nn_feat.shape:
                    min_shape = tuple((min(s1, s2) for s1, s2 in zip(feat.shape, nn_feat.shape)))
                    feat_temp = np.zeros(min_shape)
                    nn_feat_temp = np.zeros(min_shape)
                    feat_slices = tuple((slice(0, dim) for dim in min_shape))
                    feat_temp = feat[feat_slices]
                    nn_feat_temp = nn_feat[feat_slices]
                    feat = feat_temp
                    nn_feat = nn_feat_temp
                new_feat = feat + alpha * (nn_feat - feat)
                oversampled_features.append(new_feat)
    oversampled_patches_np = np.array(oversampled_patches)
    oversampled_labels_np = np.array(oversampled_labels)
    oversampled_features_np = np.array(oversampled_features) if oversampled_features is not None else None
    final_counts = Counter(oversampled_labels_np)
    if oversampled_features_np is not None:
        pass
    return (oversampled_patches_np, oversampled_labels_np, oversampled_features_np)

def calculate_stage_labels(labels, normal_label_index, abnormal_fine_to_stage2_map):
    stage1_labels = np.zeros(len(labels), dtype=np.int64)
    stage2_labels = np.full(len(labels), -100, dtype=np.int64)
    for i, label in enumerate(labels):
        if label != normal_label_index:
            stage1_labels[i] = 1
            stage2_labels[i] = abnormal_fine_to_stage2_map.get(label, -100)
    return (stage1_labels, stage2_labels)

def calculate_pos_weights(labels, stage1_labels=None, stage2_labels=None, is_two_stage=False, num_abnormal_classes=None):
    pos_weight_s1 = None
    pos_weight_s2 = None
    if is_two_stage:
        if stage1_labels is not None:
            s1_counts = Counter(stage1_labels)
            if len(s1_counts) == 2 and s1_counts.get(0, 0) > 0 and (s1_counts.get(1, 0) > 0):
                pos_weight_val = s1_counts[0] / s1_counts[1]
                pos_weight_s1 = torch.tensor([pos_weight_val])
        if stage2_labels is not None and num_abnormal_classes is not None:
            valid_stage2_labels = [lbl for lbl in stage2_labels if lbl != -100]
            if len(valid_stage2_labels) > 0:
                s2_counts = Counter(valid_stage2_labels)
                per_class_weights_s2 = torch.zeros(num_abnormal_classes)
                total_s2 = len(valid_stage2_labels)
                for class_idx_s2 in range(num_abnormal_classes):
                    count = s2_counts.get(class_idx_s2, 0)
                    if count == 0:
                        weight = 1.0
                    else:
                        weight = total_s2 / (num_abnormal_classes * count)
                    per_class_weights_s2[class_idx_s2] = weight
                pos_weight_s2 = per_class_weights_s2
    else:
        class_counts = Counter(labels)
        num_classes = max(labels) + 1 if len(labels) > 0 else 0
        per_class_weights = torch.zeros(num_classes)
        total_samples = len(labels)
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 0)
            if count == 0:
                weight = 1.0
            else:
                weight = total_samples / (num_classes * count)
            per_class_weights[class_idx] = weight
        pos_weight_s1 = per_class_weights
    return (pos_weight_s1, pos_weight_s2)
