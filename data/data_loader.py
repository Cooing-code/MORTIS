import torch
import numpy as np
import os
import wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from wfdb import processing
import copy
from scipy import signal as sp_signal
import math
import logging
from collections import Counter
import random
import gc
from .preprocessing import ECGPreprocessor
from .datasets import ECGDataset
from .rpeak_utils import detect_r_peaks_pantomkins, analyze_dataset_rr_intervals, optimize_patch_length
from .augmentation import add_gaussian_noise, add_salt_pepper_noise, adjust_abnormal_positions
logger = logging.getLogger(__name__)

def detect_r_peaks_pantomkins(signal, fs):
    lowcut = 5.0
    highcut = 15.0
    order = 1
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    try:
        b, a = sp_signal.butter(order, [low, high], btype='band')
        filtered_signal = sp_signal.filtfilt(b, a, signal)
    except ValueError as e:
        pass
        return np.array([])
    diff_signal = np.diff(filtered_signal)
    squared_signal = diff_signal ** 2
    window_size = int(0.15 * fs)
    if window_size < 1:
        window_size = 1
    integrated_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')
    min_peak_height_factor = 0.3
    min_peak_height = min_peak_height_factor * np.max(integrated_signal) if np.max(integrated_signal) > 0 else 0.1
    min_peak_distance = int(0.2 * fs)
    if min_peak_distance < 1:
        min_peak_distance = 1
    try:
        if integrated_signal is None or len(integrated_signal) < min_peak_distance:
            peaks = np.array([])
        else:
            peaks, _ = sp_signal.find_peaks(integrated_signal, height=min_peak_height, distance=min_peak_distance)
    except Exception as e_findpeaks:
        pass
        peaks = np.array([])
    r_peaks = []
    search_radius = int(0.1 * fs)
    if search_radius < 1:
        search_radius = 1
    for peak_idx in peaks:
        start = max(0, peak_idx - search_radius)
        end = min(len(signal), peak_idx + search_radius)
        if start >= end:
            continue
        try:
            local_max_idx = start + np.argmax(filtered_signal[start:end])
            r_peaks.append(local_max_idx)
        except ValueError:
            continue
    return np.array(r_peaks)

def calculate_rr_intervals(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        signal_channel = record.p_signal[:, 0]
        r_indices = detect_r_peaks_pantomkins(signal_channel, fs)
        if len(r_indices) < 5 and record.p_signal.shape[1] > 1:
            signal_channel = record.p_signal[:, 1]
            r_indices = detect_r_peaks_pantomkins(signal_channel, fs)
        if len(r_indices) < 2:
            return (800, 100, [])
    except Exception as e:
        pass
        return (800, 100, [])
    rr_intervals = np.diff(r_indices)
    rr_intervals_ms = rr_intervals / fs * 1000
    q1, q3 = np.percentile(rr_intervals_ms, [25, 75])
    iqr = q3 - q1
    lower_bound = max(200, q1 - 1.5 * iqr)
    upper_bound = min(2000, q3 + 1.5 * iqr)
    valid_rr = rr_intervals_ms[(rr_intervals_ms >= lower_bound) & (rr_intervals_ms <= upper_bound)]
    if len(valid_rr) == 0:
        return (800, 100, [])
    mean_rr = np.mean(valid_rr)
    std_rr = np.std(valid_rr)
    return (mean_rr, std_rr, valid_rr)

def optimize_patch_length(mean_rr, fs, min_len=32, max_len=256, power_of_two=True):
    if fs is None or fs <= 0:
        pass
        return (min_len, min_len // 2)
    rr_samples = int(mean_rr / 1000.0 * fs)
    target_patch_len = rr_samples * 0.8
    if target_patch_len == 0:
        target_patch_len = min_len
    if power_of_two:
        if target_patch_len > 0:
            power = math.ceil(math.log2(target_patch_len))
            patch_len_upper = 2 ** power
            patch_len_lower = 2 ** (power - 1) if power > 0 else min_len
            patch_len = patch_len_lower if abs(target_patch_len - patch_len_lower) <= abs(target_patch_len - patch_len_upper) else patch_len_upper
            patch_len = max(min_len, patch_len)
        else:
            patch_len = min_len
    else:
        patch_len = target_patch_len
    patch_len = max(min_len, min(max_len, patch_len))
    if patch_len % 2 != 0:
        patch_len += 1
    stride = patch_len
    return (patch_len, stride)

def analyze_dataset_rr_intervals(data_path, record_list_to_analyze):
    all_mean_rr = []
    all_std_rr = []
    all_rr_intervals = []
    for record_name in record_list_to_analyze:
        try:
            record_path = os.path.join(data_path, record_name)
            mean_rr, std_rr, rr_intervals = calculate_rr_intervals(record_path)
            if mean_rr is not None:
                all_mean_rr.append(mean_rr)
                all_std_rr.append(std_rr)
                all_rr_intervals.extend(rr_intervals)
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
            continue
    if len(all_mean_rr) > 0:
        dataset_mean_rr = np.mean(all_mean_rr)
        dataset_std_rr = np.mean(all_std_rr)
    else:
        dataset_mean_rr = 800
        dataset_std_rr = 100
        pass
    return (dataset_mean_rr, dataset_std_rr, all_rr_intervals)

def generate_grouped_sequences_for_records(data_path, record_list_to_load, type_to_label, target_patch_len, fs, downsample_factor, rr_info, preprocessor, dynamic_feature_dim, num_consecutive_patches, configs, is_training=False):
    orig_patch_len = target_patch_len * downsample_factor
    downsampled_fs = fs / downsample_factor
    mean_rr_global, std_rr_global = rr_info if rr_info else (800, 100)
    total_groups_yielded = 0
    if is_training and getattr(configs, 'use_augmentation', False):
        pass
    for record_name in record_list_to_load:
        record_patches = []
        record_labels = []
        record_dynamic_features_individual = []
        record_groups_yielded_from_record = 0
        try:
            record_path = os.path.join(data_path, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            raw_signals = record.p_signal
            actual_fs = record.fs
            if preprocessor:
                processed_signals = preprocessor.preprocess_signal(raw_signals, actual_fs)
            else:
                processed_signals = raw_signals
                pass
            if abs(actual_fs - fs) > 1:
                pass
            symbols = annotation.symbol
            locations = annotation.sample
            signal_for_rpeak = processed_signals[:, 0]
            r_peaks_detected = detect_r_peaks_pantomkins(signal_for_rpeak, actual_fs)
            if len(r_peaks_detected) < num_consecutive_patches:
                continue
            rr_intervals_samples = np.diff(r_peaks_detected)
            rr_intervals_ms = rr_intervals_samples / actual_fs * 1000
            rr_intervals_ms = np.insert(rr_intervals_ms, 0, rr_intervals_ms[0] if len(rr_intervals_ms) > 0 else mean_rr_global)
            for r_peak_idx, r_peak_loc in enumerate(r_peaks_detected):
                annotated_locs = np.array(locations)
                distances = np.abs(annotated_locs - r_peak_loc)
                closest_annotated_idx = np.argmin(distances)
                tolerance = int(0.1 * actual_fs)
                if distances[closest_annotated_idx] <= tolerance:
                    sym = symbols[closest_annotated_idx]
                else:
                    sym = '?'
                    continue
                if sym in type_to_label:
                    samples_before_r = orig_patch_len // 3
                    samples_after_r = orig_patch_len - samples_before_r
                    start = r_peak_loc - samples_before_r
                    end = r_peak_loc + samples_after_r
                    patch_orig = np.zeros((orig_patch_len, processed_signals.shape[1]))
                    valid_start = max(0, start)
                    valid_end = min(len(processed_signals), end)
                    len_valid = valid_end - valid_start
                    if len_valid > 0:
                        start_in_patch = max(0, -start)
                        end_in_patch = start_in_patch + len_valid
                        if start_in_patch < orig_patch_len and end_in_patch <= orig_patch_len:
                            patch_orig[start_in_patch:end_in_patch, :] = processed_signals[valid_start:valid_end, :]
                    if downsample_factor > 1:
                        target_len = target_patch_len
                        if patch_orig.shape[0] > 0:
                            patch_downsampled = np.array([sp_signal.resample(patch_orig[:, i], target_len) for i in range(patch_orig.shape[1])]).T
                        else:
                            patch_downsampled = np.zeros((target_len, patch_orig.shape[1]))
                        if patch_downsampled.shape[0] == target_patch_len:
                            patch = patch_downsampled
                        elif patch_downsampled.shape[0] > target_patch_len:
                            patch = patch_downsampled[:target_patch_len, :]
                        else:
                            patch = np.concatenate((patch_downsampled, np.zeros((target_patch_len - patch_downsampled.shape[0], patch_downsampled.shape[1]))), axis=0)
                    else:
                        patch = patch_orig
                        if patch.shape[0] != target_patch_len:
                            continue
                    if patch.shape[0] != target_patch_len:
                        continue
                    base_feature_dim = dynamic_feature_dim - 1
                    if base_feature_dim <= 0:
                        base_feature_dim = 1
                    base_dynamic_features = np.zeros((target_patch_len, base_feature_dim))
                    time_interval_ds = 1000.0 / downsampled_fs if downsampled_fs > 0 else 0
                    time_series_ms = np.arange(target_patch_len) * time_interval_ds
                    current_rr = rr_intervals_ms[r_peak_idx] if r_peak_idx < len(rr_intervals_ms) else mean_rr_global
                    prev_rr = rr_intervals_ms[r_peak_idx - 1] if r_peak_idx > 0 and r_peak_idx - 1 < len(rr_intervals_ms) else mean_rr_global
                    current_rr = max(current_rr, 1)
                    prev_rr = max(prev_rr, 1)
                    mean_rr_global_safe = max(mean_rr_global, 1)
                    n_local_rr = 5
                    local_rr_start = max(0, r_peak_idx - n_local_rr)
                    local_rrs = rr_intervals_ms[local_rr_start:min(r_peak_idx + 1, len(rr_intervals_ms))]
                    local_hrv = np.std(local_rrs) if len(local_rrs) > 1 else 0
                    r_peak_relative_loc_ds = target_patch_len // 2
                    num_base_features_to_fill = min(base_feature_dim, 11)
                    for p in range(target_patch_len):
                        feat_idx = 0
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = time_series_ms[p] % 1000 / 1000.0
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = p / target_patch_len
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            dynamic_phase = time_series_ms[p] % current_rr / current_rr if current_rr > 0 else 0
                            base_dynamic_features[p, feat_idx] = np.sin(2 * np.pi * dynamic_phase)
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = np.cos(2 * np.pi * dynamic_phase)
                            feat_idx += 1
                        else:
                            continue
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = current_rr / mean_rr_global_safe
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = (current_rr - prev_rr) / mean_rr_global_safe
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = local_hrv / mean_rr_global_safe
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            dist_to_r_samples = abs(p - r_peak_relative_loc_ds)
                            norm_dist = dist_to_r_samples / max(1, target_patch_len / 2.0)
                            base_dynamic_features[p, feat_idx] = norm_dist
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = 1.0 if p == r_peak_relative_loc_ds else 0.0
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            rr_samples_in_patch = max(1, current_rr / time_interval_ds) if time_interval_ds > 0 else 1
                            relative_pos_in_beat = (p - r_peak_relative_loc_ds) / rr_samples_in_patch
                            base_dynamic_features[p, feat_idx] = np.sin(np.pi * relative_pos_in_beat)
                            feat_idx += 1
                        if feat_idx < num_base_features_to_fill:
                            base_dynamic_features[p, feat_idx] = np.cos(np.pi * relative_pos_in_beat)
                            feat_idx += 1
                        else:
                            continue
                    label = type_to_label[sym]
                    record_patches.append(patch)
                    record_labels.append(label)
                    record_dynamic_features_individual.append(base_dynamic_features)
            augmented_record_patches = []
            augmented_record_labels = []
            augmented_record_dynamic_features = []
            noise_augmented_count = 0
            position_adjusted_count = 0
            original_count = len(record_patches)
            generated_adjusted_sequences = []
            use_augmentation = getattr(configs, 'use_augmentation', False)
            if is_training and use_augmentation:
                augment_prob = getattr(configs, 'augment_prob', 0.0)
                noise_types = getattr(configs, 'augment_noise_types', [])
                noise_std = getattr(configs, 'augment_noise_std', 0.01)
                noise_amount = getattr(configs, 'augment_noise_amount', 0.005)
                adjust_position_prob = getattr(configs, 'augment_adjust_position_prob', 0.0)
                augment_minority_only = getattr(configs, 'augment_minority_only', True)
                n_label = type_to_label.get('N', -1)
                for i in range(len(record_patches)):
                    current_label = record_labels[i]
                    is_minority = n_label != -1 and current_label != n_label
                    apply_augment = False
                    if augment_minority_only:
                        if is_minority and random.random() < augment_prob:
                            apply_augment = True
                    elif random.random() < augment_prob:
                        apply_augment = True
                    if apply_augment and noise_types:
                        original_patch = record_patches[i]
                        augmented_patch = np.copy(original_patch)
                        chosen_noise = random.choice(noise_types)
                        if chosen_noise == 'gaussian':
                            augmented_patch = add_gaussian_noise(augmented_patch, std=noise_std)
                            noise_augmented_count += 1
                        elif chosen_noise == 'salt_pepper':
                            augmented_patch = add_salt_pepper_noise(augmented_patch, amount=noise_amount)
                            noise_augmented_count += 1
                        augmented_record_patches.append(augmented_patch)
                        augmented_record_labels.append(current_label)
                        if record_dynamic_features_individual and i < len(record_dynamic_features_individual):
                            augmented_record_dynamic_features.append(record_dynamic_features_individual[i])
                if len(record_patches) >= num_consecutive_patches and random.random() < adjust_position_prob:
                    generated_adjusted_sequences = adjust_abnormal_positions(record_patches, record_labels, record_dynamic_features_individual, type_to_label)
                    position_adjusted_count = len(generated_adjusted_sequences)
                record_patches.extend(augmented_record_patches)
                record_labels.extend(augmented_record_labels)
                if record_dynamic_features_individual is not None and augmented_record_dynamic_features:
                    record_dynamic_features_individual.extend(augmented_record_dynamic_features)
            num_patches_in_record = len(record_patches)
            if num_patches_in_record < num_consecutive_patches and (not generated_adjusted_sequences):
                continue
            for adjusted_seq_tuple in generated_adjusted_sequences:
                adj_patches, adj_labels, adj_dyn_feats = adjusted_seq_tuple
                num_adj_patches = len(adj_patches)
                if num_adj_patches >= num_consecutive_patches:
                    for i in range(num_adj_patches - num_consecutive_patches + 1):
                        group_patches_members = adj_patches[i:i + num_consecutive_patches]
                        group_labels_members = adj_labels[i:i + num_consecutive_patches]
                        group_dynamic_features_base = adj_dyn_feats[i:i + num_consecutive_patches] if adj_dyn_feats else [None] * num_consecutive_patches
                        try:
                            n_label = type_to_label.get('N', -1)
                            target_labels_in_group = [lbl for lbl in group_labels_members if lbl != n_label]
                            if not target_labels_in_group:
                                final_group_label = n_label if n_label != -1 else group_labels_members[0]
                            else:
                                target_label_counts = Counter(target_labels_in_group)
                                max_freq = max(target_label_counts.values())
                                most_frequent_target_labels = [lbl for lbl, freq in target_label_counts.items() if freq == max_freq]
                                if len(most_frequent_target_labels) == 1:
                                    final_group_label = most_frequent_target_labels[0]
                                else:
                                    first_occurrence_indices = {lbl: group_labels_members.index(lbl) for lbl in most_frequent_target_labels}
                                    final_group_label = min(first_occurrence_indices, key=first_occurrence_indices.get)
                        except Exception as e_labeling_adj:
                            pass
                            continue
                        valid_base_features = all((feat is not None for feat in group_dynamic_features_base))
                        group_dynamic_features_final_list = []
                        valid_dynamic_features = True
                        if valid_base_features:
                            for j in range(num_consecutive_patches):
                                time_feat_1d = np.full((target_patch_len, 1), fill_value=j / num_consecutive_patches)
                                if group_dynamic_features_base[j].shape[1] != dynamic_feature_dim - 1:
                                    pass
                                    valid_dynamic_features = False
                                    break
                                patch_dynamic_features_final = np.concatenate((group_dynamic_features_base[j], time_feat_1d), axis=1)
                                if patch_dynamic_features_final.shape[1] != dynamic_feature_dim:
                                    pass
                                    valid_dynamic_features = False
                                    break
                                group_dynamic_features_final_list.append(patch_dynamic_features_final)
                        else:
                            dummy_feat = np.zeros((target_patch_len, dynamic_feature_dim))
                            group_dynamic_features_final_list = [dummy_feat] * num_consecutive_patches
                            valid_dynamic_features = True
                        if not valid_dynamic_features:
                            continue
                        combined_patch = np.concatenate(group_patches_members, axis=0)
                        combined_dynamic_features = np.concatenate(group_dynamic_features_final_list, axis=0)
                        expected_seq_len = num_consecutive_patches * target_patch_len
                        if combined_patch.shape[0] != expected_seq_len or combined_dynamic_features.shape[0] != expected_seq_len:
                            pass
                            continue
                        yield (combined_patch, final_group_label, combined_dynamic_features)
                        record_groups_yielded_from_record += 1
            if num_patches_in_record >= num_consecutive_patches:
                current_record_patches = record_patches
                current_record_labels = record_labels
                current_record_dyn_feats = record_dynamic_features_individual
                for i in range(num_patches_in_record - num_consecutive_patches + 1):
                    group_patches_members = current_record_patches[i:i + num_consecutive_patches]
                    group_labels_members = current_record_labels[i:i + num_consecutive_patches]
                    group_dynamic_features_base = current_record_dyn_feats[i:i + num_consecutive_patches] if current_record_dyn_feats and len(current_record_dyn_feats) > i + num_consecutive_patches - 1 else [None] * num_consecutive_patches
                    try:
                        n_label = type_to_label.get('N', -1)
                        target_labels_in_group = [lbl for lbl in group_labels_members if lbl != n_label]
                        if not target_labels_in_group:
                            final_group_label = n_label if n_label != -1 else group_labels_members[0]
                        else:
                            target_label_counts = Counter(target_labels_in_group)
                            max_freq = max(target_label_counts.values())
                            most_frequent_target_labels = [lbl for lbl, freq in target_label_counts.items() if freq == max_freq]
                            if len(most_frequent_target_labels) == 1:
                                final_group_label = most_frequent_target_labels[0]
                            else:
                                first_occurrence_indices = {lbl: group_labels_members.index(lbl) for lbl in most_frequent_target_labels}
                                final_group_label = min(first_occurrence_indices, key=first_occurrence_indices.get)
                    except Exception as e_labeling:
                        pass
                        continue
                    valid_base_features = all((feat is not None for feat in group_dynamic_features_base))
                    group_dynamic_features_final_list = []
                    valid_dynamic_features = True
                    if valid_base_features:
                        for j in range(num_consecutive_patches):
                            time_feat_1d = np.full((target_patch_len, 1), fill_value=j / num_consecutive_patches)
                            if group_dynamic_features_base[j].shape[1] != dynamic_feature_dim - 1:
                                pass
                                valid_dynamic_features = False
                                break
                            patch_dynamic_features_final = np.concatenate((group_dynamic_features_base[j], time_feat_1d), axis=1)
                            if patch_dynamic_features_final.shape[1] != dynamic_feature_dim:
                                pass
                                valid_dynamic_features = False
                                break
                            group_dynamic_features_final_list.append(patch_dynamic_features_final)
                    else:
                        dummy_feat = np.zeros((target_patch_len, dynamic_feature_dim))
                        group_dynamic_features_final_list = [dummy_feat] * num_consecutive_patches
                        valid_dynamic_features = True
                    if not valid_dynamic_features:
                        continue
                    combined_patch = np.concatenate(group_patches_members, axis=0)
                    combined_dynamic_features = np.concatenate(group_dynamic_features_final_list, axis=0)
                    expected_seq_len = num_consecutive_patches * target_patch_len
                    if combined_patch.shape[0] != expected_seq_len or combined_dynamic_features.shape[0] != expected_seq_len:
                        pass
                        continue
                    yield (combined_patch, final_group_label, combined_dynamic_features)
                    record_groups_yielded_from_record += 1
            if record_groups_yielded_from_record > 0:
                pass
            total_groups_yielded += record_groups_yielded_from_record
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
            continue

def data_provider(configs, preprocessor):
    data_path = configs.root_path
    dynamic_feature_dim = getattr(configs, 'dynamic_feature_dim', 12)
    num_consecutive_patches = getattr(configs, 'num_consecutive_patches', 3)
    try:
        all_record_names = sorted([f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.hea')])
        if not all_record_names:
            raise FileNotFoundError(f"No .hea files found in '{data_path}'.")
    except Exception as e:
        pass
        raise
    beat_types_all = {}
    for record_name in all_record_names:
        try:
            annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
            for sym in annotation.symbol:
                beat_types_all[sym] = beat_types_all.get(sym, 0) + 1
        except Exception as e:
            pass
            continue
    type_to_label = {}
    current_label_idx = 0
    sample_selection_mode = getattr(configs, 'sample_selection_mode', 'limits')
    if sample_selection_mode == 'specific_labels':
        selected_labels_list = getattr(configs, 'selected_labels', [])
        final_label_order = [s for s in selected_labels_list if s in beat_types_all]
        type_to_label = {sym: i for i, sym in enumerate(final_label_order)}
    elif sample_selection_mode == 'limits':
        sample_count_limits = getattr(configs, 'sample_count_limits', {'min_samples': 0, 'max_samples': float('inf')})
        min_samples, max_samples = (sample_count_limits.get('min_samples', 0), sample_count_limits.get('max_samples', float('inf')))
        for sym, count in sorted(beat_types_all.items()):
            if min_samples <= count <= max_samples:
                type_to_label[sym] = current_label_idx
                current_label_idx += 1
    elif sample_selection_mode == 'by_disease':
        disease_mapping = getattr(configs, 'disease_mapping', None)
        assert disease_mapping is not None
        symbols_to_train = set().union(*disease_mapping.values())
        potential_order = []
        [potential_order.extend(disease_mapping[d]) for d in sorted(disease_mapping.keys())]
        fine_grained_types_found = [sym for sym in sorted(list(set(potential_order))) if sym in beat_types_all]
        assert fine_grained_types_found, 'Error: None of the valid symbols defined in disease_mapping were found in the dataset!'
        type_to_label = {sym: idx for idx, sym in enumerate(fine_grained_types_found)}
    else:
        type_to_label = {sym: i for i, sym in enumerate(sorted(beat_types_all.keys()))}
    configs.num_class = len(type_to_label)
    assert configs.num_class > 0
    configs.label_to_type = {v: k for k, v in type_to_label.items()}
    _initial_use_two_stage = configs.use_two_stage_classifier
    configs.normal_label_index, configs.num_abnormal_classes = (None, None)
    configs.abnormal_fine_to_stage2_map, configs.stage2_to_abnormal_fine_map = (None, None)
    if _initial_use_two_stage:
        normal_label_symbol = 'N'
        temp_normal_index = type_to_label.get(normal_label_symbol, -1)
        if temp_normal_index != -1:
            configs.normal_label_index = temp_normal_index
            configs.num_abnormal_classes = max(0, configs.num_class - 1)
            if configs.num_abnormal_classes > 0:
                abnormal_map = {fine_idx: stage2_idx for stage2_idx, fine_idx in enumerate(sorted([idx for idx in type_to_label.values() if idx != configs.normal_label_index]))}
                configs.abnormal_fine_to_stage2_map = abnormal_map
                configs.stage2_to_abnormal_fine_map = {v: k for k, v in abnormal_map.items()}
            else:
                pass
                configs.use_two_stage_classifier = False
        else:
            pass
            configs.use_two_stage_classifier = False
    else:
        configs.use_two_stage_classifier = False
    try:
        record = wfdb.rdrecord(os.path.join(data_path, all_record_names[0]), sampto=10)
        detected_fs, detected_enc_in = (record.fs, record.n_sig)
        if configs.fs is None:
            configs.fs = detected_fs
        elif configs.fs != detected_fs:
            pass
        if configs.enc_in is None:
            configs.enc_in = detected_enc_in
        elif configs.enc_in != detected_enc_in:
            pass
    except Exception as e:
        pass
        if configs.fs is None:
            configs.fs = 360
            pass
        if configs.enc_in is None:
            configs.enc_in = 2
            pass
    if configs.downsample_factor <= 0:
        configs.downsample_factor = 1
    configs.actual_fs = configs.fs / configs.downsample_factor
    mean_rr, std_rr, _ = analyze_dataset_rr_intervals(data_path, all_record_names)
    configs.mean_rr, configs.std_rr = (mean_rr, std_rr)
    rr_info_global = (mean_rr, std_rr)
    if getattr(configs, 'adaptive_patch_len', True):
        optimal_patch_len, optimal_stride = optimize_patch_length(mean_rr, configs.actual_fs, min_len=getattr(configs, 'adaptive_min_patch', 32), max_len=getattr(configs, 'adaptive_max_patch', 256), power_of_two=getattr(configs, 'power_of_two_patch', True))
        configs.patch_len, configs.stride = (optimal_patch_len, optimal_stride)
    else:
        if not hasattr(configs, 'patch_len') or configs.patch_len is None:
            default_raw_len = getattr(configs, 'default_patch_len', 128)
            configs.patch_len = default_raw_len // configs.downsample_factor if configs.downsample_factor > 0 else default_raw_len
        if not hasattr(configs, 'stride') or configs.stride is None:
            configs.stride = configs.patch_len // 2
    train_ratio, val_ratio, test_ratio = (configs.train_ratio, configs.val_ratio, configs.test_ratio)
    random_seed = getattr(configs, 'seed', 42)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    train_records, temp_records = train_test_split(all_record_names, test_size=val_ratio + test_ratio, random_state=random_seed)
    if np.isclose(val_ratio + test_ratio, 0):
        val_records, test_records = ([], temp_records)
    elif np.isclose(test_ratio, 0):
        val_records, test_records = (temp_records, [])
    else:
        val_records, test_records = train_test_split(temp_records, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)
    common_args = {'data_path': data_path, 'type_to_label': type_to_label, 'target_patch_len': configs.patch_len, 'fs': configs.fs, 'downsample_factor': configs.downsample_factor, 'rr_info': rr_info_global, 'preprocessor': preprocessor, 'dynamic_feature_dim': dynamic_feature_dim, 'num_consecutive_patches': num_consecutive_patches, 'configs': configs}
    train_data_list, train_labels_list, train_dynamic_features_list = ([], [], [])
    train_generator = generate_grouped_sequences_for_records(**common_args, record_list_to_load=train_records, is_training=True)
    for data, label, dyn_feat in train_generator:
        train_data_list.append(data)
        train_labels_list.append(label)
        train_dynamic_features_list.append(dyn_feat)
    train_data = np.array(train_data_list) if train_data_list else np.empty((0, num_consecutive_patches * configs.patch_len, configs.enc_in))
    train_labels = np.array(train_labels_list) if train_labels_list else np.empty((0,))
    train_dynamic_features = np.array(train_dynamic_features_list) if train_dynamic_features_list else np.empty((0, num_consecutive_patches * configs.patch_len, dynamic_feature_dim))
    del train_data_list, train_labels_list, train_dynamic_features_list
    gc.collect()
    val_data_list, val_labels_list, val_dynamic_features_list = ([], [], [])
    val_generator = generate_grouped_sequences_for_records(**common_args, record_list_to_load=val_records, is_training=False)
    for data, label, dyn_feat in val_generator:
        val_data_list.append(data)
        val_labels_list.append(label)
        val_dynamic_features_list.append(dyn_feat)
    val_data = np.array(val_data_list) if val_data_list else np.empty((0, num_consecutive_patches * configs.patch_len, configs.enc_in))
    val_labels = np.array(val_labels_list) if val_labels_list else np.empty((0,))
    val_dynamic_features = np.array(val_dynamic_features_list) if val_dynamic_features_list else np.empty((0, num_consecutive_patches * configs.patch_len, dynamic_feature_dim))
    del val_data_list, val_labels_list, val_dynamic_features_list
    gc.collect()
    test_data_list, test_labels_list, test_dynamic_features_list = ([], [], [])
    test_generator = generate_grouped_sequences_for_records(**common_args, record_list_to_load=test_records, is_training=False)
    for data, label, dyn_feat in test_generator:
        test_data_list.append(data)
        test_labels_list.append(label)
        test_dynamic_features_list.append(dyn_feat)
    test_data = np.array(test_data_list) if test_data_list else np.empty((0, num_consecutive_patches * configs.patch_len, configs.enc_in))
    test_labels = np.array(test_labels_list) if test_labels_list else np.empty((0,))
    test_dynamic_features = np.array(test_dynamic_features_list) if test_dynamic_features_list else np.empty((0, num_consecutive_patches * configs.patch_len, dynamic_feature_dim))
    del test_data_list, test_labels_list, test_dynamic_features_list
    gc.collect()
    train_stage1_labels, train_stage2_labels = (None, None)
    val_stage1_labels, val_stage2_labels = (None, None)
    test_stage1_labels, test_stage2_labels = (None, None)
    pos_weight_s1 = None
    pos_weight_s2 = None
    if configs.use_two_stage_classifier:
        final_normal_label_index = configs.normal_label_index
        abnormal_map = configs.abnormal_fine_to_stage2_map
        if final_normal_label_index is not None and abnormal_map is not None and (configs.num_abnormal_classes > 0):

            def generate_stage_labels(original_labels):
                stage1 = np.where(original_labels == final_normal_label_index, 0, 1)
                stage2 = np.full_like(original_labels, -100)
                abnormal_mask = stage1 == 1
                original_abnormal_labels = original_labels[abnormal_mask]
                stage2[abnormal_mask] = [abnormal_map.get(l, -100) for l in original_abnormal_labels]
                return (stage1, stage2)
            train_stage1_labels, train_stage2_labels = generate_stage_labels(train_labels)
            val_stage1_labels, val_stage2_labels = generate_stage_labels(val_labels)
            test_stage1_labels, test_stage2_labels = generate_stage_labels(test_labels)
            if configs.stage1_pos_weight_strategy == 'inverse_frequency':
                s1_counts = Counter(train_stage1_labels)
                if len(s1_counts) == 2 and s1_counts.get(0, 0) > 0 and (s1_counts.get(1, 0) > 0):
                    if configs.stage1_loss_type.lower() == 'bcewithlogits':
                        pos_weight_val = s1_counts[0] / s1_counts[1]
                        pos_weight_s1 = torch.tensor([pos_weight_val])
                    else:
                        total_s1 = len(train_stage1_labels)
                        weight_0 = total_s1 / (2 * s1_counts[0])
                        weight_1 = total_s1 / (2 * s1_counts[1])
                        pos_weight_s1 = torch.tensor([weight_0, weight_1])
                else:
                    pass
                    pos_weight_s1 = torch.ones(2 if configs.stage1_loss_type.lower() != 'bcewithlogits' else 1)
            if configs.stage2_pos_weight_strategy == 'inverse_frequency' and configs.num_abnormal_classes > 0:
                valid_stage2_labels = train_stage2_labels[train_stage2_labels != -100]
                if len(valid_stage2_labels) > 0:
                    s2_counts = Counter(valid_stage2_labels)
                    per_class_weights_s2 = torch.zeros(configs.num_abnormal_classes)
                    total_s2 = len(valid_stage2_labels)
                    all_classes_present = True
                    for class_idx in range(configs.num_abnormal_classes):
                        count = s2_counts.get(class_idx, 0)
                        if count == 0:
                            weight = 1.0
                            pass
                            all_classes_present = False
                        else:
                            weight = total_s2 / (configs.num_abnormal_classes * count)
                        per_class_weights_s2[class_idx] = weight
                    if not all_classes_present and configs.stage2_loss_type.lower() == 'bcewithlogits':
                        pass
                    pos_weight_s2 = per_class_weights_s2
                else:
                    pass
                    pos_weight_s2 = torch.ones(configs.num_abnormal_classes)
        else:
            pass
    elif configs.stage1_pos_weight_strategy == 'inverse_frequency':
        if len(train_labels) == 0:
            pass
            pos_weight_s1 = torch.ones(configs.num_class)
        else:
            class_counts = Counter(train_labels)
            per_class_weights = torch.zeros(configs.num_class)
            total_samples = len(train_labels)
            all_classes_present = True
            for class_idx in range(configs.num_class):
                count = class_counts.get(class_idx, 0)
                if count == 0:
                    weight = 1.0
                    pass
                    all_classes_present = False
                else:
                    weight = total_samples / (configs.num_class * count)
                per_class_weights[class_idx] = weight
            if not all_classes_present and configs.stage1_loss_type.lower() == 'bcewithlogits':
                pass
            pos_weight_s1 = per_class_weights
    train_dataset = ECGDataset(train_data, train_labels, train_dynamic_features, dynamic_feature_dim, stage1_labels=train_stage1_labels, stage2_labels=train_stage2_labels)
    val_dataset = ECGDataset(val_data, val_labels, val_dynamic_features, dynamic_feature_dim, stage1_labels=val_stage1_labels, stage2_labels=val_stage2_labels)
    test_dataset = ECGDataset(test_data, test_labels, test_dynamic_features, dynamic_feature_dim, stage1_labels=test_stage1_labels, stage2_labels=test_stage2_labels)
    batch_size = getattr(configs, 'batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=getattr(configs, 'num_workers', 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=getattr(configs, 'num_workers', 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=getattr(configs, 'num_workers', 0))
    if len(train_data) > 0:
        dims = (train_data.shape[1], train_data.shape[2])
    else:
        pass
        dims = (0, 0)
    pos_weights_tuple = (pos_weight_s1, pos_weight_s2)
    return ((train_loader, val_loader, test_loader), dims, pos_weights_tuple)
