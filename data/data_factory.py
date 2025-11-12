import torch
import numpy as np
import pandas as pd
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
from .preprocessing import ECGPreprocessor
from .datasets import ECGDataset
from .rpeak_utils import detect_r_peaks_pantomkins, calculate_rr_intervals, analyze_dataset_rr_intervals, optimize_patch_length
from .augmentation import add_gaussian_noise, add_salt_pepper_noise, adjust_abnormal_positions
from .weight import calculate_bce_pos_weight, calculate_inverse_frequency_weights
from .smote import apply_smote
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

def load_data_for_records(data_path, record_list_to_load, type_to_label, target_patch_len, fs, downsample_factor, rr_info, preprocessor, dynamic_feature_dim, num_consecutive_patches, configs, is_training=False):
    orig_patch_len = target_patch_len * downsample_factor
    downsampled_fs = fs / downsample_factor
    all_grouped_data = []
    all_grouped_labels = []
    all_grouped_dynamic_features = []
    mean_rr_global, std_rr_global = rr_info if rr_info else (800, 100)
    total_groups_created = 0
    if is_training and getattr(configs, 'use_augmentation', False):
        pass
    for record_name in record_list_to_load:
        record_patches = []
        record_labels = []
        record_dynamic_features_individual = []
        record_groups_created = 0
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
                            padding_needed = target_patch_len - patch_downsampled.shape[0]
                            padding = np.zeros((padding_needed, patch_downsampled.shape[1]))
                            patch = np.concatenate((patch_downsampled, padding), axis=0)
                    else:
                        patch = patch_orig
                        if patch.shape[0] != target_patch_len:
                            pass
                            continue
                    if patch.shape[0] != target_patch_len:
                        pass
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
            if num_patches_in_record < num_consecutive_patches:
                pass
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
                            label_counts = Counter(group_labels_members)
                            final_group_label = label_counts.most_common(1)[0][0]
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
                        all_grouped_data.append(combined_patch)
                        all_grouped_labels.append(final_group_label)
                        all_grouped_dynamic_features.append(combined_dynamic_features)
                        record_groups_created += 1
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
                    all_grouped_data.append(combined_patch)
                    all_grouped_labels.append(final_group_label)
                    all_grouped_dynamic_features.append(combined_dynamic_features)
                    record_groups_created += 1
            if record_groups_created > 0:
                pass
            total_groups_created += record_groups_created
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
            continue
    all_grouped_data_np = np.array(all_grouped_data) if all_grouped_data else np.empty((0, num_consecutive_patches * target_patch_len, 1))
    all_grouped_labels_np = np.array(all_grouped_labels) if all_grouped_labels else np.empty((0,))
    all_grouped_dynamic_features_np = np.array(all_grouped_dynamic_features) if all_grouped_dynamic_features else np.empty((0, num_consecutive_patches * target_patch_len, dynamic_feature_dim))
    if not all_grouped_data and record_list_to_load:
        try:
            temp_rec = wfdb.rdrecord(os.path.join(data_path, record_list_to_load[0]), sampto=10)
            n_ch = temp_rec.n_sig
            all_grouped_data_np = np.empty((0, num_consecutive_patches * target_patch_len, n_ch))
        except:
            pass
    return (all_grouped_data_np, all_grouped_labels_np, all_grouped_dynamic_features_np)

def data_provider(configs, preprocessor):
    data_path = configs.root_path
    dynamic_feature_dim = getattr(configs, 'dynamic_feature_dim', 12)
    num_consecutive_patches = getattr(configs, 'num_consecutive_patches', 3)
    device = configs.device
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
            record_path = os.path.join(data_path, record_name)
            annotation = wfdb.rdann(record_path, 'atr')
            symbols = annotation.symbol
            for sym in symbols:
                beat_types_all[sym] = beat_types_all.get(sym, 0) + 1
        except FileNotFoundError:
            pass
        except Exception as e:
            pass
            continue
    for sym, count in sorted(beat_types_all.items()):
        pass
    type_to_label = {}
    current_label_idx = 0
    sample_selection_mode = getattr(configs, 'sample_selection_mode', 'limits')
    if sample_selection_mode == 'specific_labels':
        selected_labels_list = getattr(configs, 'selected_labels', None)
        if selected_labels_list is None or not isinstance(selected_labels_list, list):
            pass
            selected_labels_list = sorted(list(beat_types_all.keys()))
        temp_type_to_label = {}
        for sym in selected_labels_list:
            if sym in beat_types_all:
                temp_type_to_label[sym] = len(temp_type_to_label)
            else:
                pass
        for i, sym in enumerate(sorted(temp_type_to_label.keys())):
            type_to_label[sym] = i
    elif sample_selection_mode == 'limits':
        sample_count_limits = getattr(configs, 'sample_count_limits', {'min_samples': 0, 'max_samples': float('inf')})
        min_samples = sample_count_limits.get('min_samples', 0)
        max_samples = sample_count_limits.get('max_samples', float('inf'))
        for sym, count in sorted(beat_types_all.items()):
            if min_samples <= count <= max_samples:
                type_to_label[sym] = current_label_idx
                current_label_idx += 1
    elif sample_selection_mode == 'by_disease':
        disease_mapping = getattr(configs, 'disease_mapping', None)
        if disease_mapping is None:
            raise ValueError("Mode is 'by_disease' but configs.disease_mapping is not defined!")
        symbols_to_train = set()
        for disease, symbols in disease_mapping.items():
            symbols_to_train.update(symbols)
        fine_grained_types_found = []
        for sym in sorted(list(symbols_to_train)):
            if sym in beat_types_all:
                fine_grained_types_found.append(sym)
            else:
                pass
        if not fine_grained_types_found:
            raise ValueError('Error: None of the symbols defined in disease_mapping were found in the dataset!')
        for idx, sym in enumerate(fine_grained_types_found):
            type_to_label[sym] = idx
        if not hasattr(configs, 'disease_categories') or not hasattr(configs, 'disease_category_names'):
            pass
    else:
        raise ValueError(f"Unknown sample_selection_mode: '{sample_selection_mode}'")
    num_classes = len(type_to_label)
    if num_classes == 0:
        raise ValueError('Error: No heartbeat types were selected! Check configuration.')
    configs.num_class = num_classes
    configs.label_to_type = {v: k for k, v in type_to_label.items()}
    if sample_selection_mode == 'by_disease':
        configs.disease_mapping = getattr(configs, 'disease_mapping')
        configs.disease_categories = getattr(configs, 'disease_categories')
        configs.disease_category_names = getattr(configs, 'disease_category_names')
    _initial_use_two_stage = configs.use_two_stage_classifier
    configs.normal_label_index = None
    configs.num_abnormal_classes = None
    configs.abnormal_fine_to_stage2_map = None
    configs.stage2_to_abnormal_fine_map = None
    if _initial_use_two_stage:
        normal_label_symbol = 'N'
        found_normal = False
        temp_normal_index = -1
        for sym, idx in type_to_label.items():
            if sym == normal_label_symbol:
                temp_normal_index = idx
                found_normal = True
                break
        if found_normal:
            configs.normal_label_index = temp_normal_index
            if configs.num_class > 1:
                configs.num_abnormal_classes = configs.num_class - 1
            else:
                configs.num_abnormal_classes = 0
                pass
            if configs.num_abnormal_classes > 0:
                abnormal_map = {}
                stage2_idx = 0
                for fine_idx in sorted(type_to_label.values()):
                    if fine_idx != configs.normal_label_index:
                        abnormal_map[fine_idx] = stage2_idx
                        stage2_idx += 1
                configs.abnormal_fine_to_stage2_map = abnormal_map
                configs.stage2_to_abnormal_fine_map = {v: k for k, v in abnormal_map.items()}
                configs.use_two_stage_classifier = True
            else:
                pass
                configs.use_two_stage_classifier = False
                configs.normal_label_index = None
                configs.num_abnormal_classes = None
                configs.abnormal_fine_to_stage2_map = None
                configs.stage2_to_abnormal_fine_map = None
        else:
            pass
            configs.use_two_stage_classifier = False
            configs.normal_label_index = None
            configs.num_abnormal_classes = None
            configs.abnormal_fine_to_stage2_map = None
            configs.stage2_to_abnormal_fine_map = None
    else:
        configs.use_two_stage_classifier = False
        configs.normal_label_index = None
        configs.num_abnormal_classes = None
        configs.abnormal_fine_to_stage2_map = None
        configs.stage2_to_abnormal_fine_map = None
    try:
        first_record_path = os.path.join(data_path, all_record_names[0])
        record = wfdb.rdrecord(first_record_path, sampto=10)
        detected_fs = record.fs
        detected_enc_in = record.n_sig
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
    configs.mean_rr = mean_rr
    configs.std_rr = std_rr
    rr_info_global = (mean_rr, std_rr)
    if getattr(configs, 'adaptive_patch_len', True):
        optimal_patch_len, optimal_stride = optimize_patch_length(mean_rr, configs.actual_fs, min_len=getattr(configs, 'adaptive_min_patch', 32), max_len=getattr(configs, 'adaptive_max_patch', 256), power_of_two=getattr(configs, 'power_of_two_patch', True))
        configs.patch_len = optimal_patch_len
        configs.stride = optimal_stride
    else:
        if not hasattr(configs, 'patch_len') or configs.patch_len is None:
            default_raw_len = getattr(configs, 'default_patch_len', 128)
            configs.patch_len = default_raw_len // configs.downsample_factor if configs.downsample_factor > 0 else default_raw_len
        if not hasattr(configs, 'stride') or configs.stride is None:
            configs.stride = configs.patch_len // 2
    train_ratio = configs.train_ratio
    val_ratio = configs.val_ratio
    test_ratio = configs.test_ratio
    random_seed = getattr(configs, 'seed', 42)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        pass
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    train_records, temp_records = train_test_split(all_record_names, test_size=val_ratio + test_ratio, random_state=random_seed)
    if np.isclose(val_ratio + test_ratio, 0):
        val_records = []
        test_records = temp_records
    elif np.isclose(test_ratio, 0):
        val_records = temp_records
        test_records = []
    else:
        val_records, test_records = train_test_split(temp_records, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)
    common_args = {'data_path': data_path, 'type_to_label': type_to_label, 'target_patch_len': configs.patch_len, 'fs': configs.fs, 'downsample_factor': configs.downsample_factor, 'rr_info': rr_info_global, 'preprocessor': preprocessor, 'dynamic_feature_dim': dynamic_feature_dim, 'num_consecutive_patches': num_consecutive_patches, 'configs': configs}
    train_data, train_labels, train_dynamic_features = load_data_for_records(**common_args, record_list_to_load=train_records, is_training=True)
    if hasattr(configs, 'use_smote') and configs.use_smote:
        smote_args = {'sampling_strategy': getattr(configs, 'smote_sampling_strategy', 'auto'), 'k_neighbors': getattr(configs, 'smote_k_neighbors', 5), 'smote_type': getattr(configs, 'smote_type', 'regular')}
        train_data, train_labels, train_dynamic_features = apply_smote(train_data, train_labels, train_dynamic_features, **smote_args)
    val_data, val_labels, val_dynamic_features = load_data_for_records(**common_args, record_list_to_load=val_records, is_training=False)
    test_data, test_labels, test_dynamic_features = load_data_for_records(**common_args, record_list_to_load=test_records, is_training=False)
    train_stage1_labels, train_stage2_labels = (None, None)
    val_stage1_labels, val_stage2_labels = (None, None)
    test_stage1_labels, test_stage2_labels = (None, None)
    if configs.use_two_stage_classifier:
        final_normal_label_index = configs.normal_label_index
        abnormal_map = configs.abnormal_fine_to_stage2_map

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
        pos_weight_stage1 = None
        if configs.stage1_loss_type == 'BCEWithLogits' and configs.stage1_pos_weight_strategy != 'none':
            stage1_counts = Counter(train_stage1_labels)
            if len(stage1_counts) == 2 and stage1_counts.get(0, 0) > 0 and (stage1_counts.get(1, 0) > 0):
                if configs.stage1_pos_weight_strategy == 'inverse_frequency':
                    pos_weight_val = stage1_counts[0] / stage1_counts[1]
                else:
                    pass
                    pos_weight_val = 1.0
                pos_weight_stage1 = torch.tensor([pos_weight_val])
            else:
                pass
                pos_weight_stage1 = torch.ones(1)
        pos_weight_stage2 = None
        if configs.stage2_loss_type in ['CrossEntropy', 'BCEWithLogits'] and configs.stage2_pos_weight_strategy == 'inverse_frequency' and (configs.num_abnormal_classes > 0):
            stage2_labels_valid = train_stage2_labels[train_stage2_labels != -100]
            if len(stage2_labels_valid) == 0:
                pass
                pos_weight_stage2 = torch.ones(configs.num_abnormal_classes)
            else:
                stage2_counts = Counter(stage2_labels_valid)
                per_class_weights_s2 = torch.zeros(configs.num_abnormal_classes)
                total_valid_s2 = len(stage2_labels_valid)
                for class_idx_s2 in range(configs.num_abnormal_classes):
                    count = stage2_counts.get(class_idx_s2, 0)
                    if count == 0:
                        weight = 1.0
                        pass
                    else:
                        weight = total_valid_s2 / (configs.num_abnormal_classes * count)
                    per_class_weights_s2[class_idx_s2] = weight
                pos_weight_stage2 = per_class_weights_s2
    else:
        pos_weight = None
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
    if configs.use_two_stage_classifier:
        final_pos_weights = (pos_weight_stage1, pos_weight_stage2)
    else:
        final_pos_weights = (None, None)
    return ((train_loader, val_loader, test_loader), dims, final_pos_weights)

def get_data_loaders(configs):
    wavelet_config_dict = {'wavelet': getattr(configs, 'preprocess_wavelet_name', 'bior3.7'), 'level': getattr(configs, 'preprocess_wavelet_level', 3), 'threshold_scale': getattr(configs, 'preprocess_wavelet_threshold_scale', 0.7)}
    highpass_config_dict = {'cutoff': getattr(configs, 'preprocess_highpass_cutoff', 0.5), 'order': getattr(configs, 'preprocess_highpass_order', 2)}
    notch_config_dict = {'freq': getattr(configs, 'preprocess_notch_freq', 60.0), 'q': getattr(configs, 'preprocess_notch_q', 30.0)}
    preprocessor = ECGPreprocessor(use_wavelet=getattr(configs, 'preprocess_use_wavelet', True), wavelet_config=wavelet_config_dict, use_highpass=getattr(configs, 'preprocess_use_highpass', False), highpass_config=highpass_config_dict, use_notch=getattr(configs, 'preprocess_use_notch', True), notch_config=notch_config_dict, use_median=getattr(configs, 'preprocess_use_median', True), median_kernel_ms=getattr(configs, 'preprocess_median_kernel_ms', 50), normalization_method=getattr(configs, 'normalization_method', 'minmax'))
    data_loaders, dims, final_pos_weights = data_provider(configs, preprocessor=preprocessor)
    return (data_loaders, dims, final_pos_weights)
