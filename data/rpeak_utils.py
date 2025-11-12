import numpy as np
import wfdb
from wfdb import processing
import os
from scipy import signal as sp_signal
import math
import logging
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
