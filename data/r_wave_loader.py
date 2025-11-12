import os
import numpy as np
import wfdb
from torch.utils.data import Dataset, DataLoader
from wfdb import processing
import torch
import copy
from sklearn.model_selection import train_test_split

class RWaveECGDataset(Dataset):

    def __init__(self, data, labels, timestamps=None):
        self.data = data
        self.labels = labels
        self.timestamps = timestamps

    def __getitem__(self, index):
        if self.timestamps is not None:
            return (torch.FloatTensor(self.data[index]), torch.LongTensor([self.labels[index]])[0], torch.FloatTensor(self.timestamps[index]))
        else:
            return (torch.FloatTensor(self.data[index]), torch.LongTensor([self.labels[index]])[0])

    def __len__(self):
        return len(self.data)

def detect_r_peaks_with_wfdb(record_path, record_name):
    full_path = os.path.join(record_path, record_name)
    record = wfdb.rdrecord(full_path)
    annotation = wfdb.rdann(full_path, 'atr')
    signal = record.p_signal
    fs = record.fs
    original_r_peaks = annotation.sample
    original_symbols = annotation.symbol
    ECG_R_list = {'N', 'f', 'e', '/', 'j', 'n', 'B', 'L', 'R', 'S', 'A', 'J', 'a', 'V', 'E', 'r', 'F', 'Q', '?'}
    filtered_peaks = []
    filtered_symbols = []
    if len(original_r_peaks) > 0:
        for peak, sym in zip(original_r_peaks, original_symbols):
            if sym in ECG_R_list:
                filtered_peaks.append(peak)
                filtered_symbols.append(sym)
    if len(filtered_peaks) < 10:
        r_peaks = np.array(filtered_peaks)
        symbols = filtered_symbols
    else:
        r_peaks = np.array(filtered_peaks)
        symbols = filtered_symbols
    r_peaks_with_labels = list(zip(r_peaks, symbols))
    return (r_peaks_with_labels, signal, fs)

def r_wave_centered_patches(signal, r_peaks_with_labels, fs, window_ms=250, downsample_factor=1):
    patches = []
    labels = []
    valid_symbols = []
    window_size = int(window_ms * fs / 1000)
    n_channels = signal.shape[1]
    for r_peak, symbol in r_peaks_with_labels:
        start = max(0, r_peak - window_size)
        end = min(len(signal), r_peak + window_size)
        if r_peak - start < window_size * 0.7 or end - r_peak < window_size * 0.7:
            continue
        patch = signal[start:end]
        if downsample_factor > 1:
            patch = patch[::downsample_factor]
        patches.append(patch)
        labels.append(symbol)
        valid_symbols.append(symbol)
    if not patches:
        return (np.array([]), np.array([]), [])
    return (np.array(patches), np.array(labels), valid_symbols)

def load_mitbih_r_wave_centered(data_path, window_ms=250, normalize=True, fs=360, downsample_factor=1, use_time_features=False, sample_count_limits=None):
    if sample_count_limits:
        pass
    all_data = []
    all_labels = []
    record_list = []
    beat_types = {}
    for file in os.listdir(data_path):
        if file.endswith('.dat') and (not file.endswith('.hea')) and (not file.endswith('.atr')):
            record_name = file.split('.')[0]
            if record_name not in record_list:
                record_list.append(record_name)
    for record_name in record_list:
        try:
            record_path = os.path.join(data_path, record_name)
            annotation = wfdb.rdann(record_path, 'atr')
            symbols = annotation.symbol
            for sym in symbols:
                if sym not in beat_types:
                    beat_types[sym] = 0
                beat_types[sym] += 1
        except Exception as e:
            continue
    for sym, count in beat_types.items():
        pass
    selected_types = {}
    for sym, count in beat_types.items():
        if sample_count_limits:
            if count >= sample_count_limits['min_samples'] and count <= sample_count_limits['max_samples']:
                selected_types[sym] = count
        else:
            selected_types[sym] = count
    type_to_label = {sym: i for i, sym in enumerate(sorted(selected_types.keys()))}
    num_classes = len(type_to_label)
    for sym, label in type_to_label.items():
        pass
    extracted_counts = {label: 0 for label in type_to_label.values()}
    for record_name in record_list:
        try:
            record_path = os.path.join(data_path, record_name)
            r_peaks_with_labels, signals, record_fs = detect_r_peaks_with_wfdb(data_path, record_name)
            if abs(record_fs - fs) > 1:
                pass
            filtered_r_peaks = [(pos, sym) for pos, sym in r_peaks_with_labels if sym in type_to_label]
            if len(filtered_r_peaks) == 0:
                continue
            heartbeats, symbols, valid_symbols = r_wave_centered_patches(signals, filtered_r_peaks, record_fs, window_ms=window_ms, downsample_factor=downsample_factor)
            if len(heartbeats) == 0:
                continue
            heartbeat_labels = np.array([type_to_label[sym] for sym in symbols])
            all_data.extend(heartbeats)
            all_labels.extend(heartbeat_labels)
            for sym in valid_symbols:
                if sym in type_to_label:
                    extracted_counts[type_to_label[sym]] += 1
        except Exception as e:
            continue
    all_data = np.array(all_data) if all_data else np.array([])
    all_labels = np.array(all_labels) if all_labels else np.array([])
    if len(all_labels) > 0:
        unique, counts = np.unique(all_labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        for label, count in zip(unique, counts):
            beat_type = [k for k, v in type_to_label.items() if v == label][0]
    label_to_type = {v: k for k, v in type_to_label.items()}
    return (all_data, all_labels, None, type_to_label, num_classes, label_to_type)

def get_r_wave_data_loaders(configs):
    window_ms = configs.r_wave_window_ms if hasattr(configs, 'r_wave_window_ms') else 250
    data, labels, timestamps, type_to_label, num_classes, label_to_type = load_mitbih_r_wave_centered(configs.root_path, window_ms=window_ms, normalize=True, fs=configs.fs, downsample_factor=configs.downsample_factor, use_time_features=configs.use_time_features, sample_count_limits=configs.sample_count_limits if hasattr(configs, 'sample_count_limits') else None)
    train_ratio = configs.train_ratio
    val_ratio = configs.val_ratio
    test_ratio = configs.test_ratio
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_ratio, random_state=42, stratify=labels)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_ratio / (train_ratio + val_ratio), random_state=42, stratify=train_labels)
    train_dataset = RWaveECGDataset(train_data, train_labels)
    val_dataset = RWaveECGDataset(val_data, val_labels)
    test_dataset = RWaveECGDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader, num_classes, label_to_type)

def r_wave_data_provider(configs):
    train_loader, val_loader, test_loader, num_classes, label_to_type = get_r_wave_data_loaders(configs)
    configs.num_class = num_classes
    configs.label_to_type = label_to_type
    for label, beat_type in label_to_type.items():
        pass
    for batch in train_loader:
        if len(batch) == 3:
            x, y, timestamps = batch
        else:
            x, y = batch
        break
    seq_len = x.shape[1]
    n_channels = x.shape[2] if len(x.shape) > 2 else 1
    configs.patch_len = seq_len
    configs.stride = seq_len
    data_loaders = (train_loader, val_loader, test_loader)
    dims = (seq_len, n_channels)
    return (data_loaders, dims)
