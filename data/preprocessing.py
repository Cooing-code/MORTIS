import numpy as np
import logging
import pywt
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
logger = logging.getLogger(__name__)

class ECGPreprocessor:

    def __init__(self, use_wavelet=False, wavelet_config=None, use_highpass=False, highpass_config=None, use_notch=True, notch_config=None, use_median=False, median_kernel_ms=50, normalization_method='minmax'):
        self.use_wavelet = use_wavelet
        self.wavelet_config = wavelet_config if wavelet_config else {'wavelet': 'bior3.7', 'level': 3, 'threshold_scale': 0.7}
        self.use_highpass = use_highpass
        self.highpass_config = highpass_config if highpass_config else {'cutoff': 0.5, 'order': 2}
        self.use_notch = use_notch
        self.notch_config = notch_config if notch_config else {'freq': 60.0, 'q': 30.0}
        self.use_median = use_median
        self.median_kernel_ms = median_kernel_ms
        self.normalization_method = normalization_method

    def _apply_wavelet_denoising(self, signal_channel, fs):
        coeffs = pywt.wavedec(signal_channel, self.wavelet_config['wavelet'], level=self.wavelet_config['level'])
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal_channel))) * self.wavelet_config.get('threshold_scale', 0.7)
        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
        denoised_signal = pywt.waverec(coeffs, self.wavelet_config['wavelet'])
        return denoised_signal[:len(signal_channel)]

    def _apply_highpass_filter(self, signal_channel, fs):
        cutoff = self.highpass_config['cutoff']
        order = self.highpass_config['order']
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal_channel)

    def _apply_notch_filter(self, signal_channel, fs):
        notch_freq = self.notch_config['freq']
        q_factor = self.notch_config.get('q', 30.0)
        if q_factor is None:
            pass
            q_factor = 30.0
        nyquist = 0.5 * fs
        freq = notch_freq / nyquist
        b, a = iirnotch(freq, q_factor)
        return filtfilt(b, a, signal_channel)

    def _apply_median_filter(self, signal_channel, fs):
        kernel_size_samples = int(self.median_kernel_ms / 1000.0 * fs)
        if kernel_size_samples % 2 == 0:
            kernel_size_samples += 1
        return medfilt(signal_channel, kernel_size=kernel_size_samples)

    def _apply_normalization(self, signal_channel):
        signal_reshaped = signal_channel.reshape(-1, 1)
        if self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.normalization_method == 'zscore':
            scaler = StandardScaler()
        elif self.normalization_method == 'robust':
            scaler = RobustScaler()
        elif self.normalization_method == 'none':
            return signal_channel
        else:
            return signal_channel
        try:
            normalized_signal = scaler.fit_transform(signal_reshaped).flatten()
        except ValueError as e:
            pass
            return signal_channel
        return normalized_signal

    def preprocess_signal(self, raw_signals, fs):
        processed_signals = np.copy(raw_signals)
        num_channels = raw_signals.shape[1]
        for i in range(num_channels):
            signal_channel = processed_signals[:, i]
            if self.use_median:
                signal_channel = self._apply_median_filter(signal_channel, fs)
            if self.use_highpass:
                signal_channel = self._apply_highpass_filter(signal_channel, fs)
            if self.use_notch:
                signal_channel = self._apply_notch_filter(signal_channel, fs)
            if self.use_wavelet:
                signal_channel = self._apply_wavelet_denoising(signal_channel, fs)
            signal_channel = self._apply_normalization(signal_channel)
            processed_signals[:, i] = signal_channel
        return processed_signals
