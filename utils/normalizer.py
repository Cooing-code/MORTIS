import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

class Normalizer:

    def __init__(self, method='robust', **kwargs):
        self.method = method
        self.scaler = None
        self.fit_shape = None
        self.initialized = False
        if method == 'robust':
            self.scaler = RobustScaler(**kwargs)
        elif method == 'standard':
            self.scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            self.scaler = MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported normalization method: {method}. Supported methods: 'robust', 'standard', 'minmax'")

    def fit(self, data):
        self.fit_shape = data.shape
        if len(data.shape) == 1:
            reshaped_data = data.reshape(-1, 1)
        else:
            original_shape = data.shape
            reshaped_data = data.reshape(-1, original_shape[-1])
        self.scaler.fit(reshaped_data)
        self.initialized = True
        return self

    def transform(self, data, copy=True):
        if not self.initialized:
            raise RuntimeError('Normalizer has not been fitted yet. Call fit method first.')
        original_shape = data.shape
        if len(data.shape) == 1:
            reshaped_data = data.reshape(-1, 1)
            transformed = self.scaler.transform(reshaped_data)
            return transformed.ravel()
        else:
            reshaped_data = data.reshape(-1, original_shape[-1])
            transformed = self.scaler.transform(reshaped_data)
            return transformed.reshape(original_shape)

    def fit_transform(self, data, copy=True):
        self.fit(data)
        return self.transform(data, copy=copy)

    def inverse_transform(self, data, copy=True):
        if not self.initialized:
            raise RuntimeError('Normalizer has not been fitted yet. Call fit method first.')
        original_shape = data.shape
        if len(data.shape) == 1:
            reshaped_data = data.reshape(-1, 1)
            inverse_transformed = self.scaler.inverse_transform(reshaped_data)
            return inverse_transformed.ravel()
        else:
            reshaped_data = data.reshape(-1, original_shape[-1])
            inverse_transformed = self.scaler.inverse_transform(reshaped_data)
            return inverse_transformed.reshape(original_shape)

    @staticmethod
    def create_normalizer_from_config(config):
        method = config.normalization_method
        if method not in ['robust', 'standard', 'minmax']:
            raise ValueError(f"Unsupported normalization method: {method}. Supported methods: 'robust', 'standard', 'minmax'")
        params = config.normalization_params.get(method, {})
        return Normalizer(method=method, **params)
