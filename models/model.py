import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GATConv
    _GAT_AVAILABLE = True
except ImportError:
    GATConv = None
    _GAT_AVAILABLE = False
from layers.self_attention import DSAttention, FullAttention, ProbAttention, AttentionLayer
from layers.transformer_encoder import Encoder, EncoderLayer, ConvLayer
from models.channel_attention import ChannelAttention, CrossChannelAttention
from models.patch_embedding import AdaptivePatchEmbedding
from layers.FANLayer import FANLayer
from .morphology_cnn import MorphologyCNN

class AttentionPooling1D(nn.Module):

    def __init__(self, d_model):
        super(AttentionPooling1D, self).__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.attention_weights(x)
        scores = scores.squeeze(-1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled

class Transpose(nn.Module):

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = (dims, contiguous)

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

class TSTModel(nn.Module):

    def __init__(self, configs):
        super(TSTModel, self).__init__()
        self.configs = configs
        self.num_class = configs.num_class
        self.n_patches_group = getattr(configs, 'num_consecutive_patches', 3)
        self.patch_embedding = AdaptivePatchEmbedding(d_model=configs.d_model, patch_len=configs.patch_len, stride=configs.stride, padding=configs.padding, dropout=configs.dropout)
        self.dynamic_feature_dim = getattr(configs, 'dynamic_feature_dim', 12)
        self.dynamic_feature_projection = nn.Linear(self.dynamic_feature_dim, configs.d_model)
        self.dynamic_feature_dropout = nn.Dropout(configs.dropout)
        if configs.attention_type == 'full':
            attention_class = FullAttention
        elif configs.attention_type == 'prob':
            attention_class = ProbAttention
        else:
            attention_class = DSAttention
        self.channel_encoders = nn.ModuleList()
        positional_encoding_type = getattr(configs, 'positional_encoding_type', 'absolute')
        use_rotary_pos_enc = getattr(configs, 'use_rotary_pos_enc', False)
        use_conv_layers = getattr(configs, 'use_conv_layers_in_encoder', False)
        e_layers = getattr(configs, 'e_layers', 2)
        use_fan_flag = getattr(configs, 'use_fan_layer', False)
        for i in range(configs.enc_in):
            attention = attention_class(False, configs.factor, attention_dropout=configs.dropout, output_attention=True)
            encoder_layers = [EncoderLayer(attention=AttentionLayer(attention, configs.d_model, configs.n_heads_per_channel if hasattr(configs, 'n_heads_per_channel') else configs.n_heads, d_keys=configs.d_keys, d_values=configs.d_values, use_rotary=use_rotary_pos_enc), d_model=configs.d_model, d_ff=configs.d_ff, dropout=configs.dropout, activation=configs.activation, use_fan_layer=use_fan_flag, fan_p_ratio=getattr(configs, 'fan_p_ratio', 0.25), fan_activation=getattr(configs, 'fan_activation', 'gelu'), fan_with_gate=getattr(configs, 'fan_with_gate', False)) for _ in range(e_layers)]
            conv_layers_list = None
            if use_conv_layers:
                if e_layers > 1:
                    conv_layers_list = [ConvLayer(configs.d_model) for _ in range(e_layers - 1)]
                    norm_layer = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
                else:
                    norm_layer = nn.LayerNorm(configs.d_model)
            else:
                norm_layer = nn.LayerNorm(configs.d_model)
            channel_encoder = Encoder(attn_layers=encoder_layers, conv_layers=conv_layers_list, norm_layer=norm_layer, positional_encoding_type=positional_encoding_type, d_model=configs.d_model, max_len=configs.max_seq_len if hasattr(configs, 'max_seq_len') else 5000)
            self.channel_encoders.append(channel_encoder)
        self.use_channel_attention = getattr(configs, 'use_channel_attention', True)
        self.channel_attn = None
        if self.use_channel_attention:
            self.channel_attn = ChannelAttention(n_channels=configs.enc_in, d_model=configs.d_model, dropout=configs.dropout)
        self.use_cross_channel = getattr(configs, 'use_cross_channel', False)
        self.cross_channel_attn = None
        if self.use_cross_channel:
            self.cross_channel_attn = CrossChannelAttention(n_channels=configs.enc_in, d_model=configs.d_model, n_heads=configs.n_heads_per_channel if hasattr(configs, 'n_heads_per_channel') else 2, dropout=configs.dropout)
        self.use_gat = getattr(configs, 'use_gat', False)
        self.gat_layer = None
        if self.use_gat:
            if not _GAT_AVAILABLE:
                raise ImportError('GATConv not found. Please ensure torch_geometric is installed.')
            self.gat_layer = GATConv(configs.d_model, configs.d_model, heads=getattr(configs, 'gat_heads', 2), dropout=getattr(configs, 'gat_dropout', 0.1), negative_slope=getattr(configs, 'gat_leaky_relu_negative_slope', 0.2), concat=False)
        self.use_combined_encoder = getattr(configs, 'use_combined_encoder', True)
        if self.use_combined_encoder:
            attention = attention_class(False, configs.factor, attention_dropout=configs.dropout, output_attention=True)
            encoder_layers_combined = [EncoderLayer(attention=AttentionLayer(attention, configs.d_model, configs.n_heads, d_keys=configs.d_keys, d_values=configs.d_values, use_rotary=use_rotary_pos_enc), d_model=configs.d_model, d_ff=configs.d_ff, dropout=configs.dropout, activation=configs.activation, use_fan_layer=use_fan_flag, fan_p_ratio=getattr(configs, 'fan_p_ratio', 0.25), fan_activation=getattr(configs, 'fan_activation', 'gelu'), fan_with_gate=getattr(configs, 'fan_with_gate', False)) for _ in range(e_layers)]
            conv_layers_combined_list = None
            if use_conv_layers:
                if e_layers > 1:
                    conv_layers_combined_list = [ConvLayer(configs.d_model) for _ in range(e_layers - 1)]
                    norm_layer_combined = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
                else:
                    norm_layer_combined = nn.LayerNorm(configs.d_model)
            else:
                norm_layer_combined = nn.LayerNorm(configs.d_model)
            self.combined_encoder = Encoder(attn_layers=encoder_layers_combined, conv_layers=conv_layers_combined_list, norm_layer=norm_layer_combined, positional_encoding_type=positional_encoding_type, d_model=configs.d_model, max_len=configs.max_seq_len if hasattr(configs, 'max_seq_len') else 5000)
        self.use_morphology_cnn = getattr(configs, 'use_morphology_cnn', False)
        self.morphology_cnn = None
        cnn_output_dim = 0
        if self.use_morphology_cnn:
            self.morphology_cnn = MorphologyCNN(input_channels=configs.enc_in, output_dim=configs.morphology_cnn_output_dim, kernels=configs.morphology_cnn_kernels, num_filters=configs.morphology_cnn_num_filters)
            cnn_output_dim = configs.morphology_cnn_output_dim
        transformer_output_dim = 0
        channel_fusion_strategy = getattr(self.configs, 'channel_fusion', 'concat')
        use_ch_attn = getattr(configs, 'use_channel_attention', True)
        use_cross_attn = getattr(configs, 'use_cross_channel', False)
        use_comb_enc = getattr(configs, 'use_combined_encoder', True)
        use_gat_flag = self.use_gat
        if use_gat_flag and self.gat_layer is not None:
            transformer_output_dim = configs.d_model
        elif use_cross_attn and hasattr(self, 'cross_channel_attn') and (self.cross_channel_attn is not None):
            transformer_output_dim = configs.d_model
        elif use_ch_attn and hasattr(self, 'channel_attn') and (self.channel_attn is not None):
            transformer_output_dim = configs.d_model
        elif use_comb_enc and hasattr(self, 'combined_encoder') and (self.combined_encoder is not None):
            transformer_output_dim = configs.d_model
        else:
            transformer_output_dim = configs.d_model * configs.enc_in
        self.head_nf = transformer_output_dim
        if self.use_morphology_cnn and self.morphology_cnn is not None:
            cnn_output_dim = getattr(configs, 'morphology_cnn_output_dim', 0)
            feature_fusion_strategy = getattr(configs, 'feature_fusion_strategy', 'concat')
            if feature_fusion_strategy == 'concat':
                self.head_nf += cnn_output_dim
            elif feature_fusion_strategy in ['add', 'weighted_sum']:
                if transformer_output_dim != cnn_output_dim:
                    pass
        if self.head_nf <= 0:
            raise ValueError(f'Calculated classification head input dimension head_nf ({self.head_nf}) is invalid.')
        self.pooling_mode = getattr(configs, 'pooling_mode', 'mean')
        self.attention_pool = None
        if self.pooling_mode == 'attention':
            self.attention_pool = AttentionPooling1D(configs.d_model)
        self.projection = nn.Linear(self.head_nf, configs.num_class)
        self.stage1_classifier = None
        self.stage2_classifier = None
        self.use_two_stage_classifier = getattr(configs, 'use_two_stage_classifier', False)
        if self.use_two_stage_classifier:
            self.stage1_classifier = nn.Linear(self.head_nf, 2)
            if not hasattr(configs, 'num_abnormal_classes') or configs.num_abnormal_classes is None:
                raise ValueError('Two-stage classifier enabled but configs.num_abnormal_classes not set or invalid!')
            self.stage2_classifier = nn.Linear(self.head_nf, configs.num_abnormal_classes)
        self.tau = nn.Parameter(torch.ones(1))
        initial_num_patch_guess = (self.n_patches_group * configs.patch_len - configs.patch_len) // configs.stride + 1 if configs.patch_len and configs.stride else 128
        self.delta = nn.Parameter(torch.zeros(1, initial_num_patch_guess))
        self.is_time_length_matched = True
        num_nodes = self.configs.enc_in
        edge_index_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index_list.append([i, j])
        self.edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    def get_label_names(self, data_path=None):
        if hasattr(self.configs, 'label_to_type') and self.configs.label_to_type:
            label_names = []
            for i in range(len(self.configs.label_to_type)):
                if i in self.configs.label_to_type:
                    label_names.append(self.configs.label_to_type[i])
                else:
                    label_names.append(f'Class{i}')
            return label_names
        if hasattr(self.configs, 'label_map'):
            return [self.configs.label_map.get(i, f'Class{i}') for i in range(self.configs.num_class)]
        if hasattr(self.configs, 'num_class'):
            return [f'Class{i}' for i in range(self.configs.num_class)]
        else:
            return ['N', 'V', 'S', 'F', 'Q']

    def forward(self, x, dynamic_features, return_internals=False, return_features=False):
        batch_size = x.shape[0]
        input_seq_len = x.shape[1]
        n_vars = x.shape[2]
        use_ch_attn = getattr(self.configs, 'use_channel_attention', True)
        use_comb_enc = getattr(self.configs, 'use_combined_encoder', True)
        fusion_strategy = getattr(self.configs, 'channel_fusion', 'concat')
        processed_by_gat = False
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-05)
        x = x / stdev
        channel_embeddings, n_vars_from_patcher = self.patch_embedding(x)
        if n_vars_from_patcher != n_vars:
            n_vars = n_vars_from_patcher
        num_patch = channel_embeddings[0].shape[1]
        self.configs.num_patch = num_patch
        projected_dynamic_features = None
        if dynamic_features is not None:
            if dynamic_features.shape[1] != input_seq_len:
                pass
            if dynamic_features.shape[1] != num_patch:
                dynamic_features_resampled = F.interpolate(dynamic_features.transpose(1, 2), size=num_patch, mode='linear', align_corners=False).transpose(1, 2)
            else:
                dynamic_features_resampled = dynamic_features
            if dynamic_features_resampled.shape[2] != self.dynamic_feature_dim:
                if dynamic_features_resampled.shape[2] > self.dynamic_feature_dim:
                    dynamic_features_resampled = dynamic_features_resampled[:, :, :self.dynamic_feature_dim]
                else:
                    padding = torch.zeros(batch_size, num_patch, self.dynamic_feature_dim - dynamic_features_resampled.shape[2], device=x.device)
                    dynamic_features_resampled = torch.cat([dynamic_features_resampled, padding], dim=2)
            projected_dynamic_features = self.dynamic_feature_projection(dynamic_features_resampled)
            projected_dynamic_features = self.dynamic_feature_dropout(projected_dynamic_features)
        if projected_dynamic_features is not None:
            for i in range(len(channel_embeddings)):
                channel_embeddings[i] = channel_embeddings[i] + projected_dynamic_features
        if hasattr(self, 'delta') and self.delta.shape[1] != num_patch:
            self.delta = nn.Parameter(torch.zeros(1, num_patch, device=x.device))
        cnn_features = None
        if self.use_morphology_cnn and self.morphology_cnn is not None:
            cnn_features = self.morphology_cnn(x)
        intermediate_features = {}
        if return_internals:
            intermediate_features['patch_embeddings_channel'] = [emb.detach().cpu() for emb in channel_embeddings]
            intermediate_features['dynamic_features_projected'] = projected_dynamic_features.detach().cpu()
            if cnn_features is not None:
                intermediate_features['cnn_features'] = cnn_features.detach().cpu()
        channel_outputs = []
        channel_attns = []
        for i, channel_embed in enumerate(channel_embeddings):
            channel_output, channel_attn = self.channel_encoders[i](channel_embed)
            channel_outputs.append(channel_output)
            channel_attns.append(channel_attn)
        processed_by_gat = False
        gat_fused_output = None
        if self.use_gat and self.gat_layer is not None:
            processed_by_gat = True
            stacked_channel_output = torch.stack(channel_outputs, dim=2)
            B, N_patch, C, D_model = stacked_channel_output.shape
            try:
                flat_features = stacked_channel_output.reshape(-1, D_model)
                batch_size_gat = B * N_patch
                edges_per_graph = self.edge_index.size(1)
                batch_index = torch.arange(batch_size_gat, device=flat_features.device)
                batch_index = batch_index.repeat_interleave(C)
                batch_edge_index = []
                for b in range(batch_size_gat):
                    offset_edges = self.edge_index.clone() + b * C
                    batch_edge_index.append(offset_edges)
                batch_edge_index = torch.cat(batch_edge_index, dim=1).to(flat_features.device)
                gat_output_batch = self.gat_layer(flat_features, batch_edge_index)
                gat_output_reshaped = gat_output_batch.reshape(B, N_patch, C, D_model)
            except (RuntimeError, ValueError, TypeError) as e:
                gat_results = []
                max_samples_per_batch = max(1, 32 // N_patch)
                for b_start in range(0, B, max_samples_per_batch):
                    b_end = min(B, b_start + max_samples_per_batch)
                    current_batch = stacked_channel_output[b_start:b_end]
                    sub_B = current_batch.shape[0]
                    current_batch = current_batch.reshape(sub_B * N_patch, C, D_model)
                    sub_batch_results = []
                    for idx in range(sub_B * N_patch):
                        node_features = current_batch[idx]
                        edge_index = self.edge_index.to(node_features.device)
                        gat_output_sample = self.gat_layer(node_features, edge_index)
                        sub_batch_results.append(gat_output_sample)
                    sub_batch_output = torch.stack(sub_batch_results, dim=0)
                    sub_batch_output = sub_batch_output.reshape(sub_B, N_patch, C, D_model)
                    gat_results.append(sub_batch_output)
                gat_output_reshaped = torch.cat(gat_results, dim=0)
            gat_fused_output = gat_output_reshaped.mean(dim=2)
        cross_channel_attn_weights = None
        if not processed_by_gat and self.use_channel_attention and self.use_cross_channel and hasattr(self, 'cross_channel_attn'):
            enhanced_outputs, cross_channel_attn_weights = self.cross_channel_attn(channel_outputs)
            if isinstance(enhanced_outputs, torch.Tensor) and enhanced_outputs.shape[0] == n_vars:
                channel_outputs = list(torch.unbind(enhanced_outputs, dim=0))
                if return_internals:
                    intermediate_features['cross_channel_enhanced_outputs'] = [out.detach().cpu() for out in channel_outputs]
        fused_output = None
        channel_fusion_weights = None
        if not processed_by_gat and self.use_channel_attention and hasattr(self, 'channel_attn'):
            fused_output, channel_fusion_weights = self.channel_attn(channel_outputs)
            if return_internals:
                intermediate_features['channel_attention_fused_output'] = fused_output.detach().cpu()
        combined_encoder_input = None
        combined_output = None
        combined_attn_weights = None
        if self.use_combined_encoder and hasattr(self, 'combined_encoder'):
            if processed_by_gat and gat_fused_output is not None:
                combined_encoder_input = gat_fused_output
            elif fused_output is not None:
                combined_encoder_input = fused_output
            else:
                pass
            if combined_encoder_input is not None:
                combined_output, combined_attn_list = self.combined_encoder(combined_encoder_input)
                combined_attn_weights = combined_attn_list
                if return_internals:
                    intermediate_features['combined_encoder_output'] = combined_output.detach().cpu()
            else:
                combined_output = None
                combined_attn_weights = None
        base_sequence_output = None
        if processed_by_gat and gat_fused_output is not None:
            base_sequence_output = gat_fused_output
        elif not processed_by_gat and self.use_channel_attention and (fused_output is not None):
            base_sequence_output = fused_output
        else:
            error_msg = 'Model configuration requires either GAT (use_gat=True) or Channel Attention (use_channel_attention=True) to process channel information, but neither provided a valid output. '
            if not self.use_gat and (not self.use_channel_attention):
                error_msg += 'Please enable at least one of them in the configuration.'
            elif self.use_gat and gat_fused_output is None:
                error_msg += 'GAT was enabled but its output was None.'
            elif self.use_channel_attention and fused_output is None:
                error_msg += 'Channel Attention was enabled but its output was None.'
            raise ValueError(error_msg)
        final_sequence_output = base_sequence_output
        if self.use_combined_encoder and hasattr(self, 'combined_encoder') and (combined_output is not None):
            final_sequence_output = combined_output
        elif self.use_combined_encoder and hasattr(self, 'combined_encoder') and (combined_output is None):
            pass
        final_representation_before_pool = final_sequence_output
        if self.pooling_mode == 'mean':
            final_representation = final_sequence_output.mean(dim=1)
        elif self.pooling_mode == 'max':
            final_representation = final_sequence_output.max(dim=1)[0]
        elif self.pooling_mode == 'last':
            final_representation = final_sequence_output[:, -1, :]
        elif self.pooling_mode == 'attention' and self.attention_pool is not None:
            final_representation = self.attention_pool(final_sequence_output)
        else:
            final_representation = final_sequence_output[:, -1, :]
        if return_internals:
            intermediate_features['final_representation_before_fusion'] = final_representation.detach().cpu()
            intermediate_features['final_sequence_before_pooling'] = final_representation_before_pool.detach().cpu()
        fused_features = final_representation
        if self.use_morphology_cnn and cnn_features is not None:
            feature_fusion_strategy = getattr(self.configs, 'feature_fusion_strategy', 'concat')
            if feature_fusion_strategy == 'concat':
                if final_representation is None:
                    raise ValueError('Transformer final_representation is None, cannot concatenate.')
                fused_features = torch.cat((final_representation, cnn_features), dim=1)
            elif feature_fusion_strategy in ['add', 'weighted_sum']:
                if final_representation is not None and final_representation.shape == cnn_features.shape:
                    if feature_fusion_strategy == 'add':
                        fused_features = final_representation + cnn_features
                    else:
                        fused_features = final_representation + cnn_features
                else:
                    pass
            elif feature_fusion_strategy == 'attention':
                pass
        if fused_features is None:
            raise ValueError("Final features 'fused_features' for projection are None!")
        if return_internals:
            intermediate_features['final_fused_features_before_projection'] = fused_features.detach().cpu()
        if fused_features.shape[-1] != self.head_nf:
            raise RuntimeError(f'Dimension mismatch for projection: expected {self.head_nf}, got {fused_features.shape[-1]}')
        if self.use_two_stage_classifier and self.stage1_classifier is not None and (self.stage2_classifier is not None):
            logits_stage1 = self.stage1_classifier(fused_features)
            logits_stage2 = self.stage2_classifier(fused_features)
            output1 = logits_stage1
            output2 = logits_stage2
            if return_internals:
                intermediate_features['logits_stage1'] = logits_stage1.detach().cpu()
            if return_internals:
                intermediate_features['logits_stage2'] = logits_stage2.detach().cpu()
        else:
            output1 = self.projection(fused_features)
            output2 = None
            if return_internals:
                intermediate_features['logits_single_stage'] = output1.detach().cpu()
        attention_weights = {'channel_encoder_attns': channel_attns, 'cross_channel_attn': cross_channel_attn_weights, 'combined_encoder_attns': combined_attn_weights, 'channel_fusion_weights': channel_fusion_weights}
        embedding_output = fused_features
        if return_features:
            return (embedding_output, output1, output2, attention_weights)
        elif return_internals:
            processed_attns = {}
            for key, val in attention_weights.items():
                if isinstance(val, torch.Tensor):
                    processed_attns[key] = val.detach().cpu()
                elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                    processed_attns[key] = [v.detach().cpu() for v in val]
                else:
                    processed_attns[key] = val
            intermediate_features['embedding_output'] = embedding_output.detach().cpu()
            return (output1, output2, processed_attns, intermediate_features)
        else:
            return (output1, output2, attention_weights)
