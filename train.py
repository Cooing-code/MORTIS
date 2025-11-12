import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import random
import time
from tqdm import tqdm
from models.model import TSTModel
from data.data_factory import get_data_loaders
from utils.loss import get_loss_criterion, SingleStageLossCalculator, TwoStageLossCalculator, hard_triplet_miner
from utils.metrics import metric
from configs import Config
import logging
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from collections import Counter
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def seed_everything(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch, current_lr, configs):
    if configs.use_warmup and epoch < configs.warmup_epochs:
        lr = configs.warmup_start_lr + (configs.learning_rate - configs.warmup_start_lr) * (epoch / configs.warmup_epochs)
    else:
        lr = current_lr
    min_lr = getattr(configs, 'scheduler_eta_min', 0) if configs.scheduler_type == 'CosineAnnealingLR' else getattr(configs, 'scheduler_min_lr', 0)
    lr = max(lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def _process_train_batch(model, batch, criterion_stage1, criterion_stage2, configs, device):
    results = {'loss': torch.tensor(0.0, device=device), 'loss_s1': torch.tensor(0.0, device=device), 'loss_s2': torch.tensor(0.0, device=device), 'correct_s1': 0, 'total_s1': 0, 'correct_s2': 0, 'total_s2': 0, 'correct_final': 0, 'total_final': 0}
    if not isinstance(batch, dict):
        logger.error(f'Incorrect training batch format (expected dict, got {type(batch)})')
        return results
    required_keys = {'data', 'dynamic_features', 'fine_grained_label'}
    if configs.use_two_stage_classifier:
        required_keys.update({'stage1_label', 'stage2_label'})
    if not required_keys.issubset(batch.keys()):
        logger.error(f'Training batch missing keys (expected {required_keys}, got {batch.keys()})')
        return results
    x = batch['data'].to(device)
    y_fine = batch['fine_grained_label'].to(device)
    d_feat = batch['dynamic_features'].to(device)
    stage1_labels = batch.get('stage1_label', None)
    stage2_labels = batch.get('stage2_label', None)
    if stage1_labels is not None:
        stage1_labels = stage1_labels.to(device)
    if stage2_labels is not None:
        stage2_labels = stage2_labels.to(device)
    if configs.use_two_stage_classifier:
        if stage1_labels is None or stage2_labels is None:
            logger.error('Training error: Two-stage enabled but missing stage1/stage2 labels.')
            return results
        get_embeddings = configs.stage1_loss_type.lower() == 'tripletloss'
        features, outputs_stage1, outputs_stage2 = (None, None, None)
        if get_embeddings:
            features, outputs_stage1, outputs_stage2, _ = model(x, d_feat, return_features=True)
        else:
            outputs_stage1, outputs_stage2, _ = model(x, d_feat)
        loss_stage1 = torch.tensor(0.0, device=device, requires_grad=True)
        if outputs_stage1 is not None and criterion_stage1 is not None:
            if configs.stage1_loss_type.lower() == 'tripletloss':
                if features is None:
                    raise ValueError('Features required for TripletLoss.')
                anchor_idx, pos_idx, neg_idx = hard_triplet_miner(features, stage1_labels)
                if len(anchor_idx) > 0:
                    loss_stage1 = criterion_stage1(features[anchor_idx], features[pos_idx], features[neg_idx])
                else:
                    loss_stage1 = torch.tensor(0.0, device=device, requires_grad=True)
            elif configs.stage1_loss_type.lower() == 'bcewithlogits':
                stage1_labels_one_hot = F.one_hot(stage1_labels, num_classes=2).float()
                loss_stage1 = criterion_stage1(outputs_stage1, stage1_labels_one_hot)
            elif configs.stage1_loss_type.lower() in ['crossentropy', 'focalloss']:
                loss_stage1 = criterion_stage1(outputs_stage1, stage1_labels)
        loss_stage2 = torch.tensor(0.0, device=device)
        if outputs_stage2 is not None and criterion_stage2 is not None:
            abnormal_mask = stage1_labels == 1
            if abnormal_mask.sum() > 0:
                abnormal_outputs_s2 = outputs_stage2[abnormal_mask]
                abnormal_labels_s2 = stage2_labels[abnormal_mask]
                valid_mask_s2 = abnormal_labels_s2 != -100
                if valid_mask_s2.sum() > 0:
                    abnormal_outputs_s2 = abnormal_outputs_s2[valid_mask_s2]
                    abnormal_labels_s2 = abnormal_labels_s2[valid_mask_s2]
                    if abnormal_outputs_s2.shape[0] > 0:
                        if configs.stage2_loss_type.lower() == 'bcewithlogits':
                            y_one_hot = F.one_hot(abnormal_labels_s2, num_classes=configs.num_abnormal_classes).float()
                            loss_stage2 = criterion_stage2(abnormal_outputs_s2, y_one_hot)
                        elif configs.stage2_loss_type.lower() in ['crossentropy', 'focalloss']:
                            loss_stage2 = criterion_stage2(abnormal_outputs_s2, abnormal_labels_s2)
        loss = configs.stage1_loss_weight * loss_stage1 + configs.stage2_loss_weight * loss_stage2
        results['loss'] = loss
        results['loss_s1'] = loss_stage1.detach()
        results['loss_s2'] = loss_stage2.detach()
        if outputs_stage1 is not None:
            _, predicted_stage1 = torch.max(outputs_stage1.data, 1)
            results['correct_s1'] = (predicted_stage1 == stage1_labels).sum().item()
            results['total_s1'] = stage1_labels.size(0)
        predicted_stage2 = torch.full_like(stage2_labels, -1)
        if abnormal_mask.sum() > 0 and outputs_stage2 is not None:
            abnormal_outputs_s2_data = outputs_stage2[abnormal_mask].data
            if abnormal_outputs_s2_data.numel() > 0:
                _, predicted_stage2_abnormal = torch.max(abnormal_outputs_s2_data, 1)
                predicted_stage2[abnormal_mask] = predicted_stage2_abnormal
                valid_mask_s2_acc = stage2_labels[abnormal_mask] != -100
                if valid_mask_s2_acc.sum() > 0:
                    results['correct_s2'] = (predicted_stage2_abnormal[valid_mask_s2_acc] == stage2_labels[abnormal_mask][valid_mask_s2_acc]).sum().item()
                    results['total_s2'] = valid_mask_s2_acc.sum().item()
        final_preds_batch = []
        stage2_map_inv = getattr(configs, 'stage2_to_abnormal_fine_map', {})
        normal_label_idx = getattr(configs, 'normal_label_index', -1)
        if normal_label_idx == -1:
            logger.error('configs.normal_label_index not found!')
            return results
        if outputs_stage1 is not None:
            for i, pred1 in enumerate(predicted_stage1):
                if pred1 == 0:
                    final_preds_batch.append(normal_label_idx)
                else:
                    stage2_pred = predicted_stage2[i].item()
                    final_pred = stage2_map_inv.get(stage2_pred, normal_label_idx)
                    final_preds_batch.append(final_pred)
            final_preds_batch = torch.tensor(final_preds_batch, device=device)
            results['correct_final'] = (final_preds_batch == y_fine).sum().item()
            results['total_final'] = y_fine.size(0)
    else:
        outputs, _, attention_weights = model(x, d_feat)
        loss = torch.tensor(0.0, device=device)
        if outputs is not None and criterion_stage1 is not None:
            if configs.stage1_loss_type.lower() == 'bcewithlogits':
                y_one_hot = F.one_hot(y_fine, num_classes=configs.num_class).float()
                loss = criterion_stage1(outputs, y_one_hot)
            elif configs.stage1_loss_type.lower() in ['crossentropy', 'focalloss']:
                loss = criterion_stage1(outputs, y_fine)
            results['loss'] = loss
            _, predicted = torch.max(outputs.data, 1)
            results['correct_final'] = (predicted == y_fine).sum().item()
            results['total_final'] = y_fine.size(0)
            results['correct_s1'] = results['correct_final']
            results['total_s1'] = results['total_final']
            results['correct_s2'] = 0
            results['total_s2'] = 0
    return results

def _process_val_batch(model, batch, configs, device):
    results = {'outputs_s1': None, 'batch_y_fine': None, 'batch_s1_labels': None, 'final_preds': [], 'final_labels': [], 'final_probs': [], 's1_preds': [], 's1_labels': [], 's1_probs': [], 's2_preds': [], 's2_labels': [], 's2_probs': [], 's2_preds_abnormal_only': [], 's2_labels_abnormal_only': [], 's2_probs_abnormal_only': [], 'raw_data': None}
    is_two_stage = configs.use_two_stage_classifier
    if not isinstance(batch, dict):
        logger.error(f'Incorrect validation batch format (expected dict, got {type(batch)})')
        return results
    required_keys = {'data', 'dynamic_features', 'fine_grained_label'}
    if is_two_stage:
        required_keys.update({'stage1_label', 'stage2_label'})
    if not required_keys.issubset(batch.keys()):
        logger.error(f'Validation batch missing keys (expected {required_keys}, got {batch.keys()})')
        return results
    batch_x = batch['data'].to(device)
    batch_y_fine = batch['fine_grained_label'].to(device)
    batch_d_feat = batch['dynamic_features'].to(device)
    batch_stage1_labels = batch.get('stage1_label', None)
    batch_stage2_labels = batch.get('stage2_label', None)
    if batch_stage1_labels is not None:
        batch_stage1_labels = batch_stage1_labels.to(device)
    if batch_stage2_labels is not None:
        batch_stage2_labels = batch_stage2_labels.to(device)
    results['batch_y_fine'] = batch_y_fine.detach().cpu().numpy()
    if batch_stage1_labels is not None:
        results['batch_s1_labels'] = batch_stage1_labels.detach().cpu().numpy()
    if hasattr(configs, 'save_test_results') and configs.save_test_results:
        results['raw_data'] = batch_x.detach().cpu().numpy()
    outputs1, outputs2, _ = model(batch_x, batch_d_feat)
    results['outputs_s1'] = outputs1
    final_preds_batch = torch.full_like(batch_y_fine, -1)
    if is_two_stage:
        if outputs1 is None or batch_stage1_labels is None or batch_stage2_labels is None:
            logger.error('Validation error: Two-stage enabled but missing outputs1 or stage1/stage2 labels.')
            return results
        probs_s1 = torch.tensor([])
        pred_s1 = torch.tensor([])
        probs_s2_batch = torch.zeros(batch_x.size(0), configs.num_abnormal_classes, device=device)
        pred_s2 = torch.full_like(batch_stage2_labels, -1)
        if outputs1 is not None:
            probs_s1 = F.softmax(outputs1, dim=1)
            pred_s1 = torch.argmax(probs_s1, dim=1)
            results['s1_probs'].extend(probs_s1.detach().cpu().numpy())
            results['s1_preds'].extend(pred_s1.detach().cpu().numpy())
            results['s1_labels'].extend(batch_stage1_labels.detach().cpu().numpy())
        abnormal_mask_pred = pred_s1 == 1 if pred_s1.numel() > 0 else torch.tensor([], dtype=torch.bool, device=device)
        if abnormal_mask_pred.any() and outputs2 is not None:
            abnormal_outputs_s2 = outputs2[abnormal_mask_pred]
            if abnormal_outputs_s2.numel() > 0:
                probs_s2_abnormal = F.softmax(abnormal_outputs_s2, dim=1)
                pred_s2_abnormal = torch.argmax(probs_s2_abnormal, dim=1)
                pred_s2[abnormal_mask_pred] = pred_s2_abnormal
                probs_s2_batch[abnormal_mask_pred] = probs_s2_abnormal
        results['s2_preds'].extend(pred_s2.detach().cpu().numpy())
        results['s2_probs'].extend(probs_s2_batch.detach().cpu().numpy())
        results['s2_labels'].extend(batch_stage2_labels.detach().cpu().numpy())
        true_abnormal_mask = (batch_stage1_labels == 1) & (batch_stage2_labels != -100)
        if true_abnormal_mask.any():
            s2_preds_for_true = pred_s2[true_abnormal_mask]
            s2_labels_for_true = batch_stage2_labels[true_abnormal_mask]
            s2_probs_for_true = probs_s2_batch[true_abnormal_mask]
            results['s2_preds_abnormal_only'].extend(s2_preds_for_true.detach().cpu().numpy())
            results['s2_labels_abnormal_only'].extend(s2_labels_for_true.detach().cpu().numpy())
            results['s2_probs_abnormal_only'].extend(s2_probs_for_true.detach().cpu().numpy())
        stage2_map_inv = getattr(configs, 'stage2_to_abnormal_fine_map', {})
        normal_label_idx = getattr(configs, 'normal_label_index', -1)
        if normal_label_idx == -1:
            logger.error('configs.normal_label_index not found!')
            return results
        final_preds_batch = torch.full_like(batch_y_fine, normal_label_idx)
        if pred_s1.numel() > 0:
            abnormal_indices_pred = torch.where(pred_s1 == 1)[0]
            if len(abnormal_indices_pred) > 0:
                s2_preds_abnormal_part = pred_s2[abnormal_indices_pred]
                fine_labels_for_abnormal = torch.tensor([stage2_map_inv.get(s2_pred.item(), normal_label_idx) for s2_pred in s2_preds_abnormal_part], device=device, dtype=torch.long)
                final_preds_batch[abnormal_indices_pred] = fine_labels_for_abnormal
        batch_probs = torch.zeros(batch_x.size(0), device=device)
        if probs_s1.numel() > 0:
            for i in range(batch_x.size(0)):
                if pred_s1[i] == 0:
                    batch_probs[i] = probs_s1[i, 0] if probs_s1.shape[1] > 0 else 0.0
                elif pred_s2[i] != -1:
                    pred_s2_idx = pred_s2[i].item()
                    if 0 <= pred_s2_idx < probs_s2_batch.shape[1]:
                        prob_s1_ab = probs_s1[i, 1] if probs_s1.shape[1] > 1 else 1.0
                        batch_probs[i] = prob_s1_ab * probs_s2_batch[i, pred_s2_idx]
                    else:
                        batch_probs[i] = probs_s1[i, 1] if probs_s1.shape[1] > 1 else 1.0
                else:
                    batch_probs[i] = probs_s1[i, 1] if probs_s1.shape[1] > 1 else 1.0
        results['final_probs'].extend(batch_probs.detach().cpu().numpy())
    elif outputs1 is not None:
        probs = F.softmax(outputs1, dim=1)
        _, final_preds_batch = torch.max(probs.data, 1)
        results['final_probs'].extend(np.max(probs.detach().cpu().numpy(), axis=1))
    else:
        final_preds_batch = torch.full_like(batch_y_fine, -1)
        results['final_probs'].extend([0.0] * batch_y_fine.size(0))
    results['final_preds'].extend(final_preds_batch.detach().cpu().numpy())
    results['final_labels'].extend(batch_y_fine.detach().cpu().numpy())
    return results

def analyze_confidence_thresholds(all_preds, all_labels, all_confidences, num_thresholds=20, configs=None):
    min_conf = np.min(all_confidences)
    max_conf = np.max(all_confidences)
    thresholds = np.linspace(min_conf, max_conf, num_thresholds)
    accuracies = []
    rejection_rates = []
    retention_correct = []
    for threshold in thresholds:
        high_conf_idx = all_confidences >= threshold
        rejected_idx = all_confidences < threshold
        rejection_rate = np.mean(rejected_idx) if len(rejected_idx) > 0 else 0
        rejection_rates.append(rejection_rate)
        if np.sum(high_conf_idx) == 0:
            accuracies.append(0)
            retention_correct.append(0)
            continue
        high_conf_preds = np.array(all_preds)[high_conf_idx]
        high_conf_labels = np.array(all_labels)[high_conf_idx]
        accuracy = np.mean(high_conf_preds == high_conf_labels) if len(high_conf_preds) > 0 else 0
        accuracies.append(accuracy)
        correct_mask = np.array(all_preds) == np.array(all_labels)
        correct_and_retained = np.logical_and(correct_mask, high_conf_idx)
        retention = np.sum(correct_and_retained) / np.sum(correct_mask) if np.sum(correct_mask) > 0 else 0
        retention_correct.append(retention)
    if configs is not None and hasattr(configs, 'fixed_confidence_threshold') and (configs.fixed_confidence_threshold is not None):
        recommended_threshold = configs.fixed_confidence_threshold
    else:
        max_reject_rate = 0.3
        if configs is not None and hasattr(configs, 'max_rejection_rate'):
            max_reject_rate = configs.max_rejection_rate
        retention_correct = np.array(retention_correct)
        rejection_rates = np.array(rejection_rates)
        accuracies = np.array(accuracies)
        balance_scores = accuracies * (1 - rejection_rates)
        valid_idx = rejection_rates <= max_reject_rate
        if np.any(valid_idx):
            best_idx = np.argmax(balance_scores[valid_idx])
            recommended_idx = np.arange(len(thresholds))[valid_idx][best_idx]
        else:
            recommended_idx = np.argmin(np.abs(rejection_rates - max_reject_rate))
        recommended_threshold = thresholds[recommended_idx]
    return (thresholds, accuracies, rejection_rates, retention_correct, recommended_threshold)

def train():
    configs = Config()
    if configs.patch_len is None:
        configs.patch_len = configs.default_patch_len // configs.downsample_factor
    if configs.stride is None:
        configs.stride = configs.patch_len
    if configs.patch_len <= 0 or configs.stride <= 0:
        raise ValueError(f'Invalid parameters: patch_len={configs.patch_len}, stride={configs.stride}')
    seed_value = getattr(configs, 'seed', 3)
    seed_everything(seed_value)
    device = torch.device(configs.device if torch.cuda.is_available() else 'cpu')
    try:
        (train_loader, val_loader, test_loader), dims, pos_weights_tuple = get_data_loaders(configs)
        pos_weight_s1_calculated, pos_weight_s2_calculated = pos_weights_tuple
        if configs.use_two_stage_classifier:
            if not hasattr(configs, 'normal_label_index') or configs.normal_label_index is None:
                logger.error('Critical error: data_provider failed to set required config for two-stage classifier or it was disabled!')
                raise AttributeError('Configuration error: Two-stage classifier enabled, but normal_label_index is still not set after data loading.')
        elif hasattr(configs, '_init_use_two_stage_classifier') and configs._init_use_two_stage_classifier:
            logger.warning("Note: Two-stage classifier was automatically disabled because conditions (e.g., finding 'N' label) were not met.")
        configs._init_use_two_stage_classifier = configs.use_two_stage_classifier
        if pos_weight_s1_calculated is not None:
            pos_weight_s1_calculated = pos_weight_s1_calculated.to(device)
        if pos_weight_s2_calculated is not None:
            pos_weight_s2_calculated = pos_weight_s2_calculated.to(device)
        batch = next(iter(train_loader))
        if isinstance(batch, dict):
            x = batch.get('data')
            y = batch.get('fine_grained_label')
            d_feat = batch.get('dynamic_features')
            if x is None or y is None or d_feat is None:
                logger.error("Data loader returned batch dictionary missing 'data', 'fine_grained_label', or 'dynamic_features'")
                return
            if hasattr(configs, 'dynamic_feature_dim') and d_feat.shape[-1] != configs.dynamic_feature_dim:
                logger.warning(f'Loaded dynamic feature dimension ({d_feat.shape[-1]}) does not match config ({configs.dynamic_feature_dim})!')
        else:
            logger.error(f'Data loader did not return the expected dictionary format (got {type(batch)})')
            return
        model = TSTModel(configs).to(device)
        criterion_stage1 = None
        criterion_stage2 = None
        if configs.use_two_stage_classifier:
            weight_s1_arg = None
            if configs.stage1_loss_type.lower() in ['crossentropy', 'focalloss'] and configs.stage1_pos_weight_strategy == 'inverse_frequency' and (pos_weight_s1_calculated is not None):
                weight_s1_arg = torch.tensor([1.0, pos_weight_s1_calculated.item()], device=device)
            elif configs.stage1_loss_type.lower() == 'bcewithlogits' and configs.stage1_pos_weight_strategy == 'inverse_frequency' and (pos_weight_s1_calculated is not None):
                weight_s1_arg = pos_weight_s1_calculated
            criterion_stage1 = get_loss_criterion(loss_type=configs.stage1_loss_type, device=device, weight=weight_s1_arg if configs.stage1_loss_type.lower() in ['crossentropy', 'focalloss'] else None, pos_weight=weight_s1_arg if configs.stage1_loss_type.lower() == 'bcewithlogits' else None, num_classes=2, focal_alpha=configs.stage1_focal_alpha, focal_gamma=configs.stage1_focal_gamma, triplet_margin=configs.stage1_triplet_margin)
            weight_s2_arg = None
            if configs.stage2_pos_weight_strategy == 'inverse_frequency' and pos_weight_s2_calculated is not None:
                weight_s2_arg = pos_weight_s2_calculated
                if weight_s2_arg.shape[0] != configs.num_abnormal_classes:
                    logger.warning(f'Stage 2 pos_weight_s2 shape ({weight_s2_arg.shape[0]}) mismatch with num_abnormal_classes ({configs.num_abnormal_classes})!')
            elif configs.stage2_pos_weight_strategy == 'inverse_frequency':
                logger.warning("Stage 2 pos_weight strategy is 'inverse_frequency' but pos_weight_s2 is None!")
            if configs.stage2_loss_type.lower() == 'tripletloss':
                logger.error('TripletLoss is not supported for Stage 2. Please configure a different loss.')
                raise ValueError('Invalid configuration: TripletLoss cannot be used for Stage 2.')
            criterion_stage2 = get_loss_criterion(loss_type=configs.stage2_loss_type, device=device, weight=weight_s2_arg, pos_weight=weight_s2_arg if configs.stage2_loss_type.lower() == 'bcewithlogits' else None, num_classes=configs.num_abnormal_classes, focal_alpha=configs.stage2_focal_alpha, focal_gamma=configs.stage2_focal_gamma)
        else:
            if configs.stage1_loss_type.lower() == 'tripletloss':
                logger.error('TripletLoss is only supported for Stage 1 in two-stage classification mode. Please configure a different loss function for single-stage.')
                raise ValueError('Invalid loss configuration for single stage.')
            weight_single_arg = None
            if configs.stage1_pos_weight_strategy == 'inverse_frequency' and pos_weight_s1_calculated is not None:
                weight_single_arg = pos_weight_s1_calculated
            criterion_stage1 = get_loss_criterion(loss_type=configs.stage1_loss_type, device=device, weight=weight_single_arg, pos_weight=weight_single_arg if configs.stage1_loss_type.lower() == 'bcewithlogits' else None, num_classes=configs.num_class, focal_alpha=configs.stage1_focal_alpha, focal_gamma=configs.stage1_focal_gamma)
            criterion_stage2 = None
        if configs.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
        elif configs.optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=configs.learning_rate)
        elif configs.optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=configs.learning_rate)
        else:
            logger.warning(f'Unknown optimizer type: {configs.optimizer_type}, using Adam')
            optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
        scheduler = None
        if configs.scheduler_type == 'StepLR':
            scheduler = StepLR(optimizer, step_size=configs.scheduler_step_size, gamma=configs.scheduler_gamma)
        elif configs.scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=configs.scheduler_T_max, eta_min=configs.scheduler_eta_min)
        elif configs.scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=configs.scheduler_factor, patience=configs.scheduler_patience, min_lr=configs.scheduler_min_lr)
        elif configs.scheduler_type == 'None':
            pass
        else:
            logger.warning(f'Unknown scheduler type: {configs.scheduler_type}, not using a scheduler')
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        validation_criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(configs.num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            if configs.use_warmup and epoch < configs.warmup_epochs:
                lr_after_adjust = adjust_learning_rate(optimizer, epoch, current_lr, configs)
            else:
                lr_after_adjust = optimizer.param_groups[0]['lr']
                if epoch == configs.warmup_epochs and configs.use_warmup:
                    pass
            model.train()
            train_loss = 0.0
            train_acc_correct = 0
            train_total = 0
            train_acc_stage1_correct = 0
            train_acc_stage1_total = 0
            train_acc_stage2_correct = 0
            train_acc_stage2_total = 0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{configs.num_epochs} [Train] LR={lr_after_adjust:.1E}')
            for batch_idx, batch in enumerate(train_bar):
                try:
                    optimizer.zero_grad()
                    batch_results = _process_train_batch(model, batch, criterion_stage1, criterion_stage2, configs, device)
                    loss = batch_results['loss']
                    if loss is not None and loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                    else:
                        logger.warning(f'Batch {batch_idx + 1}: Skipping backward pass (loss is None or does not require grad).')
                        continue
                    train_loss += loss.item()
                    train_acc_correct += batch_results['correct_final']
                    train_total += batch_results['total_final']
                    if configs.use_two_stage_classifier:
                        train_acc_stage1_correct += batch_results['correct_s1']
                        train_acc_stage1_total += batch_results['total_s1']
                        train_acc_stage2_correct += batch_results['correct_s2']
                        train_acc_stage2_total += batch_results['total_s2']
                    current_avg_loss = train_loss / (batch_idx + 1) if batch_idx >= 0 else 0
                    current_acc = train_acc_correct / train_total if train_total > 0 else 0
                    postfix_dict = {'loss': f'{loss.item():.4f}', 'avg_loss': f'{current_avg_loss:.4f}', 'acc': f'{current_acc:.4f}'}
                    if configs.use_two_stage_classifier:
                        postfix_dict['acc_s1'] = f'{train_acc_stage1_correct / train_acc_stage1_total:.4f}' if train_acc_stage1_total > 0 else 'N/A'
                        postfix_dict['acc_s2'] = f'{train_acc_stage2_correct / train_acc_stage2_total:.4f}' if train_acc_stage2_total > 0 else 'N/A'
                    train_bar.set_postfix(postfix_dict)
                except Exception as e:
                    logger.error(f'Error processing batch {batch_idx + 1} in training loop: {e}', exc_info=True)
                    raise e
            avg_train_loss = train_loss / (batch_idx + 1) if batch_idx >= 0 else 0
            avg_train_acc = train_acc_correct / train_total if train_total > 0 else 0
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            model.eval()
            total_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []
            all_stage1_probs = []
            all_stage1_preds = []
            all_stage1_targets = []
            all_stage2_probs = []
            all_stage2_preds = []
            all_stage2_targets = []
            all_stage2_preds_abnormal_only = []
            all_stage2_targets_abnormal_only = []
            all_stage2_probs_abnormal_only = []
            all_batch_x = []
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{configs.num_epochs} [Val]')
                for batch_idx, batch in enumerate(val_bar):
                    try:
                        batch_results = _process_val_batch(model, batch, configs, device)
                        outputs_s1 = batch_results.get('outputs_s1')
                        batch_y = batch_results.get('batch_y_fine')
                        batch_s1_labels = batch_results.get('batch_s1_labels')
                        loss_val = torch.tensor(0.0, device=device)
                        current_batch_size = 0
                        if outputs_s1 is not None:
                            current_batch_size = outputs_s1.size(0)
                            try:
                                if configs.use_two_stage_classifier and batch_s1_labels is not None:
                                    target_labels_for_loss = torch.tensor(batch_s1_labels, dtype=torch.long, device=device)
                                    loss_val = validation_criterion(outputs_s1, target_labels_for_loss)
                                elif not configs.use_two_stage_classifier:
                                    target_labels_for_loss = torch.tensor(batch_y, dtype=torch.long, device=device)
                                    loss_val = validation_criterion(outputs_s1, target_labels_for_loss)
                            except Exception as val_loss_err:
                                logger.error(f'Error calculating validation loss: {val_loss_err}')
                                loss_val = torch.tensor(0.0, device=device)
                        else:
                            logger.warning('Validation loss cannot be calculated: missing stage1 outputs or target labels.')
                        total_loss += loss_val.item() * current_batch_size
                        all_preds.extend(batch_results['final_preds'])
                        all_labels.extend(batch_results['final_labels'])
                        all_probs.extend(batch_results['final_probs'])
                        if configs.use_two_stage_classifier:
                            all_stage1_preds.extend(batch_results['s1_preds'])
                            all_stage1_targets.extend(batch_results['s1_labels'])
                            all_stage1_probs.extend(batch_results['s1_probs'])
                            all_stage2_preds.extend(batch_results['s2_preds'])
                            all_stage2_targets.extend(batch_results['s2_labels'])
                            all_stage2_probs.extend(batch_results['s2_probs'])
                            all_stage2_preds_abnormal_only.extend(batch_results['s2_preds_abnormal_only'])
                            all_stage2_targets_abnormal_only.extend(batch_results['s2_labels_abnormal_only'])
                            all_stage2_probs_abnormal_only.extend(batch_results['s2_probs_abnormal_only'])
                        if batch_results['raw_data'] is not None:
                            all_batch_x.append(batch_results['raw_data'])
                    except Exception as e:
                        logger.error(f'Error processing validation batch {batch_idx + 1}: {e}', exc_info=True)
                        raise e
            avg_val_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
            all_preds_np = np.array(all_preds)
            all_labels_np = np.array(all_labels)
            if len(all_labels_np) > 0:
                val_acc_correct = np.sum(all_preds_np == all_labels_np)
                avg_val_acc = val_acc_correct / len(all_labels_np)
            else:
                avg_val_acc = 0.0
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)
            if scheduler is not None and (not configs.use_warmup or epoch >= configs.warmup_epochs):
                if configs.scheduler_type == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
                    if configs.scheduler_type != 'None':
                        pass
            if hasattr(configs, 'label_to_type'):
                for label, beat_type in configs.label_to_type.items():
                    pass
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                model_path = os.path.join(configs.checkpoints, 'best_model.pth')
                save_content = {'model_state_dict': model.state_dict(), 'configs_dict': configs.__dict__}
                torch.save(save_content, model_path)
            else:
                patience_counter += 1
            if hasattr(configs, 'save_epochs') and isinstance(configs.save_epochs, list) and (epoch + 1 in configs.save_epochs):
                epoch_model_path = os.path.join(configs.checkpoints, f'epoch_{epoch + 1}_model.pth')
                epoch_save_content = {'model_state_dict': model.state_dict(), 'configs_dict': configs.__dict__, 'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'train_acc': avg_train_acc, 'val_acc': avg_val_acc}
                torch.save(epoch_save_content, epoch_model_path)
            if patience_counter >= configs.patience:
                break
        last_epoch_model_path = os.path.join(configs.checkpoints, 'last_epoch_model.pth')
        last_epoch_save_content = {'model_state_dict': model.state_dict(), 'configs_dict': configs.__dict__}
        torch.save(last_epoch_save_content, last_epoch_model_path)
        best_model_path = os.path.join(configs.checkpoints, 'best_model.pth')
        test_criterion = nn.CrossEntropyLoss()
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            loaded_configs_dict = checkpoint.get('configs_dict')
            if loaded_configs_dict:
                configs_for_testing = Config()
                configs_for_testing.__dict__.update(loaded_configs_dict)
                if 'dynamic_feature_dim' in loaded_configs_dict:
                    configs.dynamic_feature_dim = loaded_configs_dict['dynamic_feature_dim']
            else:
                configs_for_testing = configs
            model = TSTModel(configs_for_testing).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            test(model, test_loader, test_criterion, device, configs_for_testing)
        else:
            last_epoch_model_path_for_test = os.path.join(configs.checkpoints, 'last_epoch_model.pth')
            if os.path.exists(last_epoch_model_path_for_test):
                checkpoint_last = torch.load(last_epoch_model_path_for_test, map_location=device, weights_only=False)
                loaded_configs_dict_last = checkpoint_last.get('configs_dict')
                if loaded_configs_dict_last:
                    configs_for_testing_last = Config()
                    configs_for_testing_last.__dict__.update(loaded_configs_dict_last)
                    if 'dynamic_feature_dim' in loaded_configs_dict_last:
                        configs.dynamic_feature_dim = loaded_configs_dict_last['dynamic_feature_dim']
                else:
                    configs_for_testing_last = configs
                model = TSTModel(configs_for_testing_last).to(device)
                model.load_state_dict(checkpoint_last['model_state_dict'])
                test(model, test_loader, test_criterion, device, configs_for_testing_last)
            else:
                logger.error('Neither best_model.pth nor last_epoch_model.pth found. Cannot perform testing.')
    except Exception as e:
        logger.error(f'Error occurred during training process: {str(e)}', exc_info=True)
        raise

def test(model, test_loader, criterion, device, configs):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_batch_x = []
    all_batch_y_fine = []
    all_dynamic_features = []
    all_stage1_probs = []
    all_stage1_preds = []
    all_stage1_targets = []
    all_stage2_probs = []
    all_stage2_preds = []
    all_stage2_targets = []
    all_stage2_preds_abnormal_only = []
    all_stage2_targets_abnormal_only = []
    all_stage2_probs_abnormal_only = []
    num_test_samples = 0

    def map_fine_to_disease(fine_label_idx, label_to_type, disease_mapping, disease_categories):
        fine_symbol = label_to_type.get(fine_label_idx)
        if fine_symbol is None:
            return disease_categories.get('Other', -1)
        for disease_name, symbols_list in disease_mapping.items():
            if fine_symbol in symbols_list:
                return disease_categories.get(disease_name, disease_categories.get('Other', -1))
        return disease_categories.get('Other', -1)
    with torch.no_grad():
        is_two_stage = configs.use_two_stage_classifier
        normal_label_index = getattr(configs, 'normal_label_index', None)
        stage2_map_inv = getattr(configs, 'stage2_to_abnormal_fine_map', None)
        abnormal_fine_to_stage2_map = getattr(configs, 'abnormal_fine_to_stage2_map', None)
        evaluation_mode = getattr(configs, 'sample_selection_mode', 'fine-grained')
        label_to_type_map = getattr(configs, 'label_to_type', {})
        disease_mapping = getattr(configs, 'disease_mapping', {})
        disease_categories = getattr(configs, 'disease_categories', {})
        disease_category_names = getattr(configs, 'disease_category_names', {})
        if is_two_stage and (normal_label_index is None or stage2_map_inv is None or abnormal_fine_to_stage2_map is None):
            logger.error('Testing cannot get required mappings for two-stage mode from loaded configs!')
            return (float('nan'), float('nan'))
        if evaluation_mode == 'by_disease' and (not disease_mapping or not disease_categories or (not disease_category_names)):
            logger.error("Testing in 'by_disease' mode but disease mapping/categories/names missing from loaded configs!")
            return (float('nan'), float('nan'))
        if not label_to_type_map:
            logger.error(f'Testing requires label_to_type mapping from loaded configs!')
            return (float('nan'), float('nan'))
        test_bar = tqdm(test_loader, desc='Testing')
        for batch_idx, batch in enumerate(test_bar):
            try:
                batch_results = _process_val_batch(model, batch, configs, device)
                outputs_s1 = batch_results.get('outputs_s1')
                batch_y_fine_np = batch_results.get('batch_y_fine')
                batch_s1_labels_np = batch_results.get('batch_s1_labels')
                final_preds_fine_np = np.array(batch_results['final_preds'])
                if outputs_s1 is None or batch_y_fine_np is None:
                    logger.warning(f'Skipping test batch {batch_idx + 1} due to missing model output or fine-grained labels.')
                    continue
                current_batch_size = len(batch_y_fine_np)
                num_test_samples += current_batch_size
                batch_y_eval_np = batch_y_fine_np
                final_preds_eval_np = final_preds_fine_np
                if evaluation_mode == 'by_disease':
                    if 'disease_label' in batch:
                        batch_y_eval_np = batch['disease_label'].numpy()
                    else:
                        batch_y_eval_np = np.array([map_fine_to_disease(fine_idx, label_to_type_map, disease_mapping, disease_categories) for fine_idx in batch_y_fine_np])
                    final_preds_eval_np = np.array([map_fine_to_disease(fine_pred_idx, label_to_type_map, disease_mapping, disease_categories) for fine_pred_idx in final_preds_fine_np])
                loss_test = torch.tensor(0.0, device=device)
                try:
                    if criterion is not None:
                        target_labels_for_loss = torch.tensor(batch_y_eval_np, dtype=torch.long, device=device)
                        loss_input_logits = None
                        if is_two_stage and batch_s1_labels_np is not None:
                            loss_input_logits = outputs_s1
                            target_labels_for_loss = torch.tensor(batch_s1_labels_np, dtype=torch.long, device=device)
                        elif not is_two_stage:
                            loss_input_logits = outputs_s1
                            target_labels_for_loss = torch.tensor(batch_y_eval_np, dtype=torch.long, device=device)
                        if loss_input_logits is not None:
                            loss_test = criterion(loss_input_logits, target_labels_for_loss)
                        else:
                            logger.warning('Test loss skipped: Cannot determine appropriate model output for loss.')
                    else:
                        logger.warning('Test loss skipped: Criterion is None.')
                except Exception as test_loss_err:
                    logger.error(f'Error calculating test loss: {test_loss_err}')
                    loss_test = torch.tensor(0.0, device=device)
                total_loss += loss_test.item() * current_batch_size
                all_labels.extend(batch_y_eval_np)
                all_preds.extend(final_preds_eval_np)
                all_probs.extend(batch_results['final_probs'])
                all_batch_y_fine.extend(batch_y_fine_np)
                if is_two_stage:
                    all_stage1_preds.extend(batch_results['s1_preds'])
                    all_stage1_targets.extend(batch_results['s1_labels'])
                    all_stage1_probs.extend(batch_results['s1_probs'])
                    all_stage2_preds.extend(batch_results['s2_preds'])
                    all_stage2_targets.extend(batch_results['s2_labels'])
                    all_stage2_probs.extend(batch_results['s2_probs'])
                    all_stage2_preds_abnormal_only.extend(batch_results['s2_preds_abnormal_only'])
                    all_stage2_targets_abnormal_only.extend(batch_results['s2_labels_abnormal_only'])
                    all_stage2_probs_abnormal_only.extend(batch_results['s2_probs_abnormal_only'])
                if batch_results['raw_data'] is not None:
                    all_batch_x.append(batch_results['raw_data'])
                if hasattr(configs, 'save_test_results') and configs.save_test_results and ('dynamic_features' in batch):
                    all_dynamic_features.append(batch['dynamic_features'].cpu().numpy())
            except Exception as e:
                logger.error(f'Error processing test batch {batch_idx + 1}: {e}', exc_info=True)
                raise e
        avg_test_loss = total_loss / num_test_samples if num_test_samples > 0 else 0.0
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        if len(all_labels_np) == 0:
            logger.error('No test labels collected, cannot perform evaluation.')
            return (float('nan'), float('nan'))
        correct = np.sum(all_preds_np == all_labels_np)
        test_acc = 100.0 * correct / len(all_labels_np)
        unique_eval_labels = sorted(list(set(all_labels_np) | set(all_preds_np)))
        report_labels = [int(idx) for idx in unique_eval_labels if idx >= 0]
        eval_label_names_map = {}
        num_eval_classes = 0
        if evaluation_mode == 'by_disease':
            eval_label_names_map = {v: k for k, v in disease_categories.items()}
            num_eval_classes = len(disease_categories)
        else:
            eval_label_names_map = label_to_type_map
            num_eval_classes = getattr(configs, 'num_class', 0)
        if num_eval_classes == 0 or not eval_label_names_map:
            logger.error(f"Cannot get label names or class count for evaluation mode '{evaluation_mode}'!")
            report_target_names = [str(idx) for idx in report_labels]
        else:
            report_target_names = [eval_label_names_map.get(idx, f'Unknown{idx}') for idx in report_labels]
        if not report_labels or len(all_labels_np) == 0 or len(all_preds_np) == 0:
            logger.warning('Not enough valid labels or predictions collected to generate report.')
        else:
            report = classification_report(all_labels_np, all_preds_np, labels=report_labels, target_names=report_target_names, zero_division=0)
            logger.info(f'\nTest Classification Report ({evaluation_mode}):\n{report}')
            conf_matrix = confusion_matrix(all_labels_np, all_preds_np, labels=report_labels)
            logger.info(f'\nTest Confusion Matrix ({evaluation_mode}):\n{conf_matrix}')
            precision = precision_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            recall = recall_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            for i, label_idx in enumerate(report_labels):
                label_name = report_target_names[i]
                if i < len(precision):
                    logger.info(f'{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}')
        all_probs_np = np.array(all_probs) if all_probs else np.array([])
        confidences = all_probs_np
        correct_confidences = np.array([])
        incorrect_confidences = np.array([])
        if len(all_preds_np) == len(all_labels_np) and len(confidences) == len(all_labels_np):
            correct_indices = all_preds_np == all_labels_np
            correct_confidences = confidences[correct_indices]
            incorrect_confidences = confidences[~correct_indices]
        else:
            logger.warning(f'Length mismatch prevents confidence analysis: preds={len(all_preds_np)}, labels={len(all_labels_np)}, confidences={len(confidences)}')
        if len(all_labels_np) > 0 and len(confidences) == len(all_labels_np):
            thresholds, accuracies, rejection_rates, retention_correct, recommended_threshold = analyze_confidence_thresholds(all_preds_np, all_labels_np, confidences, configs=configs)
            logger.info(f'Recommended confidence threshold: {recommended_threshold:.4f}')
            high_conf_mask = confidences >= recommended_threshold
            high_conf_preds = all_preds_np[high_conf_mask]
            high_conf_labels = all_labels_np[high_conf_mask]
            if len(high_conf_labels) > 0:
                high_conf_accuracy = np.mean(high_conf_preds == high_conf_labels)
                rejection_rate = 1.0 - np.mean(high_conf_mask)
                correct_mask = all_preds_np == all_labels_np
                correct_and_retained = np.logical_and(correct_mask, high_conf_mask)
                retention_rate = np.sum(correct_and_retained) / np.sum(correct_mask) if np.sum(correct_mask) > 0 else 0
                logger.info(f'High confidence predictions: Accuracy={high_conf_accuracy:.4f}, Rejection rate={rejection_rate:.4f}, Retention rate={retention_rate:.4f}')
                if len(set(high_conf_labels)) > 1 and len(high_conf_preds) > 0:
                    high_conf_report = classification_report(high_conf_labels, high_conf_preds, labels=report_labels, target_names=report_target_names, zero_division=0)
                    logger.info(f'\nHigh Confidence Classification Report:\n{high_conf_report}')
        else:
            logger.warning('Cannot perform rejection analysis: Incomplete label or confidence data.')
        results_filename = os.path.join(configs.checkpoints, 'test_data_and_results.npz')
        data_to_save = {'x_test': np.concatenate(all_batch_x, axis=0) if all_batch_x else np.array([]), 'y_test_fine': np.array(all_batch_y_fine) if all_batch_y_fine else np.array([]), 'dynamic_features_test': np.concatenate(all_dynamic_features, axis=0) if all_dynamic_features else np.array([]), 'preds_eval_labels': all_preds_np, 'trues_eval_labels': all_labels_np, 'preds_confidence': all_probs_np, 'evaluation_mode': evaluation_mode, 'label_to_type': label_to_type_map}
        if evaluation_mode == 'by_disease':
            data_to_save['disease_category_names'] = disease_category_names
            data_to_save['disease_categories'] = disease_categories
        if is_two_stage:
            data_to_save.update({'stage1_probs': np.array(all_stage1_probs) if all_stage1_probs else np.array([]), 'stage1_preds': np.array(all_stage1_preds) if all_stage1_preds else np.array([]), 'stage1_targets': np.array(all_stage1_targets) if all_stage1_targets else np.array([]), 'stage2_probs': np.array(all_stage2_probs) if all_stage2_probs else np.array([]), 'stage2_preds': np.array(all_stage2_preds) if all_stage2_preds else np.array([]), 'stage2_targets': np.array(all_stage2_targets) if all_stage2_targets else np.array([]), 'stage2_preds_abnormal_only': np.array(all_stage2_preds_abnormal_only) if all_stage2_preds_abnormal_only else np.array([]), 'stage2_targets_abnormal_only': np.array(all_stage2_targets_abnormal_only) if all_stage2_targets_abnormal_only else np.array([]), 'stage2_probs_abnormal_only': np.array(all_stage2_probs_abnormal_only) if all_stage2_probs_abnormal_only else np.array([]), 'normal_label_index': normal_label_index, 'abnormal_fine_to_stage2_map': abnormal_fine_to_stage2_map, 'stage2_to_abnormal_fine_map': stage2_map_inv})
        if len(all_labels_np) > 0 and len(confidences) == len(all_labels_np):
            data_to_save.update({'confidence_thresholds': thresholds, 'accuracy_at_thresholds': accuracies, 'rejection_rates': rejection_rates, 'retention_correct_rates': retention_correct, 'recommended_threshold': recommended_threshold})
        np.savez(results_filename, **data_to_save)
        logger.info(f'Test results saved to {results_filename}')
        logger.info(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        return (avg_test_loss, test_acc)
if __name__ == '__main__':
    seed_everything(3)
    train()
