import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import random
import time
from tqdm import tqdm
from models.model import TSTModel
from data.data_factory import get_data_loaders
from utils.loss import get_loss_criterion
from utils.metrics import metric
from configs import Config
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)

def seed_everything(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _process_val_batch(model, batch, configs, device):
    results = {'outputs_s1': None, 'batch_y_fine': None, 'batch_s1_labels': None, 'final_preds': [], 'final_labels': [], 'final_probs': [], 's1_preds': [], 's1_labels': [], 's1_probs': [], 's2_preds': [], 's2_labels': [], 's2_probs': [], 's2_preds_abnormal_only': [], 's2_labels_abnormal_only': [], 's2_probs_abnormal_only': [], 'raw_data': None}
    is_two_stage = configs.use_two_stage_classifier
    if not isinstance(batch, dict):
        pass
        return results
    required_keys = {'data', 'dynamic_features', 'fine_grained_label'}
    if is_two_stage:
        required_keys.update({'stage1_label', 'stage2_label'})
    if not required_keys.issubset(batch.keys()):
        pass
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
            pass
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
            pass
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

def map_fine_to_disease(fine_label_idx, label_to_type, disease_mapping, disease_categories):
    fine_symbol = label_to_type.get(fine_label_idx)
    if fine_symbol is None:
        return disease_categories.get('Other', -1)
    for disease_name, symbols_list in disease_mapping.items():
        if fine_symbol in symbols_list:
            return disease_categories.get(disease_name, disease_categories.get('Other', -1))
    return disease_categories.get('Other', -1)

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
            pass
            return (float('nan'), float('nan'))
        if evaluation_mode == 'by_disease' and (not disease_mapping or not disease_categories or (not disease_category_names)):
            pass
            return (float('nan'), float('nan'))
        if not label_to_type_map:
            pass
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
                    pass
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
                            pass
                    else:
                        pass
                except Exception as test_loss_err:
                    pass
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
                pass
                raise e
        avg_test_loss = total_loss / num_test_samples if num_test_samples > 0 else 0.0
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        if len(all_labels_np) == 0:
            pass
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
            pass
            report_target_names = [str(idx) for idx in report_labels]
        else:
            report_target_names = [eval_label_names_map.get(idx, f'Unknown{idx}') for idx in report_labels]
        if not report_labels or len(all_labels_np) == 0 or len(all_preds_np) == 0:
            pass
        else:
            report = classification_report(all_labels_np, all_preds_np, labels=report_labels, target_names=report_target_names, zero_division=0)
            conf_matrix = confusion_matrix(all_labels_np, all_preds_np, labels=report_labels)
            precision = precision_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            recall = recall_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average=None, labels=report_labels, zero_division=0)
            for i, label_idx in enumerate(report_labels):
                label_name = report_target_names[i]
                if i < len(precision):
                    pass
        all_probs_np = np.array(all_probs) if all_probs else np.array([])
        confidences = all_probs_np
        correct_confidences = np.array([])
        incorrect_confidences = np.array([])
        if len(all_preds_np) == len(all_labels_np) and len(confidences) == len(all_labels_np):
            correct_indices = all_preds_np == all_labels_np
            correct_confidences = confidences[correct_indices]
            incorrect_confidences = confidences[~correct_indices]
        else:
            pass
        if len(all_labels_np) > 0 and len(confidences) == len(all_labels_np):
            thresholds, accuracies, rejection_rates, retention_correct, recommended_threshold = analyze_confidence_thresholds(all_preds_np, all_labels_np, confidences, configs=configs)
            high_conf_mask = confidences >= recommended_threshold
            high_conf_preds = all_preds_np[high_conf_mask]
            high_conf_labels = all_labels_np[high_conf_mask]
            if len(high_conf_labels) > 0:
                high_conf_accuracy = np.mean(high_conf_preds == high_conf_labels)
                rejection_rate = 1.0 - np.mean(high_conf_mask)
                correct_mask = all_preds_np == all_labels_np
                correct_and_retained = np.logical_and(correct_mask, high_conf_mask)
                retention_rate = np.sum(correct_and_retained) / np.sum(correct_mask) if np.sum(correct_mask) > 0 else 0
                if len(set(high_conf_labels)) > 1 and len(high_conf_preds) > 0:
                    high_conf_report = classification_report(high_conf_labels, high_conf_preds, labels=report_labels, target_names=report_target_names, zero_division=0)
        else:
            pass
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
        return (avg_test_loss, test_acc)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TST-FAN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--config_seed', type=int, default=3, help='Seed used for data splitting (should match training)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for testing (e.g., cuda:0 or cpu)')
    args = parser.parse_args()
    seed_everything(args.config_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.checkpoint):
        pass
        exit(1)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    loaded_configs_dict = checkpoint.get('configs_dict')
    if loaded_configs_dict:
        configs = Config()
        original_seed = configs.seed
        original_device = configs.device
        configs.__dict__.update(loaded_configs_dict)
        configs.seed = args.config_seed
        configs.device = args.device
        os.makedirs(configs.checkpoints, exist_ok=True)
        
    else:
        pass
        exit(1)
    model = TSTModel(configs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        (_, _, test_loader), _, _ = get_data_loaders(configs)
    except Exception as e:
        pass
        exit(1)
    test_criterion = nn.CrossEntropyLoss().to(device)
    try:
        test_loss, test_acc = test(model, test_loader, test_criterion, device, configs)
    except Exception as e:
        pass
        exit(1)
