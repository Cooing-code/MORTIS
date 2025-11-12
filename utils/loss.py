import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from collections import Counter
logger = logging.getLogger(__name__)

def hard_triplet_miner(embeddings, labels):
    dist_mat = torch.cdist(embeddings, embeddings, p=2)
    N = labels.size(0)
    is_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
    is_neg = ~is_pos
    is_pos.fill_diagonal_(False)
    dist_pos = dist_mat.clone()
    dist_pos[~is_pos] = -1000000000.0
    hardest_positive_dist, hardest_positive_indices = torch.max(dist_pos, dim=1)
    dist_neg = dist_mat + is_pos * 1000000000.0
    dist_neg.fill_diagonal_(1000000000.0)
    hardest_negative_dist, hardest_negative_indices = torch.min(dist_neg, dim=1)
    valid_anchors_mask = (torch.sum(is_pos, dim=1) > 0) & (torch.sum(is_neg, dim=1) > 0)
    anchor_indices = torch.where(valid_anchors_mask)[0]
    if anchor_indices.numel() == 0:
        return ([], [], [])
    positive_indices = hardest_positive_indices[anchor_indices]
    negative_indices = hardest_negative_indices[anchor_indices]
    valid_triplet_mask = (anchor_indices != positive_indices) & (anchor_indices != negative_indices) & (positive_indices != negative_indices)
    if not valid_triplet_mask.all():
        anchor_indices = anchor_indices[valid_triplet_mask]
        positive_indices = positive_indices[valid_triplet_mask]
        negative_indices = negative_indices[valid_triplet_mask]
    anchor_list = anchor_indices.tolist()
    positive_list = positive_indices.tolist()
    negative_list = negative_indices.tolist()
    return (anchor_list, positive_list, negative_list)

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=None, device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.device = device
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                if num_classes == 2:
                    self.alpha = torch.tensor([1 - alpha, alpha], device=device)
                elif num_classes is not None and num_classes > 0:
                    self.alpha = torch.full((num_classes,), alpha, device=device)
                    pass
                else:
                    self.alpha = None
                    pass
            elif isinstance(alpha, (list, tuple)):
                assert len(alpha) == num_classes, f'Alpha list len ({len(alpha)}) != num_classes ({num_classes})'
                self.alpha = torch.tensor(alpha, device=device)
            elif isinstance(alpha, torch.Tensor):
                assert alpha.shape[0] == num_classes, f'Alpha tensor size ({alpha.shape[0]}) != num_classes ({num_classes})'
                self.alpha = alpha.to(device)
            else:
                raise TypeError('Alpha must be float, list, tuple, or Tensor.')
        else:
            self.alpha = None

    def forward(self, logits, targets):
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        epsilon = 1e-09
        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        focal_factor = (1 - pt.clamp(max=1.0 - epsilon)) ** self.gamma
        loss = focal_factor * nll_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Invalid reduction type: {self.reduction}')

def get_loss_criterion(loss_type: str, device: str, weight=None, pos_weight=None, num_classes: int=None, focal_alpha=None, focal_gamma: float=2.0, triplet_margin: float=1.0) -> nn.Module:
    loss_type = loss_type.lower().replace('loss', '')
    if weight is not None:
        weight = weight.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    if loss_type == 'crossentropy':
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif loss_type == 'bcewithlogits':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'focal':
        effective_alpha = focal_alpha if focal_alpha is not None else weight
        criterion = FocalLoss(alpha=effective_alpha, gamma=focal_gamma, num_classes=num_classes, device=device)
    elif loss_type == 'triplet':
        criterion = nn.TripletMarginLoss(margin=triplet_margin, reduction='mean')
    else:
        raise ValueError(f"Unsupported loss type: '{loss_type}'. Supported: 'CrossEntropy', 'BCEWithLogits', 'FocalLoss', 'TripletLoss'")
    return criterion

class BaseLossCalculator:

    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device

    def calculate(self, model_outputs, batch_labels):
        raise NotImplementedError

class SingleStageLossCalculator(BaseLossCalculator):

    def __init__(self, criterion, configs):
        super().__init__(configs)
        self.criterion = criterion
        self.loss_type = configs.stage1_loss_type.lower()

    def calculate(self, model_outputs, batch_labels):
        if isinstance(model_outputs, (tuple, list)):
            logits = model_outputs[0]
        else:
            logits = model_outputs
        targets = batch_labels.long().to(self.device)
        loss = torch.tensor(0.0, device=self.device)
        if self.loss_type == 'bcewithlogits':
            y_one_hot = F.one_hot(targets, num_classes=self.configs.num_class).float()
            loss = self.criterion(logits, y_one_hot)
        elif self.loss_type in ['crossentropy', 'focal']:
            loss = self.criterion(logits, targets)
        else:
            pass
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return {'total_loss': loss}

class TwoStageLossCalculator(BaseLossCalculator):

    def __init__(self, criterion_stage1, criterion_stage2, configs):
        super().__init__(configs)
        self.criterion_stage1 = criterion_stage1
        self.criterion_stage2 = criterion_stage2
        self.stage1_loss_type = configs.stage1_loss_type.lower()
        self.stage2_loss_type = configs.stage2_loss_type.lower()
        self.stage1_weight = configs.stage1_loss_weight
        self.stage2_weight = configs.stage2_loss_weight
        self.num_abnormal_classes = configs.num_abnormal_classes
        if self.stage1_loss_type == 'triplet':
            self.triplet_miner = hard_triplet_miner
        elif self.stage1_loss_type not in ['crossentropy', 'bcewithlogits', 'focal']:
            pass
        if self.stage2_loss_type not in ['crossentropy', 'bcewithlogits', 'focal']:
            pass

    def calculate(self, model_outputs, batch_labels):
        outputs_stage1, outputs_stage2 = (model_outputs[0], model_outputs[1])
        features = model_outputs[2] if len(model_outputs) > 2 else None
        stage1_labels, stage2_labels = (batch_labels[0].long().to(self.device), batch_labels[1].long().to(self.device))
        loss_stage1 = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_stage2 = torch.tensor(0.0, device=self.device)
        if self.stage1_loss_type == 'triplet':
            if features is None:
                raise ValueError('Features required for TripletLoss in Stage 1, but not provided by model.')
            anchor_idx, pos_idx, neg_idx = self.triplet_miner(features, stage1_labels)
            if len(anchor_idx) > 0:
                loss_stage1 = self.criterion_stage1(features[anchor_idx], features[pos_idx], features[neg_idx])
            else:
                loss_stage1 = torch.tensor(0.0, device=self.device, requires_grad=True)
        elif self.stage1_loss_type == 'bcewithlogits':
            y_one_hot_s1 = F.one_hot(stage1_labels, num_classes=2).float()
            loss_stage1 = self.criterion_stage1(outputs_stage1, y_one_hot_s1)
        elif self.stage1_loss_type in ['crossentropy', 'focal']:
            loss_stage1 = self.criterion_stage1(outputs_stage1, stage1_labels)
        else:
            pass
        abnormal_mask = stage1_labels == 1
        if abnormal_mask.sum() > 0 and self.criterion_stage2 is not None:
            abnormal_outputs_stage2 = outputs_stage2[abnormal_mask]
            abnormal_stage2_labels = stage2_labels[abnormal_mask]
            valid_stage2_mask = abnormal_stage2_labels != -100
            if valid_stage2_mask.sum() > 0:
                abnormal_outputs_stage2 = abnormal_outputs_stage2[valid_stage2_mask]
                abnormal_stage2_labels = abnormal_stage2_labels[valid_stage2_mask]
                if abnormal_outputs_stage2.shape[0] > 0:
                    if self.stage2_loss_type == 'bcewithlogits':
                        if self.num_abnormal_classes is None or self.num_abnormal_classes <= 0:
                            raise ValueError('num_abnormal_classes must be set in configs for Stage 2 BCE.')
                        y_one_hot_s2 = F.one_hot(abnormal_stage2_labels, num_classes=self.num_abnormal_classes).float()
                        loss_stage2 = self.criterion_stage2(abnormal_outputs_stage2, y_one_hot_s2)
                    elif self.stage2_loss_type in ['crossentropy', 'focal']:
                        loss_stage2 = self.criterion_stage2(abnormal_outputs_stage2, abnormal_stage2_labels)
                    else:
                        pass
        if not isinstance(loss_stage1, torch.Tensor):
            loss_stage1 = torch.tensor(loss_stage1, device=self.device)
        if not isinstance(loss_stage2, torch.Tensor):
            loss_stage2 = torch.tensor(loss_stage2, device=self.device)
        total_loss = self.stage1_weight * loss_stage1 + self.stage2_weight * loss_stage2
        return {'total_loss': total_loss, 'loss_stage1': loss_stage1.detach(), 'loss_stage2': loss_stage2.detach()}
