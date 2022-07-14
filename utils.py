import torch
import numpy as np
from glob import glob
import os
import cv2
import monai
import segmentation_models_pytorch as smp
from loss_functions import HausdorffDTLoss, CenterOfMassLoss
from metrics import BinaryMetrics


def get_loss_function(class_weights=None, key='dice', dist_matrix=None, weighting_mode='GDL'): 
    if dist_matrix is None:
        dist_matrix = 1 - np.eye(len(class_weights))
    losses_dict = {
        'dice': monai.losses.DiceLoss(softmax=False),
        'ce': torch.nn.CrossEntropyLoss(weight=class_weights),
        'dice_ce': monai.losses.DiceCELoss(softmax=False, ce_weight=class_weights),
        'focal': monai.losses.FocalLoss(weight=class_weights),
        'tversky': monai.losses.TverskyLoss(softmax=False),
        'gd': monai.losses.GeneralizedDiceLoss(softmax=False),
        'dice_focal': monai.losses.DiceFocalLoss(softmax=False, focal_weight=class_weights),
        'iou': monai.losses.DiceLoss(softmax=False, jaccard=True),
        'gwd': monai.losses.GeneralizedWassersteinDiceLoss(dist_matrix=dist_matrix, weighting_mode=weighting_mode)
    }
    loss_fn = losses_dict[key] if key in losses_dict else losses_dict['ce']
    return loss_fn


def get_wasserstein_distance_matrix(n_classes):
    mat = np.ones((n_classes,n_classes))
    mat[1:,1:] = 0.5
    mat = mat*(1-np.eye(n_classes))
    return mat


def get_list_of_image_and_mask(img_dir, mask_dir):
    images = os.listdir(img_dir)
    masks = os.listdir(mask_dir)
    common = list(set(images).intersection(masks))
    images = [os.path.join(img_dir, l) for l in common]
    masks = [os.path.join(mask_dir, l) for l in common]
    
    return images, masks


def get_class_weights(mask_dir, n_classes):
    class_weights = list()
    masks = sorted(glob(os.path.join(mask_dir, "*.png")))
    img_dict = dict()
    for i in range(n_classes):
        img_dict[i] = list()
    for img in masks:
        mask = cv2.imread(img,0)
        items, count = np.unique(mask, return_counts=True)
        mask_dict = {i:np.sum(count)/c for (i,c) in zip(items, count)}
        for i in range(n_classes):
            if i in mask_dict:
                img_dict[i].append(mask_dict[i])
    for i in range(n_classes):
        class_weights.append(np.mean(img_dict[i]))
    class_weights = torch.tensor(class_weights)/torch.sum(torch.tensor(class_weights))
    return class_weights.float()


def get_loss_and_metrics(n_classes, prefix, out, out1, mask, mask1):
    wasserstein_distance_matrix = get_wasserstein_distance_matrix(n_classes)
    d_loss = monai.losses.DiceLoss(softmax=False)
    ce_loss = torch.nn.CrossEntropyLoss(weight=None)
    df_loss = monai.losses.DiceFocalLoss(softmax=False, focal_weight=None)
    d_ce_loss = monai.losses.DiceCELoss(softmax=False, ce_weight=None)
    gd_loss = monai.losses.GeneralizedDiceLoss(softmax=False)
    gwd_loss = monai.losses.GeneralizedWassersteinDiceLoss(dist_matrix=wasserstein_distance_matrix, weighting_mode='GDL')
    f_loss = monai.losses.FocalLoss(weight=None)
    t_loss = monai.losses.TverskyLoss(softmax=False)
    iou_loss = monai.losses.DiceLoss(softmax=False, jaccard=True)
    
    logs = {
        #prefix+'_ce_loss': torch.nn.CrossEntropyLoss(weight=None)(out, mask), 
        prefix+'_dice_loss': monai.losses.DiceLoss(softmax=False)(out, mask1), 
        #prefix+'_dice_focal_loss': monai.losses.DiceFocalLoss(softmax=False, focal_weight=None)(out, mask1),
        #prefix+'_dice_ce_loss': monai.losses.DiceCELoss(softmax=False, ce_weight=None)(out, mask1),
        prefix+'_iou_loss': monai.losses.DiceLoss(softmax=False, jaccard=True)(out, mask1), 
        #prefix+'_gd_loss': gd_loss(out, mask1),
        #prefix+'_gwd_loss': gwd_loss(out, mask),
        #prefix+'_focal_loss': monai.losses.FocalLoss(weight=None)(out, mask1),
        #prefix+'_tversky_loss': monai.losses.TverskyLoss(softmax=False)(out, mask1),
        prefix+'_hd_loss': HausdorffDTLoss(include_background=True)(out1,mask1),
        prefix+'_ed_loss': CenterOfMassLoss(include_background=True)(out1,mask1),
        #prefix+'_accuracy': smp.utils.metrics.Accuracy()(out1, mask1),
        #prefix+'_recall': smp.utils.metrics.Recall()(out1, mask1),
        prefix+'_f1_score': smp.utils.metrics.Fscore()(out1, mask1),
        prefix+'_iou': smp.utils.metrics.IoU()(out1, mask1),
    }
    for i in range(n_classes):
        mask_tensor = torch.unsqueeze(mask1[:,i,...], 1).float()
        out_tensor = torch.unsqueeze(out[:,i,...], 1).float()
        out1_tensor = torch.unsqueeze(out1[:,i,...], 1).float()
        #cm = monai.metrics.get_confusion_matrix(out1, mask1)
        #print(out1_tensor.shape, mask_tensor.shape)
        #bs, ch = out1_tensor.shape[:2]
        acc, dice, precision, specificity, recall, count_ratio  = BinaryMetrics(activation=None)(torch.squeeze(mask_tensor), out1_tensor)
        #pred, tar = out1_tensor.view(bs, ch, -1), mask_tensor.view(bs, ch, -1).long()
        #pred, tar = out1_tensor, mask_tensor.long()
        logs[prefix+'_dice_loss_object_'+str(i)] = d_loss(out_tensor,mask_tensor)
        #logs[prefix+'_df_loss_object_'+str(i)] = df_loss(out_tensor,mask_tensor)
        #logs[prefix+'_d_ce_loss_object_'+str(i)] = d_ce_loss(out_tensor,mask_tensor)
        logs[prefix+'_iou_loss_object_'+str(i)] = iou_loss(out_tensor,mask_tensor)
        #logs[prefix+'_gd_loss_object_'+str(i)] = gd_loss(out_tensor,mask_tensor)
        #logs[prefix+'_f_loss_object_'+str(i)] = f_loss(out_tensor,mask_tensor)
        logs[prefix+'_hd_loss_object_'+str(i)] = HausdorffDTLoss(include_background=True)(out1_tensor,mask_tensor)
        logs[prefix+'_ed_loss_object_'+str(i)] = CenterOfMassLoss(include_background=True)(out1_tensor,mask_tensor)
        #logs[prefix+'_accuracy_object_'+str(i)] = smp.utils.metrics.Accuracy()(out1_tensor, mask_tensor)
        #logs[prefix+'_recall_object_'+str(i)] = smp.utils.metrics.Recall()(out1_tensor, mask_tensor)
        logs[prefix+'_f1_score_object_'+str(i)] = smp.utils.metrics.Fscore()(out1_tensor, mask_tensor)
        logs[prefix+'_iou_object_'+str(i)] = smp.utils.metrics.IoU()(out1_tensor, mask_tensor)
        #logs[prefix+'_sensitivity_object_'+str(i)] = monai.metrics.compute_confusion_matrix_metric('sensitivity', cm)#[1]
        #logs[prefix+'_specificity_object_'+str(i)] = monai.metrics.compute_confusion_matrix_metric('specificity', cm)#[1]
        #logs[prefix+'_precision_object_'+str(i)] = monai.metrics.compute_confusion_matrix_metric('precision', cm)#[1]
        #logs[prefix+'_f1_score_cm_object_'+str(i)] = monai.metrics.compute_confusion_matrix_metric('f1 score', cm)#[1]
        #logs[prefix+'_sensitivity_object_'+str(i)] = torchmetrics.compute_confusion_matrix_metric('sensitivity', cm)#[1]
        logs[prefix+'_specificity_object_'+str(i)] = specificity
        logs[prefix+'_recall_object_'+str(i)] = recall
        logs[prefix+'_precision_object_'+str(i)] = precision
        logs[prefix+'_dice_object_'+str(i)] = dice
        logs[prefix+'_accuracy_object_'+str(i)] = acc
        logs[prefix+'_count_ratio_object_'+str(i)] = count_ratio
        #logs[prefix+'_f1_score_cm_object_'+str(i)] = torchmetrics.compute_confusion_matrix_metric('f1 score', cm)#[1]
        
    return logs
