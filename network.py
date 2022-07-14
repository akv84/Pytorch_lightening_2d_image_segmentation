import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torchmetrics

from utils import get_wasserstein_distance_matrix, get_loss_function, get_loss_and_metrics


class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''
    encoders = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'efficientnet-b0', 'efficientnet-b1',
                'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 
                'efficientnet-b7', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11',
                'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

    def __init__(self, 
                 n_channels=3, 
                 n_classes=12, 
                 pretrained=0, 
                 encoder=13, 
                 lr=0.0001, 
                 encoders=encoders, 
                 class_weights=None,
                 post_trans=False,
                 loss_fn=None):
        super().__init__()
        self.lr = lr  #hparams['lr']
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pretrained = pretrained
        self.encoders = encoders
        self.encoder = self.encoders[encoder] if encoder < len(self.encoders) else 'resnet34'
        self.pretrained_weights = None if (self.pretrained==0) else 'imagenet'
        self.activation = 'sigmoid' if self.n_classes==1 else None  #'softmax2d'
        self.class_weights = class_weights if class_weights is not None else torch.ones((self.n_classes,)).float()
        self.post_trans = post_trans 
        self.loss_fn = loss_fn
        self.net = smp.Unet(
            encoder_name=self.encoder, #'resnet34',# 
            encoder_weights=self.pretrained_weights, 
            activation=self.activation, 
            in_channels=self.n_channels, 
            classes=self.n_classes)
        
    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_nb):
        img, mask = batch  #['img'], batch['seg']
        img = img.float()
        mask = mask.long()
        out = self(img)
        out = F.softmax(out, dim=1).float()
        
        out1 = F.one_hot(torch.Tensor.argmax(out, dim=1), self.n_classes).permute(0,3,1,2).float()
        mask1 = F.one_hot(mask, self.n_classes).permute(0, 3, 1, 2).float()
        
        logs = get_loss_and_metrics(self.n_classes, 'train', out, out1, mask, mask1)
        loss_func = get_loss_function(class_weights=self.class_weights.to(out.device), 
                                      key=self.loss_fn, 
                                      dist_matrix=get_wasserstein_distance_matrix(self.n_classes), 
                                      weighting_mode='GDL'
                                     )
        logs['train_loss'] = loss_func(out, mask) if self.loss_fn in ['ce', 'gwd'] else loss_func(out, mask1)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': logs['train_loss']}

    def validation_step(self, batch, batch_idx):
        img, mask = batch  #['img'], batch['seg']
        img = img.float()
        mask = mask.long()
        out = self(img)
        out = F.softmax(out, dim=1).float()
        if self.post_trans:
            out = torch.from_numpy(np.array([self.post_trans(i).cpu().numpy() for i in out])).to(mask.device)
        
        out1 = F.one_hot(torch.Tensor.argmax(out, dim=1), self.n_classes).permute(0,3,1,2).float()
        mask1 = F.one_hot(mask, self.n_classes).permute(0, 3, 1, 2).float()
        
        logs = get_loss_and_metrics(self.n_classes, 'val', out, out1, mask, mask1)
        
        loss_func = get_loss_function(class_weights=self.class_weights.to(out.device), 
                                      key=self.loss_fn, 
                                      dist_matrix=get_wasserstein_distance_matrix(self.n_classes), 
                                      weighting_mode='GDL'
                                     )
        logs['val_loss'] = loss_func(out, mask) if self.loss_fn in ['ce', 'gwd'] else loss_func(out, mask1)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': logs['val_loss']}
           
    def test_step(self, batch, batch_idx):
        img, mask = batch  #['img'], batch['seg']
        img = img.float()
        mask = mask.long()
        out = self(img)
        out = F.softmax(out, dim=1).float()
        if self.post_trans:
            out = torch.from_numpy(np.array([self.post_trans(i).cpu().numpy() for i in out])).to(mask.device)
        
        out1 = F.one_hot(torch.Tensor.argmax(out, dim=1), self.n_classes).permute(0,3,1,2).float()
        mask1 = F.one_hot(mask, self.n_classes).permute(0, 3, 1, 2).float()
        
        logs = get_loss_and_metrics(self.n_classes, 'test', out, out1, mask, mask1)
        
        loss_func = get_loss_function(class_weights=self.class_weights.to(out.device), 
                                      key=self.loss_fn, 
                                      dist_matrix=get_wasserstein_distance_matrix(self.n_classes), 
                                      weighting_mode='GDL'
                                     )
        logs['test_loss'] = loss_func(out, mask) if self.loss_fn in ['ce', 'gwd'] else loss_func(out, mask1)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': logs['test_loss']}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]       


class ClassificationModel(pl.LightningModule):
    def __init__(self, model, n_classes):
        super().__init__()
        self.net = model
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score(num_classes=n_classes)
        self.mat_corr = torchmetrics.MatthewsCorrCoef(num_classes=n_classes)
        self.specificity = torchmetrics.Specificity(num_classes=n_classes)
        self.recall = torchmetrics.Recall(num_classes=n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        #choosing an optimizer
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1, verbose=False)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": sch,
            "monitor": "val_loss",
        },
    }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        logs = {
            'train_acc': self.accuracy(y_pred, y), 
            'train_mat_corr': self.mat_corr(y_pred, y), 
            'train_recall': self.recall(y_pred, y), 
            'train_specificity': self.specificity(y_pred, y), 
            'train_f1_score': self.f1_score(y_pred, y), 
            'train_loss': self.loss_func(y_pred,y)
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs['train_loss']
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        logs = {
            'val_acc': self.accuracy(y_pred, y), 
            'val_mat_corr': self.mat_corr(y_pred, y), 
            'val_recall': self.recall(y_pred, y), 
            'val_specificity': self.specificity(y_pred, y), 
            'val_f1_score': self.f1_score(y_pred, y), 
            'val_loss': self.loss_func(y_pred,y)
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs['val_loss']
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        logs = {
            'test_acc': self.accuracy(y_pred, y), 
            'test_mat_corr': self.mat_corr(y_pred, y), 
            'test_recall': self.recall(y_pred, y), 
            'test_specificity': self.specificity(y_pred, y), 
            'test_f1_score': self.f1_score(y_pred, y), 
            'test_loss': self.loss_func(y_pred,y)
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs['test_loss']

