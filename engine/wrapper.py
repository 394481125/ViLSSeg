import os
from utils.model import ViLSSeg
from monai.losses import GeneralizedDiceFocalLoss
from torchmetrics import Accuracy, Dice, Precision, Recall, Specificity
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import save_image
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime


class ViLSSegWrapper(pl.LightningModule):

    def __init__(self, args):

        super(ViLSSegWrapper, self).__init__()

        self.model = ViLSSeg(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}

        self.loss_fn = GeneralizedDiceFocalLoss()
        # self.loss_fn = GeneralizedDiceFocalLoss(gamma=3.0, lambda_focal=2.0) #mono

        metrics_dict = {"acc": Accuracy(task='binary'),
                        "precision": Precision(task='binary'),
                        "recall": Recall(task='binary'),
                        "dice": Dice(),
                        "MIoU": BinaryJaccardIndex(),
                        "Specificity": Specificity(average='micro', task='binary')
                        }
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)  # 微调
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)  # qata
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001, last_epoch=-1) #从零开始训练
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        return self.model.forward(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        preds, os4, uncertainty_mask = self(x)

        loss = self.loss_fn(preds, y) + 0.5 * self.loss_fn(os4, y) + 0.01 * self.loss_fn(uncertainty_mask,
                                                                            torch.zeros_like(uncertainty_mask))

        return {'loss': loss, 'preds': preds.detach(), 'os4': os4.detach(), 'y':
            y.detach(),
                'uncertainty_mask': uncertainty_mask.detach()}


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        # Call shared_step to compute loss and predictions
        outputs = self.shared_step(batch, batch_idx)

        return outputs


    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        else:
            return self(batch)

    def shared_step_end(self, outputs, stage):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage == "train":
                self.log(name, step_metric, prog_bar=True)
        return outputs["loss"].mean()

    def training_step_end(self, outputs):
        return {'loss': self.shared_step_end(outputs, "train")}

    def validation_step_end(self, outputs):
        return {'val_loss': self.shared_step_end(outputs, "val")}

    def test_step_end(self, outputs):
        return {'test_loss': self.shared_step_end(outputs, "test")}

    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)

        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage + "_loss").replace('train_', '')] for t in outputs])).item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric
        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        # log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor
        mode = ckpt_cb.mode
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor, arr_scores[best_score_idx]), file=sys.stderr)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + "%s" % nowtime)