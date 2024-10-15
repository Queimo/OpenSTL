import numpy as np
import torch.nn as nn
import torch
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric

def total_variation_loss(x):
    """
    Total Variation Loss for smoothness. Encourages smoothness by penalizing 
    large differences between neighboring values in the predicted field.
    """
    diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # Differences between rows
    diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # Differences between columns
    return diff_i.mean() + diff_j.mean()

class CustomLoss(nn.MSELoss):
    
    def __init__(self, **args):
        super().__init__(**args)
    
    def forward(self, pred, true):
        # Calculate regular MSE loss between pred and true
        loss = super().forward(pred, true)  # Use the built-in MSELoss from nn.MSELoss
        
        # Differentiable approximation for argmax
        softmax_pred = torch.softmax(pred[:, :, 0, ...].T, dim=-1)
        softmax_true = torch.softmax(true[:, :, 0, ...].T, dim=-1)
        
        # Create index tensor for the weighted sum of softmax
        indices = torch.arange(softmax_pred.shape[-1], dtype=torch.float32).to(pred.device)
        
        # Compute the softmax-weighted indices (replaces argmax)
        pred_x_i = (softmax_pred * indices).sum(-1).mean(0).flatten()
        true_x_i = (softmax_true * indices).sum(-1).mean(0).flatten()
        
        # Heuristic loss (MSE between softmax-weighted indices)
        heuristic_loss = super().forward(pred_x_i, true_x_i)
        
        # Combine the original MSE loss with the heuristic loss
        total_loss = loss + heuristic_loss*10 + total_variation_loss(pred[:, :, 0, ...].T)
        
        return total_loss

class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        
        self.criterion = CustomLoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
        
        
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all