import torch
import torch.nn as nn

from openstl.modules import ConvLSTMCell

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
        total_loss = loss + heuristic_loss
        
        return total_loss

    
    
class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = CustomLoss()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
    


    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None

        return next_frames, loss
