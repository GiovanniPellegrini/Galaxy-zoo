import torch.nn as nn
import torch
from galaxy_classification.training_utils import CLASS_GROUPS


class LogCoshLoss(nn.Module):
    
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.log(torch.cosh(diff + self.eps))  
        return loss.mean()



def regression_loss(outputs, labels):
    mse = nn.MSELoss(reduction='none')
    logcosh= LogCoshLoss()
    losses = []
    for key in outputs.keys():
        pred, targ = outputs[key], labels[key] 
        
        if key in ("q6", "q8"):                       
            losses.append(logcosh(pred, targ))
        else:                                           # MSE standard
            se = mse(pred, targ) #* w.unsqueeze(0)
            losses.append(se.mean())
    return sum(losses)


