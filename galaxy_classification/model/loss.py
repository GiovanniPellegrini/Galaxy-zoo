import torch.nn as nn
import torch
from galaxy_classification.training_utils import CLASS_GROUPS


class LogCoshLoss(nn.Module):
    
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, weight=None):
        diff = pred - target
        loss = torch.log(torch.cosh(diff + self.eps))   
        if weight is not None:
            loss = loss * weight.unsqueeze(0)          
        return loss.mean()



def regression_loss(outputs, labels):
    mse = nn.MSELoss(reduction='none')
    logcosh= LogCoshLoss()
    losses = []
    for key in outputs.keys():
        pred, targ = outputs[key], labels[key] 
                                  
        # compute the weights for each class
        #w = torch.tensor([weights[c] for c in CLASS_GROUPS[f'Class {key[-1]}']], 
                          #device=se.device, dtype=se.dtype)
        if key in ("q6", "q8"):                       
            losses.append(logcosh(pred, targ))
        else:                                           # MSE standard
            se = mse(pred, targ) #* w.unsqueeze(0)
            losses.append(se.mean())
    return sum(losses)


