from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
import os



CLASS_GROUPS = {
    'Class 1': ['Class1.1', 'Class1.2', 'Class1.3'],
    'Class 2': ['Class2.1', 'Class2.2'],
    'Class 6': ['Class6.1', 'Class6.2'],
    'Class 7': ['Class7.1', 'Class7.2', 'Class7.3'],
    'Class 8' : ['Class8.1', 'Class8.2', 'Class8.3, Class8.4', 'Class8.5', 'Class8.6', 'Class8.7']
                  }

"""
Transormations of some labels
x-> (x+eps)^gamma
"""
EPS = 1e-6
GAMMA= 0.3
CLASS_GROUPS_TRANSFORMED={
    'Class 1': ['Class1.3'],
    'Class 2': ['Class2.1'],
    'Class 7': ['Class7.1', 'Class7.2','Class7.3'],
    'Class 8': ['Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7']
}
HEAD_IDXS_TRANSFORMED = {
    "q1": [2],        # Class1.3
    "q2": [0],        # Class2.1
    "q7": [0, 1, 2],     # Class7.1, Class7.2, Class7.3
    "q8": [0, 1, 2, 3, 4, 5, 6]  # Class8.1 to Class8.7
}


"""
    Trains a model for a specified number of epochs.
    Computes and training and validation loss and accuracy and return them in a summary object.
    """
def fit(
    model:Module,
    optimizer:Optimizer,
    loss_fun:Callable[[Tensor| dict, Tensor| dict], Tensor],
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    num_epochs:int,
    run_dir:str,
    mode: str,
    device:str="cpu"
) -> TrainingSummary:
    from galaxy_classification.training_summary import TrainingSummary    
    checkpoint = BestModel(run_dir)
    early_stopper = EarlyStopper(patience=4, delta=1e-5)
    summary = TrainingSummary(interval=1, mode=mode)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = compute_epoch_loss(model, train_dataloader, loss_fun, optimizer,device)
        train_accuracy = compute_epoch_accuracy(model, train_dataloader,mode,device)

        model.eval()
        with torch.no_grad():
            # Validation loss and accuracy
            val_loss = compute_epoch_loss(model, val_dataloader, loss_fun, None,device)
            val_accuracy = compute_epoch_accuracy(model, val_dataloader, mode,device)
    
        summary.append_epoch_summary(
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )
        checkpoint.check(model, val_loss)
        if early_stopper.check(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return summary


"""
    Computes the average loss over an epoch.
    If optimizer is provided, performs a training step. Otherwise, runs in evaluation mode.
    """
def compute_epoch_loss(
        model:Module,
        dataloader:DataLoader,
        loss_fun:Callable[[Tensor|dict, Tensor|dict], Tensor],
        optimizer:Optimizer | None = None,
        device:str="cpu"
):  
    total_loss = 0.0
    num_batches = len(dataloader)
    model.to(device)
    
    for batch in dataloader:
        inputs = batch["images"].to(device)
        labels = batch["labels"]
        if isinstance(labels, dict):
            labels = {k: v.to(device) for k, v in labels.items()}
        else:
            labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / num_batches
    return average_loss

"""
    Computes accuracy for a classification model.
    Assumes labels are integer indices and predictions are logits.
"""
def compute_epoch_accuracy(
        model:Module,
        dataloader:DataLoader,
        mode: str,
        device:str
):
    if mode=="regression":
        return 0.0
    elif mode == "classification":
        model.to(device)
        model.eval()
        prediction_count = 0
        correct_prediction_count = 0
        for batch in dataloader:
            images = batch["images"].to(device)
            labels=batch["labels"].to(device)
            labels_predicted = model(images)

            labels_predicted = labels_predicted.argmax(dim=1)
            correct_prediction_count += torch.sum(labels_predicted == labels).item()
            prediction_count += len(batch["images"])

        return correct_prediction_count / prediction_count
    else:
        raise ValueError("Invalid mode. Choose 'classification' or 'regression'.")


class EarlyStopper():
    """
    Early stopping based on validation loss.
    Stops training if validation loss does not improve for a specified number of epochs.
    """
    def __init__(self,patience:int=3,delta:float=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.epochs_no_improvement=0

    def check(self, val_loss:float) -> bool:
        """
        Check if validation loss has improved.
        If not, increment the counter.
        If it has improved, reset the counter and update the best loss.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.epochs_no_improvement = 0
            return False
        else:
            self.epochs_no_improvement += 1
            return True if self.epochs_no_improvement >= self.patience else False
        
class BestModel():
    """
    Save the best model based on validation loss.
    """
    def __init__(self,path:str):
        os.makedirs(path, exist_ok=True)
        self.best_path = os.path.join(path, "best_model.pth")
        self.best_model = None
        self.best_loss = float("inf")
        

    def check(self, model:Module, loss:float):
        """
        Check if validation loss has improved.
        If not, increment the counter.
        If it has improved, reset the counter and update the best loss.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = model
            torch.save(model.state_dict(), self.best_path)
            print(f"Best model saved with loss: {self.best_loss:.4f}")

"""
    Test time augmentation.
    Applies test time augmentation to the model and returns the predictions.
    """

tta_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])


def tta(model:Module, images,mode:str,n:int=5, device:str="cpu"):

    model.eval()
    preds = []
    for _ in range(n):
        img_aug = torch.stack([tta_transform(img) for img in images])
        img_aug = img_aug.to(device)
        with torch.no_grad():
            output = model(img_aug)
            if mode == "classification":
                output = torch.softmax(output, dim=1)
            preds.append(output)

    if mode == "classification":
        return torch.stack(preds).mean(dim=0)  # Tensor [B, C]
    elif mode == "regression":
        # preds = list of dicts: merge by key
        merged = {}
        for key in preds[0].keys():
            merged[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
        return merged
    else:
        raise ValueError("Invalid mode for TTA.")