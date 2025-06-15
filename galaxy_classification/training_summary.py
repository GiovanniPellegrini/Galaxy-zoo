import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from galaxy_classification.training_utils import EPS, GAMMA, HEAD_IDXS_TRANSFORMED

import torch
import numpy as np
import pickle
from typing import Dict, Tuple

# --- Classification head (exercise 1: Class1.1, Class1.2, Class1.3) ---
CLASS1_LABELS: dict[int, str] = {
    0: "Smooth and round (no disk)",  # Class1.1
    1: "Has disk",                     # Class1.2
}

# --- Regression head q1 (exercise 2: Class1.1, Class1.2, Class1.3) ---
Q1_LABELS: dict[int, str] = {
    0: "Smooth and round (no disk)",  # Class1.1
    1: "Has disk",                     # Class1.2
    2: "Image flawed"                  # Class1.3
}

# --- Regression head q2 (exercise 2: Class2.1, Class2.2) ---
Q2_LABELS: dict[int, str] = {
    0: "Not edge-on disk",  # Class2.1
    1: "Edge-on disk"       # Class2.2
}

# --- Regression head q7 (exercise 2: Class7.1, Class7.2, Class7.3) ---
Q7_LABELS: dict[int, str] = {
    0: "Completely round",   # Class7.1
    1: "In between",         # Class7.2
    2: "Cigar-shaped"        # Class7.3
}

# --- Regression head q6 (exercise 3: Class6.1, Class6.2) ---
Q6_LABELS: dict[int, str] = {
    0: "No odd features",   # Class6.1
    1: "Has odd features"   # Class6.2
}

# --- Regression head q8 (exercise 3: Class8.1, Class8.2, Class8.3, Class8.4, Class8.5, Class8.6, Class8.7) ---
Q8_LABELS: dict[int, str] = {
    0: "Ring",         # Class8.1
    1: "Lens or arc",  # Class8.2
    2: "Disturbed",    # Class8.3
    3: "Irregular",    # Class8.4
    4: "Other",        # Class8.5
    5: "Merger",       # Class8.6
    6: "Dust lane"     # Class8.7
}


HEAD_LABELS: dict[str, dict[int, str]] = {
    "q1": Q1_LABELS, 
    "q2": Q2_LABELS,
    "q7": Q7_LABELS,
    "q6": Q6_LABELS,
    "q8": Q8_LABELS
}


"""
Class to summarize the training process:
- It stores the training and validation loss and accuracy in lists, and save them in a pickle file.
- It prints the training and validation loss and accuracy every epoch_printing_interval epochs.
- It save the auc e roc curves for each class for classification tasks.
- 

"""
class TrainingSummary:
    def __init__(self, interval:int, mode:str, device:str="cpu"):
        self.epochs_printing_interval:int=interval
        self.training_losses: list[float]=[]
        self.training_accuracies: list[float]=[]
        self.validation_losses: list[float]=[]
        self.validation_accuracies: list[float]=[]
        self.gamma: float = GAMMA
        self.eps: float = EPS
        self.predictions:Dict[str, Tuple[torch.Tensor, torch.Tensor]]={}
        self.mode:str=mode
        self.device=device
        self.epoch_index:int=0

    def append_epoch_summary(
        self,
        training_loss: float,
        training_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
    ):
        if (self.epoch_index + 1) % self.epochs_printing_interval == 0:
            if self.mode == "classification":
                print(  "*" *50 +"\n"+
                        f"Epoch {self.epoch_index + 1}, "
                        f"Train Loss: {training_loss:.6f}, Accuracy: {training_accuracy * 100.0:.2f}% | "
                        f"Val Loss: {validation_loss:.6f}, Accuracy: {validation_accuracy * 100.0:.2f}%"
                    )
            elif self.mode == "regression":
                    print( "*" * 50 + "\n" +
                        f"Epoch {self.epoch_index + 1}, "
                        f"Train Loss: {training_loss:.6f} | Val Loss: {validation_loss:.6f}"
                    )

        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        if self.mode == "classification":
            self.validation_accuracies.append(validation_accuracy)
            self.training_accuracies.append(training_accuracy)
        self.epoch_index += 1
    
    def save_summary(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_path = os.path.join(path, "training_summary.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Training summary saved to {file_path}")



    def save_plots(self, path):
        plot_dir = os.path.join(path, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Numero di subplots: 2 per classification, 1 per regression
        if self.mode == "classification":
            fig, ax = plt.subplots(2, figsize=(12, 8))
        elif self.mode == "regression":
            fig, ax = plt.subplots(1, figsize=(12, 4))
            ax = [ax]  # per gestire come lista

        # Plot della loss
        ax[0].plot(self.training_losses, label="Training Loss")
        ax[0].plot(self.validation_losses, label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid()

        # Plot dell'accuracy solo se in classificazione
        if self.mode == "classification":
            ax[1].plot(self.training_accuracies, label="Training Accuracy")
            ax[1].plot(self.validation_accuracies, label="Validation Accuracy")
            ax[1].set_title("Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].legend()
            ax[1].grid()

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "loss_accuracy_plot.png" if self.mode == "classification" else "loss_plot.png"))
        plt.close(fig)
        print(f"Plots saved in {plot_dir}")
            

    #       ------------------------ CLASSIFICATION ------------------------


    def plot_roc_auc(self,model,dataloader, path):
        if self.mode != "classification":
            print("Skipping ROC AUC (not applicable for regression).")
            return
        plots_dir = os.path.join(path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        model.eval()
        y_probs = []
        y_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                y_probs.append(probs.cpu())
                y_labels.append(labels.cpu())


        y_probs = torch.cat(y_probs).numpy()   # shape: (N, C)
        y_true = torch.cat(y_labels).numpy() # shape: (N,)
        n_classes = y_probs.shape[1]
        #y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        #print("Class distribution in test-set (positivi):", y_true_bin.sum(axis=0))


        plt.figure(figsize=(10, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc_val = auc(fpr, tpr)
            class_name = CLASS1_LABELS.get(i, f"Class {i}")
            plt.plot(fpr, tpr, lw=2,
                     label=f"{class_name} (AUC = {roc_auc_val:.2f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        output_path = os.path.join(plots_dir, "roc_auc.png")
        plt.savefig(output_path)
        plt.close()
        print(f"ROC AUC plot saved to {output_path}")
        
    
    def plot_confusion_matrix(self, model, dataloader, path, normalize: bool = True):
        if self.mode != "classification":
            print("Skipping confusion matrix (not applicable for regression).")
            return

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        cm = confusion_matrix(all_targets, all_preds, normalize='true' if normalize else None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS1_LABELS.values()))
        disp.plot(cmap=plt.cm.Blues)
        disp.figure_.tight_layout()
        output_path = os.path.join(path, "plots", "confusion_matrix.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
        
    
    
    #       ------------------------ REGRESSION ------------------------

    
      # Compute different metrics for Regression  
    def evaluate_regression_metrics(self, model, dataloader, path: str):
        if self.mode != "regression":
            print("Skipping regression metrics (not applicable for classification).")
            return

        model.eval()
        all_preds = {}
        all_targets = {}

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["images"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
                outputs = model(inputs)

                for key in outputs.keys():
                    if key not in all_preds:
                        all_preds[key] = []
                        all_targets[key] = []

                    all_preds[key].append(outputs[key].cpu())
                    all_targets[key].append(labels[key].cpu())

        metrics = {}
        for key in all_preds.keys():
            y_pred = torch.cat(all_preds[key]).numpy()
            y_true = torch.cat(all_targets[key]).numpy()

            metrics[key] = {
                "MSE": mean_squared_error(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred)
            }

        print("\nRegression Metrics:")
        for key, m in metrics.items():
            print(f"{key.upper()} — MSE: {m['MSE']:.6f}, MAE: {m['MAE']:.6f}, R²: {m['R2']:.6f}")

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "regression_metrics.txt"), "w") as f:
            for key, m in metrics.items():
                f.write(f"{key.upper()} — MSE: {m['MSE']:.6f}, MAE: {m['MAE']:.6f}, R²: {m['R2']:.6f}\n")

        return metrics
    
     # Anti transformation the labels: 
    def inverse_transform(self, x:torch.Tensor, head:str) -> torch.Tensor:
        """
        Inverse transform
        """
        x = x.clone()
        if head in HEAD_IDXS_TRANSFORMED and self.gamma != 1.0:
            idxs = torch.tensor(HEAD_IDXS_TRANSFORMED[head], device=x.device)

            # Clamp only the transformed columns
            sel = x[:, idxs]
            sel = torch.clamp(sel, min=self.eps)

            # Apply inverse power-law
            x[:, idxs] = sel.pow(1.0 / self.gamma)

        return x
    
     #  Save predictions and labels for dataset in self.predicitions
    def save_predictions(self, model, dataloader, device: str = "cpu"):
        model.eval()
        model.to(device)

        
        dict_true: dict[str, list[torch.Tensor]] = {}
        dict_pred: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(device)
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                outputs = model(images)

                for key, pred in outputs.items():
                    # inverse‐transform only selected channels of this head
                    y_pred_it = self.inverse_transform(pred.detach().cpu(), head=key)
                    y_true_it = self.inverse_transform(labels[key].detach().cpu(), head=key)

                    # if key == "q1":
                      #   # renormalize q1 so che somma a 1
                        # denom_pred = y_pred_it.sum(dim=1, keepdim=True).clamp_min(self.eps)
                         #y_pred_it = y_pred_it / denom_pred
                         #denom_true = y_true_it.sum(dim=1, keepdim=True).clamp_min(self.eps)
                         #y_true_it = y_true_it / denom_true
                        
                    dict_true.setdefault(key, []).append(y_true_it)
                    dict_pred.setdefault(key, []).append(y_pred_it)

        # concateno le liste di tensori per ogni head
        self.predictions = {
            k: (torch.cat(dict_true[k]), torch.cat(dict_pred[k]))
            for k in dict_true
        }
        print("Predictions saved.")

        
     #  plots different lalebels and their predictions   

    def plot_true_pred_distributions(self, path: str, bins: int = 50):
        """
        For each regression head in self.predictions, plot overlapping histograms
        of the true vs. predicted probability distributions.
        """
        if self.mode != "regression":
            print("Skipping distribution plots (not applicable for classification).")
            return

        plots_dir = os.path.join(path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        heads = list(self.predictions.keys())
        n_heads = len(heads)
        # un subplot per head, impila in colonna
        fig, axs = plt.subplots(n_heads, 1, figsize=(8, 4 * n_heads), squeeze=False)

        for i, head in enumerate(heads):
            ax = axs[i, 0]
            y_true, y_pred = self.predictions[head]  
            Ck = y_true.shape[1]

            for j in range(Ck):
                true_vals = y_true[:, j].numpy()
                pred_vals = y_pred[:, j].numpy()
                def choose_bins(data):
                    unique_count = np.unique(data).size
                    if unique_count <= 1:
                        return 1
                    return min(bins, unique_count)
                bins_true = choose_bins(true_vals)
                bins_pred = choose_bins(pred_vals)
                
                ax.hist(
                    true_vals,
                    bins=bins_true,
                    alpha=0.4,
                    label=f"True {HEAD_LABELS.get(head, {}).get(j, f'{head}.{j+1}')}"
                )
                ax.hist(
                    pred_vals,
                    bins=bins_pred,
                    histtype="step",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Pred {HEAD_LABELS.get(head, {}).get(j, f'{head}.{j+1}')}"
                )

            ax.set_title(f"Distribution for {head}")
            ax.set_xlabel("Probability Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, "true_vs_pred_distributions.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Distribution plots saved to {out_path}")
        