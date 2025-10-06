import warnings
# suppress warnings about numpy.longdouble signature mismatch
warnings.filterwarnings(
    "ignore",
    message="Signature.*numpy.longdouble.*"
)


import os
import shutil
import argparse
import yaml
import torch

from galaxy_classification.galaxy_dataloader import (
    GalaxyDataset,
    SplitGalaxyClassificationDataSet,
    SplitGalaxyRegressionDataset,
)
from galaxy_classification.model.galaxy_cnn import (
    GalaxyClassificationCNNConfig,
    GalaxyRegressionCNNConfig,
)
from galaxy_classification.model.build import build_network
from galaxy_classification.training_utils import compute_epoch_accuracy, tta
from galaxy_classification.training_summary import TrainingSummary

# ------------------------ ARGUMENTS ------------------------
parser = argparse.ArgumentParser(
    description="Evaluate a trained Galaxy CNN (classification or regression)"
)
parser.add_argument(
    "--model_directory", type=str, required=True,
    help="Name of the trained run directory under outputs/"
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to the utilized config YAML file for this model"
)
parser.add_argument(
    "--run_name", type=str, required=True,
    help="Name for this evaluation run directory under outputs/"
)
parser.add_argument(
    "--tta", action="store_true",
    help="Apply test-time augmentation"
)
args = parser.parse_args()

# ------------------------ DEVICE SETUP ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("EVALUATING GALAXY CNN")
print(f"Device: {device}")
print("---------------------")

# ------------------------ PATHS & CONFIG ------------------------
model_dir = os.path.join("outputs", args.model_directory)
eval_dir = os.path.join("outputs", args.run_name)
os.makedirs(eval_dir, exist_ok=True)

config_path = args.config
if not os.path.exists(config_path):
    config_path = os.path.join(model_dir, "utilized_config.yaml")
    print(f"Using default config path: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)
print(f"Loaded configuration from {config_path}")

mode = full_config["model"]["network_id"]
print(f"Task mode: {mode}")


# ------------------------ DATASET & DATALOADERS ------------------------
data_conf = full_config["data"]


dataset = GalaxyDataset.load(
    image_path=data_conf["image_path"],
    label_path=data_conf["label_path"],
)


if mode == "classification":
    dataloaders = SplitGalaxyClassificationDataSet(
        dataset,
        batch_size=data_conf["batch_size"],
        validation_fraction=data_conf["validation_fraction"],
        test_fraction=data_conf["test_fraction"],
    )
elif mode == "regression":
    dataloaders = SplitGalaxyRegressionDataset(
        dataset,
        batch_size=data_conf["batch_size"],
        validation_fraction=data_conf["validation_fraction"],
        test_fraction=data_conf["test_fraction"],
    )
else:
    raise ValueError(f"Unsupported network_id: {mode}")

test_loader = dataloaders.test_dataloader
print(f"Test set size: {len(test_loader.dataset)}")


# ------------------------ MODEL SETUP ------------------------
if mode == "classification":
    model_conf = GalaxyClassificationCNNConfig(**full_config["model"])
elif mode == "regression":
    model_conf = GalaxyRegressionCNNConfig(**full_config["model"])

model = build_network(
    input_image_shape=tuple(data_conf["input_image_shape"]),
    config=model_conf,
).to(device)

# Load best checkpoint
checkpoint_path = os.path.join(model_dir, "best_model.pth")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Loaded model weights from {checkpoint_path}")

# ------------------------ SUMMARY ------------------------
summary = TrainingSummary(interval=1, mode=mode, device=device)

# ------------------------ EVALUATION ------------------------
if mode == "classification":
    print("Running classification evaluation...")
    if args.tta:
        print("Applying test-time augmentation...")
        correct, total = 0, 0
        for batch in test_loader:
            imgs = batch["images"]
            labels = batch["labels"].to(device)
            outputs = tta(
                model,
                imgs,
                mode=mode,
                n=5,
                device=device,
            )
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
    else:
        accuracy = compute_epoch_accuracy(
            model,
            test_loader,
            mode=mode,
            device=device,
        )
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # Save classification plots
    summary.plot_roc_auc(model, test_loader, path=eval_dir)
    summary.plot_confusion_matrix(model, test_loader, path=eval_dir)

elif mode == "regression":
    print("Running regression evaluation...")
    summary.evaluate_regression_metrics(model, test_loader, path=eval_dir)
    summary.save_predictions(model, test_loader, device=device)
    summary.plot_true_pred_distributions(path=eval_dir)

# ------------------------ SAVE & CLEANUP ------------------------

shutil.copy(config_path, os.path.join(eval_dir, "utilized_config.yaml"))

print("Evaluation complete.")
