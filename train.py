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
import numpy as np
from torch import nn, optim
from sklearn.utils.class_weight import compute_class_weight

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
from galaxy_classification.training_utils import fit, compute_epoch_accuracy
from galaxy_classification.model.loss import regression_loss

# ------------------------ ARGUMENTS ------------------------
parser = argparse.ArgumentParser(
    description="Train a Galaxy CNN (classification or regression)"
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to config YAML file"
)
parser.add_argument(
    "--run_name", type=str, required=True,
    help="Name of this run for outputs/ directory"
)
args = parser.parse_args()

# ------------------------ DEVICE SETUP ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("TRAINING GALAXY CNN")
print(f"Device: {device}")
print('-' * 60)

# ------------------------ LOAD CONFIG ------------------------
with open(args.config, 'r') as f:
    full_config = yaml.safe_load(f)
print(f"Loaded configuration from {args.config}")
mode = full_config['model']['network_id']



#create output directory
run_dir = os.path.join("outputs", args.run_name)
os.makedirs(run_dir, exist_ok=True)



# ------------------------ DATASET & DATALOADERS ------------------------
data_conf = full_config['data']

dataset = GalaxyDataset.load(
    image_path=data_conf['image_path'],
    label_path=data_conf['label_path'],
)
print(f"Total samples: {len(dataset)}")

if mode == 'classification':
    dataloaders = SplitGalaxyClassificationDataSet(
        dataset,
        batch_size=data_conf['batch_size'],
        validation_fraction=data_conf['validation_fraction'],
        test_fraction=data_conf['test_fraction'],
    )
elif mode == 'regression':
    dataloaders = SplitGalaxyRegressionDataset(
        dataset,
        batch_size=data_conf['batch_size'],
        validation_fraction=data_conf['validation_fraction'],
        test_fraction=data_conf['test_fraction'],
    )
else:
    raise ValueError(f"Unsupported network_id: {mode}")

print(f"Training samples: {len(dataloaders.training_dataloader.dataset)}")
print(f"Validation samples: {len(dataloaders.validation_dataloader.dataset)}")
print(f"Test samples: {len(dataloaders.test_dataloader.dataset)}")

# ------------------------ MODEL SETUP ------------------------
if mode == 'classification':
    model_conf = GalaxyClassificationCNNConfig(**full_config['model'])
elif mode == 'regression':
    model_conf = GalaxyRegressionCNNConfig(**full_config['model'])

model = build_network(
    input_image_shape=tuple(data_conf['input_image_shape']),
    config=model_conf,
).to(device)
print(f"Model built: {model_conf.network_id}")

best_model_path = os.path.join(run_dir, 'best_model.pth')
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded existing best model from {best_model_path}")




# ------------------------ OPTIMIZER & LOSS ------------------------
optimizer_conf = full_config.get('training')
optimizer = optim.Adam(
    model.parameters(),
    lr=float(optimizer_conf.get('learning_rate')),
    weight_decay=float(optimizer_conf.get('weight_decay'))
)

if mode == 'classification':
    # compute class weights if provided
    train_labels = dataloaders.training_dataloader.dataset.labels.values.flatten()
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float, device=device)
    )
else:
    # regression uses custom weighted MSE
    criterion = lambda outputs, labels: regression_loss(
        outputs, labels
    )
# ------------------------ PRINT CONFIG ------------------------
print("Input image shape:", data_conf['input_image_shape'])
print("Batch size:", data_conf['batch_size'])
print("num_epochs:", optimizer_conf['num_epochs'])
print("Learning rate:", optimizer_conf['learning_rate'])


# ------------------------ TRAINING ------------------------
run_dir = os.path.join('outputs', args.run_name)
os.makedirs(run_dir, exist_ok=True)

summary = fit(
    model=model,
    optimizer=optimizer,
    loss_fun=criterion,
    train_dataloader=dataloaders.training_dataloader,
    val_dataloader=dataloaders.validation_dataloader,
    num_epochs=optimizer_conf['num_epochs'],
    run_dir=run_dir,
    mode=mode,
    device=device,
)

# ------------------------ LOAD BEST MODEL ------------------------
best_model_path = os.path.join(run_dir, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()
print(f"Loaded best model from {best_model_path}")

# ------------------------ EVALUATION ------------------------
if mode == 'classification':
    test_accuracy = compute_epoch_accuracy(
        model,
        dataloaders.test_dataloader,
        mode,
        device,
    )
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    summary.plot_roc_auc(model, dataloaders.test_dataloader, path=run_dir)
    summary.plot_confusion_matrix(model, dataloaders.test_dataloader, path=run_dir)
else:
    summary.evaluate_regression_metrics(model, dataloaders.test_dataloader, path=run_dir)
    summary.save_predictions(model, dataloaders.test_dataloader)
    summary.plot_true_pred_distributions(path=run_dir)

# ------------------------ SAVE & CLEANUP ------------------------
summary.save_plots(run_dir)
summary.save_summary(run_dir)
shutil.copy(args.config, os.path.join(run_dir, 'utilized_config.yaml'))

print("Training and evaluation complete.")
