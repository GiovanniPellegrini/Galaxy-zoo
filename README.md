# Galaxy Zoo — CNN Classification & Regression with PyTorch

## 


This codebase trains and evaluates Convolutional Neural Networks that **classify** galaxy morphology or **regress** probability vectors for several Galaxy Zoo questions, using the public *Galaxy Zoo* image+label dataset. The pipeline covers data loading, stratified splits, training with early‑stopping, checkpointing of the best model, and rich evaluation plots.

---

## Dependencies (baseline)

Create a fresh *venv/conda* environment and install:

```bash
pip install torch torchvision  # GPU or CPU build as appropriate
pip install pandas numpy scikit-learn matplotlib pillow pyyaml
```



*(Add/remove anything extra you need, e.g. CUDA‑specific wheels, Weights & Biases, etc.)*

---
## Download the images
the image datasets can be downloaded from [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). The file you should download images_train_rev1.
zip into data/images and unzip it (unzip images_train_rev1.zip)


All the labels for the tasks are inside `data\`

---

---

## Training a model — `train.py`

### Prepare a YAML config

A single file groups `model`, `data`, and `training` hyper‑params. Minimal example:

```yaml
# configs/base_classification.yaml
model:
  network_id: "classification"   # or "regression"
  channel_count_hidden: 32        # CNN width
  convolution_kernel_size: 3
  mlp_hidden_unit_count: 128

data:
  image_path: "data/images"        # Folder with .jpg
  label_path: "data/labels.csv"    # Kaggle CSV
  input_image_shape: [64, 64]
  batch_size: 256
  validation_fraction: 0.1
  test_fraction: 0.1

training:
  num_epochs: 20
  learning_rate: 0.02
  weight_decay: 1.0e-5
```

###  Launch training

```bash
python train.py --run_name my_first_run --config config_base_classification.yaml
```

* `--run_name` creates `outputs/my_first_run/` where weights, plots and logs are stored. If the directory already exists, it will update with `best_model.pth`
* `--config` points to the YAML above.

At the end you will find:

* `best_model.pth` — best checkpoint selected on validation loss.
* `traning_summary.pkl` - save all the losses and accuracy of a training
* `utilized_config.yaml` — the exact config copied for reproducibility.
* `plots/` — loss/accuracy curves, ROC AUC or regression scatter plots.
* Console printout of **test accuracy** (classification) or regression metrics.

---

## Evaluating a trained model — `evaluate.py`

```bash
python evaluate.py \
  --model_directory my_first_run \
  --config config.yaml
  --run_name eval_my_first_run \
  --tta                 # optional test‑time augmentation for classification
```

* `--model_directory` points to a folder inside `outputs/` that already contains `best_model.pth`
* `--config` to specify the size of `test_dataloader`
* `--run_name` decides where the new evaluation artefacts go (`outputs/eval_my_first_run`).
* `--tta` averages 5 augmented inferences for extra robustness.

Outputs include:

* Final **test accuracy** or regression metrics printed to console.
* Confusion matrix & ROC curves (classification) or distribution plots (regression).
* `utilized_config.yaml` copied alongside plots for traceability.

# License

Galaxy zoo is licensed under the MIT License.

- **Open a pull request**: If you have written code that improves the project, you can submit it as a pull request.
- **Report bugs**: If you find a bug, you can report it by creating an issue. Please provide a detailed description of the bug and include the steps necessary to reproduce it.

You can contact me on my github profile
-[Giovanni Pellegrini](https://github.com/GiovanniPellegrini)
