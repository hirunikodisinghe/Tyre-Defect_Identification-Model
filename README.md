# Tire Defect Identification Model

A Python project for classifying tire images using an EfficientNet-B0 model. The prediction script opens an image selection dialog and displays the top three predicted classes, model confidence, and an uncertainty decision.

## Current prediction configuration

- Entry point: `predict_single_image.py`
- Model: `best_tire_defect_model_4.pth` in the repository root
- Labels: `class_names.txt` in the repository root
- Device: CUDA when available; otherwise CPU
- Image preprocessing: RGB, resize to 224 × 224, and ImageNet normalization
- Prediction: average of original and horizontally flipped image probabilities

`tyre_defect_app.py` currently contains only a comment; use `predict_single_image.py` to run predictions.

## Classes

The current `class_names.txt` order is:

1. Abnormal_Wear
2. Bead_Damage
3. Bulge
4. Cut
5. Shoulder_Crack
6. Sidewall_Crack
7. Sidewall_Damage
8. Tire_Burst
9. Tread_Chunking
10. Tread_Damage
11. Tread_Separation
12. noDefect

Keep labels in the same order used when training the selected model. Do not reorder or rename labels without checking the training metadata.

## Setup on Windows

Use a Python version supported by the selected PyTorch release, with Tkinter available. You also need Git, Git LFS, access to this private repository, and enough disk space for the datasets and models.

### Download the repository

Run in Command Prompt from the parent folder where you want the project:

```cmd
git lfs install
git clone https://github.com/hirunikodisinghe/Tyre-Defect_Identification-Model.git
cd Tyre-Defect_Identification-Model
git lfs pull
```

The repository includes large datasets and models. A full clone may take time and consume substantial disk space and LFS download bandwidth.

### Create an environment and install prediction dependencies

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For a CPU-specific or CUDA-specific PyTorch installation, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system and hardware.

The supplied `requirements.txt` covers the inspected prediction script only. It is not a complete dependency list for all training, review, or extraction scripts. Versions are unpinned; a fresh installation has not been tested. Preserve a record of the package versions in your working environment before attempting to reproduce it elsewhere.

`tkinter`, `pathlib`, and `math` are Python standard-library modules, not entries to install from this requirements file. If Tkinter is unavailable, install Python with Tcl/Tk support.

## Run a prediction

From the repository folder, with the environment active:

```cmd
python predict_single_image.py
```

1. Choose a tire image in the file dialog and click **Open**.
2. Read the top three predictions and diagnostic values in the terminal.
3. Read the result dialog and click **OK** to finish.

If the dialog is behind another window, use **Alt + Tab**. Cancelling image selection prints `No image selected.` and exits normally.

## Uncertainty handling

The script flags a prediction as uncertain when any of these conditions holds:

- Top model probability is below 0.60.
- The difference between the top two probabilities is below 0.15.
- Normalized prediction entropy exceeds 0.65.

These thresholds are heuristics. They do not constitute a tire-versus-non-tire detector and do not guarantee correctness. Model confidence is not overall test accuracy or a calibrated probability of correctness.

## Included files and data

| Path | Purpose |
| --- | --- |
| `predict_single_image.py` | Interactive image prediction |
| `best_tire_defect_model_4.pth` | Currently selected prediction model |
| `class_names.txt` | Ordered prediction labels |
| `train_tire_model.py` | Training script |
| `train_split_dataset/` | Train, validation, and test folders referenced by the current training script |
| `#TireDefect/` | Imported tire dataset, excluding `N_A.zip` |
| `classification_report.txt` | Saved evaluation report |
| `confusion_matrix.png` | Saved confusion matrix |
| `training_history.png` | Saved training chart |
| `TyreImageExtractor/tire_website_image_extractor.py` | Image extraction script |
| `accuracy_87/best_tire_defect_model.pth` | Archived alternative model, not used by the current prediction script |

The repository also contains data preparation scripts and model checkpoints. `.pth` files and the two included dataset directories are managed through Git LFS.

Dataset copies, older split directories, review outputs, and selected logs are excluded through `.gitignore`. They remain local and are not part of this GitHub backup. The oversized `N_A.zip` archive is also excluded and kept separately. Keep separate backups of any excluded files you still need.

## Training and validation status

The current training script points to `train_split_dataset/`. That path alone does not establish which dataset produced `best_tire_defect_model_4.pth`. Before retraining or reporting evaluation results, verify the model's training configuration, class order, preprocessing, dataset split, and matching evaluation report.

The selected model was observed loading on CPU and producing a prediction for one image. This is a functional smoke check, not an accuracy evaluation. The archived report and chart files have not been verified as belonging to model version 4. The name `accuracy_87` is not evidence of validated accuracy.

## Troubleshooting

- **Model not found:** Check that the real `best_tire_defect_model_4.pth` file is in the repository root and that Git LFS downloads completed.
- **Class names file not found:** Restore the matching `class_names.txt` beside the script.
- **Missing package:** Activate the environment and install the prediction requirements. Other scripts may require additional dependencies.
- **Model loading mismatch:** Check the model architecture, number of classes, and checkpoint format. Do not solve this by arbitrarily renaming a different model.
- **Uncertain result:** Try a clear image and review the output. Changing thresholds alone does not improve the trained model.

## References

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [pip requirements files](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [GitHub Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
