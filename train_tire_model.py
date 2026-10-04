from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "train_split_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

CHECKPOINT_PATH = PROJECT_DIR / "training_checkpoint.pth"
BEST_MODEL_PATH = PROJECT_DIR / "best_tire_defect_model.pth"

CLASS_REPORT_PATH = PROJECT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = PROJECT_DIR / "training_history.png"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"


# ============================================================
# 2. TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 8

MAX_EPOCHS = 30

LEARNING_RATE = 0.0001

EARLY_STOPPING_PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("TIRE DEFECT CLASSIFICATION - EFFICIENTNET-B0")
print("=" * 70)

print("Device:", DEVICE)

print("\nDataset paths:")
print("Train :", TRAIN_DIR)
print("Val   :", VAL_DIR)
print("Test  :", TEST_DIR)


# ============================================================
# 3. CHECK DATASET FOLDERS
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR}"
    )


# ============================================================
# 4. TRAINING AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
        ratio=(0.90, 1.10)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.10
    ),

    transforms.RandomAutocontrast(
        p=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. VALIDATION / TEST TRANSFORM
# ============================================================

val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)


class_names = train_dataset.classes

num_classes = len(class_names)


print("\nClasses found:")

for index, class_name in enumerate(class_names):
    print(
        f"{index}: {class_name}"
    )


print("\nNumber of classes :", num_classes)
print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))
print("Test images       :", len(test_dataset))


# ============================================================
# 7. VERIFY CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:

    raise ValueError(
        "Train and validation classes are different."
    )


if train_dataset.classes != test_dataset.classes:

    raise ValueError(
        "Train and test classes are different."
    )


# ============================================================
# 8. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 9. CLASS COUNTS
# ============================================================

class_counts = np.zeros(
    num_classes,
    dtype=np.int64
)


for _, label in train_dataset.samples:

    class_counts[label] += 1


print("\nTraining images per class:")


for index, class_name in enumerate(class_names):

    print(
        f"{class_name:25s}: "
        f"{class_counts[index]}"
    )


if np.any(class_counts == 0):

    raise ValueError(
        "One or more classes contain 0 training images."
    )


# ============================================================
# 10. CLASS WEIGHTS
# ============================================================

total_train = class_counts.sum()


class_weights = np.sqrt(

    total_train /

    (
        num_classes
        *
        class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")


for index, class_name in enumerate(class_names):

    print(
        f"{class_name:25s}: "
        f"{class_weights[index].item():.4f}"
    )


# ============================================================
# 11. MODEL
# ============================================================

print(
    "\nLoading pretrained EfficientNet-B0..."
)


weights = EfficientNet_B0_Weights.DEFAULT


model = efficientnet_b0(
    weights=weights
)


input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)


model = model.to(DEVICE)


# ============================================================
# 12. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.05
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# ============================================================
# 14. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


# ============================================================
# 15. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    epoch_loss = (
        running_loss
        /
        total
    )


    epoch_accuracy = (
        correct
        /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 16. NORMAL EVALUATION
# Used for validation
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()


    running_loss = 0.0

    total = 0


    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            total += labels.size(0)


    average_loss = (
        running_loss
        /
        total
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 17. TTA EVALUATION
#
# Uses:
# 1. Normal image
# 2. Horizontally flipped image
#
# Then averages probabilities.
# ============================================================

def evaluate_model_tta(
    model,
    loader,
    criterion,
    device
):

    model.eval()


    running_loss = 0.0

    total = 0


    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)


            # -----------------------------------------------
            # NORMAL IMAGE
            # -----------------------------------------------

            outputs_normal = model(
                images
            )


            probabilities_normal = torch.softmax(
                outputs_normal,
                dim=1
            )


            # -----------------------------------------------
            # HORIZONTAL FLIP
            # -----------------------------------------------

            flipped_images = torch.flip(
                images,
                dims=[3]
            )


            outputs_flipped = model(
                flipped_images
            )


            probabilities_flipped = torch.softmax(
                outputs_flipped,
                dim=1
            )


            # -----------------------------------------------
            # AVERAGE
            # -----------------------------------------------

            probabilities = (

                probabilities_normal
                +
                probabilities_flipped

            ) / 2.0


            predictions = probabilities.argmax(
                dim=1
            )


            # Loss calculated using normal prediction

            loss = criterion(
                outputs_normal,
                labels
            )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            total += labels.size(0)


    average_loss = (
        running_loss
        /
        total
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 18. TRAINING STATE
# ============================================================

start_epoch = 0

best_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0


train_loss_history = []

train_accuracy_history = []

val_loss_history = []

val_accuracy_history = []

val_f1_history = []


# ============================================================
# 19. CHECKPOINT RESUME
# ============================================================

if CHECKPOINT_PATH.exists():

    print(
        "\nCheckpoint found."
    )

    print(
        "Resuming previous training..."
    )


    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

        weights_only=False
    )


    saved_classes = checkpoint.get(
        "class_names"
    )


    if saved_classes != class_names:

        raise ValueError(
            "Checkpoint classes do not match the current dataset."
        )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )


    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )


    start_epoch = (
        checkpoint["epoch"] + 1
    )


    best_f1 = checkpoint.get(
        "best_f1",
        -1.0
    )


    best_epoch = checkpoint.get(
        "best_epoch",
        0
    )


    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )


    train_loss_history = checkpoint.get(
        "train_loss_history",
        []
    )


    train_accuracy_history = checkpoint.get(
        "train_accuracy_history",
        []
    )


    val_loss_history = checkpoint.get(
        "val_loss_history",
        []
    )


    val_accuracy_history = checkpoint.get(
        "val_accuracy_history",
        []
    )


    val_f1_history = checkpoint.get(
        "val_f1_history",
        []
    )


else:

    print(
        "\nNo checkpoint found."
    )

    print(
        "Starting NEW training."
    )


# ============================================================
# 20. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    start_epoch,
    MAX_EPOCHS
):

    print(
        f"\nEpoch {epoch + 1}/{MAX_EPOCHS}"
    )

    print("-" * 50)


    train_loss, train_accuracy = train_one_epoch(

        model,

        train_loader,

        criterion,

        optimizer,

        DEVICE
    )


    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _
    ) = evaluate_model(

        model,

        val_loader,

        criterion,

        DEVICE
    )


    scheduler.step(
        val_f1
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    train_loss_history.append(
        train_loss
    )

    train_accuracy_history.append(
        train_accuracy
    )

    val_loss_history.append(
        val_loss
    )

    val_accuracy_history.append(
        val_accuracy
    )

    val_f1_history.append(
        val_f1
    )


    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss       : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Macro F1   : "
        f"{val_f1:.4f}"
    )

    print(
        f"Learning Rate  : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_epoch = (
            epoch + 1
        )

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


        print(
            "Best model saved."
        )


    else:

        epochs_without_improvement += 1


        print(
            f"No improvement: "
            f"{epochs_without_improvement} / "
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_f1":
            best_f1,

        "best_epoch":
            best_epoch,

        "epochs_without_improvement":
            epochs_without_improvement,

        "class_names":
            class_names,

        "train_loss_history":
            train_loss_history,

        "train_accuracy_history":
            train_accuracy_history,

        "val_loss_history":
            val_loss_history,

        "val_accuracy_history":
            val_accuracy_history,

        "val_f1_history":
            val_f1_history
    }


    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )


    print(
        "Checkpoint saved."
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print("\n" + "=" * 70)

        print(
            "EARLY STOPPING"
        )

        print("=" * 70)

        print(
            f"No validation improvement "
            f"for {EARLY_STOPPING_PATIENCE} epochs."
        )

        break


# ============================================================
# 21. TRAINING FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)


print(
    "Best Epoch:",
    best_epoch
)


print(
    "Best Validation Macro F1:",
    f"{best_f1:.4f}"
)


# ============================================================
# 22. LOAD BEST MODEL
# ============================================================

if not BEST_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Best model file was not created."
    )


model.load_state_dict(

    torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE,

        weights_only=True
    )
)


# ============================================================
# 23. NORMAL TEST EVALUATION
# ============================================================

print(
    "\nEvaluating BEST model WITHOUT TTA..."
)


(
    test_loss_normal,
    test_accuracy_normal,
    test_f1_normal,
    true_labels_normal,
    predicted_labels_normal
) = evaluate_model(

    model,

    test_loader,

    criterion,

    DEVICE
)


print("\n" + "=" * 70)

print(
    "NORMAL TEST RESULTS"
)

print("=" * 70)


print(
    f"Test Accuracy : "
    f"{test_accuracy_normal * 100:.2f}%"
)

print(
    f"Test Macro F1 : "
    f"{test_f1_normal:.4f}"
)


# ============================================================
# 24. TTA TEST EVALUATION
# ============================================================

print(
    "\nEvaluating BEST model WITH TTA..."
)


(
    test_loss,
    test_accuracy,
    test_f1,
    true_labels,
    predicted_labels
) = evaluate_model_tta(

    model,

    test_loader,

    criterion,

    DEVICE
)


print("\n" + "=" * 70)

print(
    "TTA TEST RESULTS"
)

print("=" * 70)


print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1 : "
    f"{test_f1:.4f}"
)


# ============================================================
# 25. IMPROVEMENT
# ============================================================

accuracy_change = (
    test_accuracy
    -
    test_accuracy_normal
) * 100


f1_change = (
    test_f1
    -
    test_f1_normal
)


print("\n" + "=" * 70)

print(
    "TTA COMPARISON"
)

print("=" * 70)


print(
    f"Normal Accuracy : "
    f"{test_accuracy_normal * 100:.2f}%"
)

print(
    f"TTA Accuracy    : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Accuracy Change : "
    f"{accuracy_change:+.2f} percentage points"
)


print(
    f"\nNormal Macro F1 : "
    f"{test_f1_normal:.4f}"
)

print(
    f"TTA Macro F1    : "
    f"{test_f1:.4f}"
)

print(
    f"Macro F1 Change : "
    f"{f1_change:+.4f}"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_labels,

    predicted_labels,

    target_names=class_names,

    digits=4,

    zero_division=0
)


print(
    "\nClassification Report:\n"
)

print(
    report
)


with open(

    CLASS_REPORT_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        report
    )


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    true_labels,

    predicted_labels
)


print(
    "\nConfusion Matrix:\n"
)

print(
    cm
)


plt.figure(
    figsize=(14, 12)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Tire Defect Confusion Matrix - TTA"
)


plt.colorbar()


tick_marks = np.arange(
    len(class_names)
)


plt.xticks(
    tick_marks,
    class_names,
    rotation=90
)


plt.yticks(
    tick_marks,
    class_names
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "True Class"
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center"
        )


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_PATH,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 28. SAVE CLASS NAMES
# ============================================================

with open(

    CLASS_NAMES_PATH,

    "w",

    encoding="utf-8"

) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# ============================================================
# 29. TRAINING HISTORY
# ============================================================

epochs_completed = range(
    1,
    len(train_loss_history) + 1
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs_completed,
    train_accuracy_history,
    label="Train Accuracy"
)


plt.plot(
    epochs_completed,
    val_accuracy_history,
    label="Validation Accuracy"
)


plt.plot(
    epochs_completed,
    val_f1_history,
    label="Validation Macro F1"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Score"
)


plt.title(
    "Training History"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    TRAINING_HISTORY_PATH,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING SUMMARY")
print("=" * 70)


print(
    f"Best Epoch                 : "
    f"{best_epoch}"
)


print(
    f"Best Validation Macro F1   : "
    f"{best_f1:.4f}"
)


print(
    f"Normal Test Accuracy       : "
    f"{test_accuracy_normal * 100:.2f}%"
)


print(
    f"Normal Test Macro F1       : "
    f"{test_f1_normal:.4f}"
)


print(
    f"TTA Test Accuracy          : "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"TTA Test Macro F1          : "
    f"{test_f1:.4f}"
)


if test_accuracy >= 0.75:

    print(
        "\nTARGET ACHIEVED: Test accuracy >= 75%"
    )

else:

    print(
        "\nTarget 75% has not been reached yet."
    )


print("\nDone.")