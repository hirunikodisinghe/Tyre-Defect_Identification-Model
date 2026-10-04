import torch
from pathlib import Path


CHECKPOINT_PATH = Path(
    r"E:\Hiruni\TireDefectProject\training_checkpoint.pth"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)


print("=" * 70)
print("MODEL CHECKPOINT INFORMATION")
print("=" * 70)


print("\nKeys found:")

for key in checkpoint.keys():
    print(" -", key)


print("\n" + "=" * 70)
print("TRAINING INFORMATION")
print("=" * 70)


keys_to_check = [
    "epoch",
    "best_epoch",
    "best_f1",
    "epochs_without_improvement",
    "class_names",
]


for key in keys_to_check:

    if key in checkpoint:

        print(
            f"{key:30} : "
            f"{checkpoint[key]}"
        )


print("\n" + "=" * 70)
print("LAST RECORDED METRICS")
print("=" * 70)


history_keys = [
    "train_loss_history",
    "train_accuracy_history",
    "val_loss_history",
    "val_accuracy_history",
    "val_f1_history",
]


for key in history_keys:

    if key in checkpoint:

        values = checkpoint[key]

        print(f"\n{key}")

        print(
            f"Number of epochs recorded: "
            f"{len(values)}"
        )

        if len(values) > 0:

            print(
                f"Last value: "
                f"{values[-1]}"
            )

            print(
                f"Best value: "
                f"{max(values) if 'loss' not in key else min(values)}"
            )


print("\n" + "=" * 70)
print("SPECIFIC BEST VALUES")
print("=" * 70)


if "val_f1_history" in checkpoint:

    values = checkpoint[
        "val_f1_history"
    ]

    if values:

        best_f1 = max(values)

        best_f1_epoch = (
            values.index(best_f1)
            + 1
        )

        print(
            f"Best Validation Macro F1 : "
            f"{best_f1:.4f}"
        )

        print(
            f"Best F1 Epoch             : "
            f"{best_f1_epoch}"
        )


if "val_accuracy_history" in checkpoint:

    values = checkpoint[
        "val_accuracy_history"
    ]

    if values:

        best_accuracy = max(
            values
        )

        best_accuracy_epoch = (
            values.index(
                best_accuracy
            )
            + 1
        )

        print(
            f"Best Validation Accuracy  : "
            f"{best_accuracy:.2f}"
        )

        print(
            f"Best Accuracy Epoch        : "
            f"{best_accuracy_epoch}"
        )


print("\nDone.")