import torch
from pathlib import Path


PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)


FILES = [
    PROJECT_DIR / "best_tire_defect_model.pth",
    PROJECT_DIR / "training_checkpoint.pth",
]


for file_path in FILES:

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    if not file_path.exists():

        print("File not found.")
        continue


    checkpoint = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False
    )


    print(
        "Loaded object type:",
        type(checkpoint)
    )


    if isinstance(
        checkpoint,
        dict
    ):

        print("\nKeys inside file:")

        for key in checkpoint.keys():

            print(
                f" - {key}"
            )


        print("\nImportant values:")


        for possible_key in [
            "epoch",
            "best_epoch",
            "best_val_macro_f1",
            "val_macro_f1",
            "accuracy",
            "test_accuracy",
        ]:

            if possible_key in checkpoint:

                print(
                    f"{possible_key}: "
                    f"{checkpoint[possible_key]}"
                )


        # Check whether it contains a model state dictionary
        for possible_key in [
            "model_state_dict",
            "state_dict",
            "model"
        ]:

            if possible_key in checkpoint:

                value = checkpoint[
                    possible_key
                ]

                print(
                    f"\nFound model weights under: "
                    f"{possible_key}"
                )

                if isinstance(
                    value,
                    dict
                ):

                    print(
                        f"Number of model parameters/keys: "
                        f"{len(value)}"
                    )

                    print(
                        "\nFirst 10 model keys:"
                    )

                    for name in list(
                        value.keys()
                    )[:10]:

                        tensor = value[
                            name
                        ]

                        if hasattr(
                            tensor,
                            "shape"
                        ):

                            print(
                                f" - {name}: "
                                f"{tuple(tensor.shape)}"
                            )

                        else:

                            print(
                                f" - {name}"
                            )

                break


    else:

        print(
            "\nThis file appears to contain "
            "only model weights/state."
        )