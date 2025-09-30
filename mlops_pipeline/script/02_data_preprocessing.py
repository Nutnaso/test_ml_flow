# =============================================
# 02_data_preprocessing.py — Mushroom Images
# =============================================
import os
import json
import mlflow
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import joblib


def preprocess_images(
    data_path: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 2,
    resize: tuple[int, int] = (256, 256),   # <-- เพิ่ม parameter resize
    experiment_name: str = "Mushroom EfficientNet - Data Preprocessing",
):
    """Prepare DataLoaders for train/val/test with augmentation & class balancing."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("data_path", os.path.abspath(data_path))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("resize", resize)

        # Augmentation transforms for train
        train_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Validation/test transforms (no augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        datasets_dict = {}
        dataloaders = {}

        for split in ["train", "val", "test"]:
            split_path = os.path.join(data_path, split)
            if not os.path.exists(split_path):
                continue

            ds = datasets.ImageFolder(
                split_path,
                transform=train_transform if split == "train" else eval_transform
            )

            # =====================
            # WeightedRandomSampler
            # =====================
            if split == "train":
                targets = [s[1] for s in ds.samples]
                class_sample_counts = np.bincount(targets)
                class_weights = 1.0 / class_sample_counts
                sample_weights = [class_weights[t] for t in targets]

                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers)
                mlflow.log_param("class_sample_counts", class_sample_counts.tolist())
            else:
                dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

            datasets_dict[split] = ds
            dataloaders[split] = dl
            mlflow.log_metric(f"{split}_num_images", len(ds))

        # Save class mapping
        if "train" in datasets_dict:
            class_to_idx = datasets_dict["train"].class_to_idx
        else:
            class_to_idx = next(iter(datasets_dict.values())).class_to_idx

        os.makedirs("preprocessing_artifacts", exist_ok=True)
        with open("preprocessing_artifacts/class_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

        # Save transforms config
        transform_config = {
            "resize": resize,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": True
        }
        with open("preprocessing_artifacts/transforms.json", "w", encoding="utf-8") as f:
            json.dump(transform_config, f, indent=2)

        # =====================
        # Save label encoder
        # =====================
        os.makedirs("transformers", exist_ok=True)
        label_encoder_obj = {"classes_": list(class_to_idx.keys())}
        joblib.dump(label_encoder_obj, "transformers/label_encoder.pkl")
        mlflow.log_artifacts("transformers", artifact_path="transformers")

        # Log other artifacts
        mlflow.log_artifacts("preprocessing_artifacts", artifact_path="preprocessing")
        mlflow.log_param("num_classes", len(class_to_idx))
        mlflow.log_param("classes", list(class_to_idx.keys()))

        print("✅ Preprocessing completed. Run ID:", run_id)
        print("Classes mapping:", class_to_idx)
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"preprocessing_run_id={run_id}", file=f)

        return datasets_dict, dataloaders


if __name__ == "__main__":
    # ตัวอย่างเรียกใช้พร้อมกำหนด resize
    preprocess_images(resize=(64, 64))
