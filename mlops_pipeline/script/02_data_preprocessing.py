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
from pathlib import Path


def safe_mlflow_path(path: str) -> str:
    """Convert path to valid MLflow URI (cross-platform safe)."""
    abs_path = os.path.abspath(path)

    # 🔹 ป้องกัน path ที่ผิดรูปแบบใน Linux เช่น /C:/...
    if abs_path.startswith("/C:"):
        abs_path = abs_path.replace("/C:", "C:")

    # 🔹 ป้องกัน backslash ของ Windows
    abs_path = abs_path.replace("\\", "/")

    return abs_path


def preprocess_images(
    data_path: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 2,
    resize: tuple[int, int] = (256, 256),
    experiment_name: str = "Mushroom EfficientNet - Data Preprocessing",
):
    """Prepare DataLoaders for train/val/test with augmentation & class balancing."""

    # -----------------------------
    # ตั้ง MLflow tracking URI แบบ cross-platform
    # -----------------------------
    workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    mlruns_dir = Path(workspace_dir) / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file://{safe_mlflow_path(mlruns_dir)}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("data_path", os.path.abspath(data_path))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("resize", resize)

        # -----------------------------
        # Define transforms
        # -----------------------------
        train_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        datasets_dict = {}
        dataloaders = {}

        for split in ["train", "val", "test"]:
            split_path = os.path.join(data_path, split)
            if not os.path.exists(split_path):
                continue

            ds = datasets.ImageFolder(
                split_path,
                transform=train_transform if split == "train" else eval_transform,
            )

            # =====================
            # WeightedRandomSampler
            # =====================
            if split == "train":
                targets = [s[1] for s in ds.samples]
                class_sample_counts = np.bincount(targets)
                class_weights = 1.0 / np.maximum(class_sample_counts, 1)
                sample_weights = [class_weights[t] for t in targets]

                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
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

        # -----------------------------
        # Save class mapping
        # -----------------------------
        if "train" in datasets_dict:
            class_to_idx = datasets_dict["train"].class_to_idx
        else:
            class_to_idx = next(iter(datasets_dict.values())).class_to_idx

        # -----------------------------
        # Save preprocessing artifacts
        # -----------------------------
        preproc_dir = Path(workspace_dir) / "preprocessing_artifacts"
        preproc_dir.mkdir(parents=True, exist_ok=True)

        with open(preproc_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

        transform_config = {
            "resize": resize,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": True,
        }
        with open(preproc_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transform_config, f, indent=2)

        # -----------------------------
        # Save label encoder
        # -----------------------------
        transformers_dir = Path(workspace_dir) / "transformers"
        transformers_dir.mkdir(parents=True, exist_ok=True)

        label_encoder_obj = {"classes_": list(class_to_idx.keys())}
        joblib.dump(label_encoder_obj, transformers_dir / "label_encoder.pkl")

        # -----------------------------
        # Log artifacts safely
        # -----------------------------
        try:
            mlflow.log_artifacts(str(transformers_dir.resolve()), artifact_path="transformers")
            mlflow.log_artifacts(str(preproc_dir.resolve()), artifact_path="preprocessing")
        except PermissionError as e:
            print(f"⚠️ Warning: Skipped artifact logging due to permission issues: {e}")
        except Exception as e:
            print(f"⚠️ Warning: Artifact logging failed: {e}")

        mlflow.log_param("num_classes", len(class_to_idx))
        mlflow.log_param("classes", list(class_to_idx.keys()))

        print("✅ Preprocessing completed. Run ID:", run_id)
        print("Classes mapping:", class_to_idx)

        # For GitHub Actions Output
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"preprocessing_run_id={run_id}", file=f)

        return datasets_dict, dataloaders


if __name__ == "__main__":
    preprocess_images(resize=(64, 64))
