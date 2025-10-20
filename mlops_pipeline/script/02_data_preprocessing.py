# =============================================
# 02_data_preprocessing.py — Mushroom Images (Final)
# =============================================
import os
import json
import mlflow
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import joblib
import shutil


def safe_path(path: str) -> str:
    """Return absolute cross-platform safe path."""
    abs_path = os.path.abspath(path)
    if abs_path.startswith("/C:"):
        abs_path = abs_path.replace("/C:", "C:")
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
    # MLflow Tracking setup
    # -----------------------------
    workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    mlruns_dir = os.path.join(workspace_dir, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{safe_path(mlruns_dir)}")
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
        # Save artifacts
        # -----------------------------
        class_to_idx = (datasets_dict.get("train") or
                        next(iter(datasets_dict.values()))).class_to_idx

        preproc_dir = os.path.join(workspace_dir, "preprocessing_artifacts")
        transformers_dir = os.path.join(workspace_dir, "transformers")
        os.makedirs(preproc_dir, exist_ok=True)
        os.makedirs(transformers_dir, exist_ok=True)

        with open(os.path.join(preproc_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)

        transform_config = {
            "resize": resize,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": True,
        }
        with open(os.path.join(preproc_dir, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump(transform_config, f, indent=2)

        label_encoder_obj = {"classes_": list(class_to_idx.keys())}
        joblib.dump(label_encoder_obj, os.path.join(transformers_dir, "label_encoder.pkl"))

        # -----------------------------
        # Copy artifacts inside mlruns directory for safe logging
        # -----------------------------
        run_artifact_base = os.path.join(mlruns_dir, run.info.experiment_id, run_id, "artifacts")
        os.makedirs(run_artifact_base, exist_ok=True)

        local_preproc = os.path.join(run_artifact_base, "preprocessing")
        local_trans = os.path.join(run_artifact_base, "transformers")
        shutil.copytree(preproc_dir, local_preproc, dirs_exist_ok=True)
        shutil.copytree(transformers_dir, local_trans, dirs_exist_ok=True)

        mlflow.log_artifacts(local_trans, artifact_path="transformers")
        mlflow.log_artifacts(local_preproc, artifact_path="preprocessing")

        # -----------------------------
        # Log metadata
        # -----------------------------
        mlflow.log_param("num_classes", len(class_to_idx))
        mlflow.log_param("classes", list(class_to_idx.keys()))

        print("✅ Preprocessing completed. Run ID:", run_id)
        print("Classes mapping:", class_to_idx)

        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"preprocessing_run_id={run_id}", file=f)

        return datasets_dict, dataloaders


if __name__ == "__main__":
    preprocess_images(resize=(64, 64))
