# =============================================
# 02_data_preprocessing.py — Mushroom Images (Final, CI-ready)
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
from pathlib import Path

def safe_mlflow_path(path: str) -> str:
    """Convert path to valid MLflow URI (cross-platform safe)."""
    abs_path = os.path.abspath(str(path))

    # ป้องกัน path ที่ผิดรูปแบบใน Linux เช่น /C:/...
    if abs_path.startswith("/C:"):
        # แปลงเป็น "C:" (จะเป็น relative แต่เราจะใช้เป็นส่วนของ URI)
        abs_path = abs_path.replace("/C:", "C:")

    # เปลี่ยน backslash เป็น slash เพื่อความสอดคล้อง
    abs_path = abs_path.replace("\\", "/")

    return abs_path

def preprocess_images(
    data_path: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 2,
    resize: tuple[int, int] = (256, 256),
    experiment_name: str = "Mushroom EfficientNet - Data Preprocessing",
):
    """Prepare DataLoaders for train/val/test with augmentation & class balancing.

    ผลลัพธ์:
      - คืนค่า (datasets_dict, dataloaders)
      - Log params/metrics และ artifacts ไปยัง MLflow (mlruns folder หรือ MLFLOW_TRACKING_URI ถ้ากำหนด)
    """

    # -----------------------------
    # Workspace / mlruns setup
    # -----------------------------
    workspace_dir = Path(os.getenv("GITHUB_WORKSPACE", os.getcwd())).resolve()
    mlruns_dir = workspace_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # ถ้ามี MLFLOW_TRACKING_URI ใน environment ให้ใช้ค่านั้น (รองรับ CI)
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        # ถ้า user ระบุเป็น file:// หรือ path ใด ๆ ให้ใช้มัน (แต่ sanitize ถ้าจำเป็น)
        if env_uri.startswith("file://"):
            mlflow_uri = env_uri
        else:
            mlflow_uri = f"file://{safe_mlflow_path(env_uri)}"
    else:
        mlflow_uri = f"file://{safe_mlflow_path(mlruns_dir)}"

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("data_path", os.path.abspath(data_path))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("resize", resize)

        # -----------------------------
        # Transforms
        # -----------------------------
        train_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
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
                # ข้ามถ้าส่วนของข้อมูลไม่ได้มี
                continue

            ds = datasets.ImageFolder(
                split_path,
                transform=train_transform if split == "train" else eval_transform,
            )

            if split == "train":
                targets = [s[1] for s in ds.samples]
                class_sample_counts = np.bincount(targets)
                # ป้องกัน divide-by-zero
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
        # Save class mapping & transforms config
        # -----------------------------
        if "train" in datasets_dict:
            class_to_idx = datasets_dict["train"].class_to_idx
        else:
            # ถ้าไม่มี train ให้ใช้ instance แรก (เช่น ใน test-only runs)
            class_to_idx = next(iter(datasets_dict.values())).class_to_idx

        preproc_dir = workspace_dir / "preprocessing_artifacts"
        transformers_dir = workspace_dir / "transformers"
        preproc_dir.mkdir(parents=True, exist_ok=True)
        transformers_dir.mkdir(parents=True, exist_ok=True)

        # class_to_idx
        with open(preproc_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

        # transforms config
        transform_config = {
            "resize": resize,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": True,
        }
        with open(preproc_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transform_config, f, indent=2, ensure_ascii=False)

        # label encoder
        label_encoder_obj = {"classes_": list(class_to_idx.keys())}
        joblib.dump(label_encoder_obj, transformers_dir / "label_encoder.pkl")

        # -----------------------------
        # Log artifacts to MLflow (หลัก)
        # -----------------------------
        # ใช้ path แบบ resolved (absolute) เพื่อหลีกเลี่ยงพฤติกรรมแปลกๆ ของ path
        local_transformers = str((transformers_dir).resolve())
        local_preproc = str((preproc_dir).resolve())

        try:
            mlflow.log_artifacts(local_transformers, artifact_path="transformers")
            mlflow.log_artifacts(local_preproc, artifact_path="preprocessing")
        except PermissionError as e:
            # ถ้า MLflow ไม่สามารถเขียน artifacts ได้ (permission) ให้พยายามคัดลอกเข้าโฟลเดอร์ mlruns ของ run โดยตรง
            print(f"⚠️ Warning: mlflow.log_artifacts permission error: {e}")
        except Exception as e:
            print(f"⚠️ Warning: mlflow.log_artifacts failed: {e}")

        # -----------------------------
        # Fallback: ensure artifacts exist under mlruns/<exp_id>/<run_id>/artifacts/
        # เพื่อให้ mlflow.artifacts.download_artifacts() ทำงานได้แน่นอน
        # -----------------------------
        try:
            # ระบุ path ภายใน mlruns สำหรับ run นี้
            # mlruns structure: mlruns/<experiment_id>/<run_id>/artifacts/
            run_artifact_base = Path(safe_mlflow_path(mlruns_dir)) / str(exp_id) / str(run_id) / "artifacts"
            run_artifact_base.mkdir(parents=True, exist_ok=True)

            # คัดลอก (หรือ merge) โฟลเดอร์ artifacts เข้า run folder
            target_preproc = run_artifact_base / "preprocessing"
            target_transformers = run_artifact_base / "transformers"

            # คัดลอกด้วย dirs_exist_ok=True (Python>=3.8)
            shutil.copytree(local_preproc, target_preproc, dirs_exist_ok=True)
            shutil.copytree(local_transformers, target_transformers, dirs_exist_ok=True)
        except Exception as e:
            # ถ้า copy พังก็ไม่ให้ fail ทั้ง pipeline — แต่ log ไว้เพื่อ debug
            print(f"⚠️ Warning: failed to copy artifacts into mlruns run folder: {e}")

        # -----------------------------
        # Log metadata
        # -----------------------------
        mlflow.log_param("num_classes", len(class_to_idx))
        mlflow.log_param("classes", list(class_to_idx.keys()))

        # บันทึก run_id สำหรับ GitHub Actions output
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"preprocessing_run_id={run_id}", file=f)

        print("✅ Preprocessing completed. Run ID:", run_id)
        print("Classes mapping:", class_to_idx)


        # -----------------------------
        # 🔹 Save artifacts directly into MLflow run folder
        # -----------------------------
        exp_id = run.info.experiment_id
        run_artifacts_path = mlruns_dir / str(exp_id) / str(run_id) / "artifacts"
        (run_artifacts_path / "transformers").mkdir(parents=True, exist_ok=True)
        (run_artifacts_path / "preprocessing").mkdir(parents=True, exist_ok=True)

        # คัดลอก artifacts เข้า run folder โดยตรง
        shutil.copytree(preproc_dir, run_artifacts_path / "preprocessing", dirs_exist_ok=True)
        shutil.copytree(transformers_dir, run_artifacts_path / "transformers", dirs_exist_ok=True)

        # log ผ่าน MLflow (ถ้าไม่มี permission ก็ข้ามได้)
        try:
            mlflow.log_artifacts(str(run_artifacts_path / "transformers"), artifact_path="transformers")
            mlflow.log_artifacts(str(run_artifacts_path / "preprocessing"), artifact_path="preprocessing")
        except Exception as e:
            print(f"⚠️ Warning: mlflow.log_artifacts failed: {e}")

        # ตรวจสอบว่ามีไฟล์จริง
        print("🧩 Artifact structure created under:", run_artifacts_path)
        for sub in ["transformers", "preprocessing"]:
            subdir = run_artifacts_path / sub
            print("  ", subdir, "contains:", os.listdir(subdir))
        
        return datasets_dict, dataloaders



if __name__ == "__main__":
    # ตัวอย่างเรียกใช้
    preprocess_images(resize=(64, 64))
