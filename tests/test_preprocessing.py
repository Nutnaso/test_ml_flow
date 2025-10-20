import importlib.util
import os
from pathlib import Path
from shutil import copy
import mlflow

# สำหรับ PyTorch
import torch
from torchvision import datasets, transforms

# Paths ที่อาจจะมีไฟล์ preprocess
CANDIDATE_PATHS = [
    "02_data_preprocessing.py",
    "mlops_pipeline/script/02_data_preprocessing.py",
    "script/02_data_preprocessing.py",
]

def resolve_preprocess_path() -> str:
    """ค้นหาไฟล์ 02_data_preprocessing.py ใน repo"""
    repo_root = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    for rel in CANDIDATE_PATHS:
        p = Path(repo_root) / rel
        if p.exists():
            return str(p.resolve())
    for p in Path(repo_root).rglob("02_data_preprocessing.py"):
        return str(p.resolve())
    raise FileNotFoundError(
        "Cannot locate 02_data_preprocessing.py. Checked: " + ", ".join(CANDIDATE_PATHS)
    )

def load_module_func(py_path: str, func_name: str):
    """โหลดฟังก์ชันจากไฟล์ Python"""
    spec = importlib.util.spec_from_file_location("preprocess_module", py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Invalid module spec for {py_path}"
    spec.loader.exec_module(module)
    return getattr(module, func_name)

def run_preprocessing_test(tmp_path: Path):
    """ทดสอบ preprocessing ด้วย image test เดียว"""
    os.environ["MLFLOW_TRACKING_URI"] = str(tmp_path / "mlruns")
    os.environ["MLFLOW_ARTIFACTS_DIR"] = str(tmp_path / "preprocessing_artifacts")
 
    preproc_path = resolve_preprocess_path()
    preprocess_images = load_module_func(preproc_path, "preprocess_images")

    # ไฟล์ภาพตัวอย่าง
    image_path = Path("./tests/Amanita_brunnescens/Amanita_brunnescens_101.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    # สร้างโฟลเดอร์ชั่วคราวสำหรับ ImageFolder
    class_name = "Amanita_brunnescens"
    test_dir = tmp_path / "test_images" / "test" / class_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # คัดลอกไฟล์ภาพไปยัง temp folder
    copy(image_path, test_dir / image_path.name)

    # --- ตั้ง MLflow ให้ใช้ folder ชั่วคราว ---
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # เรียกใช้ preprocess
    datasets_dict, dataloaders = preprocess_images(
        data_path=str(tmp_path / "test_images"),
        batch_size=1,
        num_workers=0,
        experiment_name="CI Test Preprocessing",
        resize=(224, 224)
    )

    # ตรวจสอบ dataloader
    test_loader = dataloaders.get("test")
    if test_loader is None:
        raise RuntimeError("Test dataloader not created.")

    # ดู batch แรก
    for batch_imgs, batch_labels in test_loader:
        assert batch_imgs.shape[0] == 1
        assert batch_imgs.shape[1:] == (3, 224, 224)
        assert batch_labels.shape[0] == 1
        print("✅ Preprocessing test batch successful:", batch_imgs.shape, batch_labels)
        break

if __name__ == "__main__":
    # ใช้ temporary directory แทน tmp_path ของ pytest
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
       run_preprocessing_test(tmp_path=Path(tmp))

