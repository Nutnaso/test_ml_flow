import importlib.util
import os
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from shutil import copy

CANDIDATE_PATHS = [
    "02_data_preprocessing.py",
    "mlops_pipeline/script/02_data_preprocessing.py",
    "script/02_data_preprocessing.py",
]

def resolve_preprocess_path() -> str:
    repo_root = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    for rel in CANDIDATE_PATHS:
        p = Path(repo_root) / rel
        if p.exists():
            return str(p.resolve())
    for p in Path(repo_root).rglob("02_data_preprocessing.py"):
        return str(p.resolve())
    raise FileNotFoundError(
        "Cannot locate 02_data_preprocessing.py in repo. Checked: " + ", ".join(CANDIDATE_PATHS)
    )

def load_module_func(py_path: str, func_name: str):
    spec = importlib.util.spec_from_file_location("preprocess_module", py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Invalid module spec for {py_path}"
    spec.loader.exec_module(module)
    fn = getattr(module, func_name)
    return fn

def test_preprocess_image(tmp_path):
    # Arrange: locate preprocessing function
    preproc_path = resolve_preprocess_path()
    preprocess_images = load_module_func(preproc_path, "preprocess_images")

    # Act: set image path (แก้ไขให้ใช้ raw string)
    image_path = Path(r".\tests\Amanita_brunnescens\Amanita_brunnescens_101.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    # Create temporary folder structure for ImageFolder (ต้องมีชื่อ class เป็น folder)
    class_name = "Amanita_brunnescens"
    test_dir = tmp_path / "test_images" / "test" / class_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy test image
    copy(image_path, test_dir / image_path.name)

    # Call preprocess_images
    datasets_dict, dataloaders = preprocess_images(
        data_path=str(tmp_path / "test_images"),
        batch_size=1,
        num_workers=0,
        experiment_name="CI Test Preprocessing",
        resize=(224, 224)  # สามารถปรับขนาดที่นี่
    )

    # Assert: dataloader contains image
    test_loader = dataloaders.get("test")
    assert test_loader is not None

    # Fetch one batch
    for batch_imgs, batch_labels in test_loader:
        assert batch_imgs.shape[0] == 1  # batch size
        assert batch_imgs.shape[1:] == (3, 224, 224)  # C,H,W
        assert batch_labels.shape[0] == 1
        print("✅ Preprocessing test batch successful:", batch_imgs.shape, batch_labels)
        break

if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_preprocess_image(tmp_path=Path(tmp))
