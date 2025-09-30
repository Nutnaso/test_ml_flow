# =============================================
# 01_data_validation.py  — Mushroom Images
# =============================================
import os
import json
import mlflow
from PIL import Image
import numpy as np
from collections import Counter

def _validate_images(base_dir: str, splits=("train", "val", "test")):
    stats = {}
    all_classes = set()
    corrupted_files = []

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        split_stats = {"num_images": 0, "class_distribution": {}, "image_sizes": []}

        if not os.path.exists(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            all_classes.add(class_name)

            image_count = 0
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        split_stats["image_sizes"].append(img.size)  # (width, height)
                        image_count += 1
                except Exception:
                    corrupted_files.append(fpath)

            split_stats["class_distribution"][class_name] = image_count
            split_stats["num_images"] += image_count

        # สรุปสถิติขนาดภาพ
        if split_stats["image_sizes"]:
            arr = np.array(split_stats["image_sizes"])
            widths, heights = arr[:, 0], arr[:, 1]
            split_stats["size_summary"] = {
                "min_width": int(widths.min()),
                "max_width": int(widths.max()),
                "mean_width": float(widths.mean()),
                "min_height": int(heights.min()),
                "max_height": int(heights.max()),
                "mean_height": float(heights.mean()),
            }

        stats[split] = split_stats

    return stats, sorted(all_classes), corrupted_files


def validate_image_data(
    data_path: str = "dataset",
    experiment_name: str = "Mushroom EfficientNet - Data Validation",
):
    """Validate image dataset structure and log to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_path", os.path.abspath(data_path))

        # 1) ตรวจสอบ dataset
        stats, classes, corrupted = _validate_images(data_path)

        # 2) log metrics
        for split, s in stats.items():
            mlflow.log_metric(f"{split}_num_images", s["num_images"])
            for cname, count in s["class_distribution"].items():
                mlflow.log_metric(f"{split}_{cname}_count", count)

        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("classes", ",".join(classes))
        mlflow.log_metric("num_corrupted_files", len(corrupted))

        # 3) save artifacts
        os.makedirs("validation_artifacts", exist_ok=True)
        with open("validation_artifacts/data_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        with open("validation_artifacts/class_distribution.json", "w", encoding="utf-8") as f:
            json.dump(dict(Counter(classes)), f, indent=2)
        with open("validation_artifacts/corrupted_files.json", "w", encoding="utf-8") as f:
            json.dump(corrupted, f, indent=2)

        mlflow.log_artifacts("validation_artifacts", artifact_path="data_validation")

        print("✅ Validation completed. Run ID:", run_id)
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"validation_run_id={run_id}", file=f)


if __name__ == "__main__":
    validate_image_data()
