# =============================================
# 03_train_evaluate_register.py — Mushroom EfficientNet Training (Auto Channel)
# =============================================
import os
import sys
import json
import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import matplotlib.pyplot as plt
from mlflow.artifacts import download_artifacts
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from mlflow.exceptions import MlflowException
from PIL import Image
from pathlib import Path

DEF_EXPERIMENT = "Mushroom - EfficientNet Training"

# --- Auto detect image channels from dataset ---
def detect_image_channels(dataset_dir: str):
    train_dir = Path(dataset_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    first_class = next(train_dir.iterdir())
    first_image_path = next(first_class.iterdir())

    img = Image.open(first_image_path)
    mode = img.mode
    if mode == "L":  # grayscale
        channels = 1
        weights = None
        color_mode = "grayscale"
    elif mode in ["RGB", "RGBA"]:
        channels = 3
        weights = "imagenet"
        color_mode = "rgb"
    else:
        channels = 3
        weights = "imagenet"
        color_mode = "rgb"

    print(f"✅ Detected image mode: {mode}, using channels={channels}, weights={weights}")
    return channels, weights, color_mode

# --- Safe artifact download ---
def _safe_download_artifact(run_id: str, artifact_path: str):
    try:
        path = download_artifacts(run_id=run_id, artifact_path=artifact_path)
        print(f"✅ Downloaded artifact '{artifact_path}' to {path}")
        return Path(path)
    except MlflowException as e:
        print(f"⚠️ Warning: Artifact '{artifact_path}' not found in run {run_id}")
        print("⚠️ Exception:", e)
        return None

# --- Load preprocessing artifacts ---
def _load_artifacts_from_preprocessing_run(run_id: str):
    local_transformers = _safe_download_artifact(run_id, "transformers")
    local_preproc = _safe_download_artifact(run_id, "preprocessing")

    # fallback local mlruns
    if not local_transformers or not local_preproc:
        workspace = Path(os.getenv("GITHUB_WORKSPACE", os.getcwd())).resolve()
        mlruns_dir = workspace / "mlruns"
        print("🔍 Fallback: Searching in local mlruns directory...")
        for exp_dir in mlruns_dir.glob("*"):
            candidate = exp_dir / run_id / "artifacts"
            if candidate.exists():
                local_transformers = candidate / "transformers"
                local_preproc = candidate / "preprocessing"
                print(f"✅ Fallback: Using local artifacts at {candidate}")
                break
        else:
            print("❌ No local fallback artifacts found in mlruns.")

    label_encoder_path = local_transformers / "label_encoder.pkl"
    transform_config_path = local_preproc / "transforms.json"

    if not label_encoder_path.exists():
        raise FileNotFoundError("❌ label_encoder.pkl not found in preprocessing artifacts")

    label_encoder_obj = joblib.load(label_encoder_path)

    if transform_config_path.exists():
        with open(transform_config_path, "r", encoding="utf-8") as f:
            transform_config = json.load(f)
    else:
        print(f"⚠️ transforms.json not found at {transform_config_path}")
        transform_config = {}

    return label_encoder_obj, transform_config

# --- Helper: log artifact safely ---
def _safe_log_artifact(file_path: Path, artifact_path: str = "evaluation"):
    """Safely log artifact across OS (Windows, Linux, macOS, GitHub Actions)."""
    try:
        file_path = Path(file_path)

        # Ensure absolute path & readable
        if not file_path.exists():
            print(f"⚠️ File not found, skipping: {file_path}")
            return
        if not os.access(file_path, os.R_OK):
            print(f"⚠️ No read permission for {file_path}, skipping log.")
            return

        # Resolve workspace
        workspace = Path(os.getenv("GITHUB_WORKSPACE", Path.cwd())).resolve()

        # Normalize path (fix 'C:\' or '/C:' issues)
        file_str = str(file_path).replace("\\", "/")
        if ":" in file_str and not platform.system().lower().startswith("win"):
            # Remove Windows-style drive letter on Linux/macOS
            file_str = file_str.split(":", 1)[-1]
            file_str = file_str.lstrip("/")

        safe_path = workspace / Path(file_str)
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file into safe workspace location if outside
        if not str(file_path).startswith(str(workspace)):
            temp_path = workspace / "eval_artifacts" / file_path.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(file_path, temp_path)
            safe_path = temp_path

        rel_path = os.path.relpath(safe_path, workspace)
        print(f"📦 Logging artifact: {rel_path} → {artifact_path}")
        mlflow.log_artifact(str(safe_path), artifact_path=artifact_path)

    except Exception as e:
        print(f"⚠️ Failed to log artifact safely ({file_path}): {e}")



# --- Plot and log confusion matrix ---
def _plot_and_log_confusion(cm: np.ndarray, classes: list, artifact_dir="eval_artifacts"):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    path = artifact_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    _safe_log_artifact(path)

# --- Main training function ---
def train_evaluate_register(preprocessing_run_id: str,
                            dataset_dir: str = "mlops_pipeline/dataset",
                            model_registry_name: str = "Mushroom-EfficientNet",
                            batch_size: int = 16,
                            epochs: int = 1):
    gpus = tf.config.list_physical_devices('GPU')
    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"✅ Using device: {device}")

    mlruns_path = Path.cwd() / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_path.resolve()}")
    mlflow.set_experiment(DEF_EXPERIMENT)

    label_encoder_obj, transform_config = _load_artifacts_from_preprocessing_run(preprocessing_run_id)
    classes_order = label_encoder_obj.get("classes_", [])
    if not classes_order:
        raise ValueError("❌ classes_ is empty in label_encoder.pkl")

    img_size = tuple(transform_config.get("resize", (256, 256)))
    eval_dir = Path("eval_artifacts")
    eval_dir.mkdir(parents=True, exist_ok=True)

    with tf.device(device):
        with mlflow.start_run(run_name=f"efficientnet_from_{preprocessing_run_id}"):
            mlflow.set_tag("ml.step", "model_training_evaluation")
            mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
            mlflow.log_param("img_size", img_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("num_classes", len(classes_order))

            # Detect channels
            channels, weights, color_mode = detect_image_channels(dataset_dir)
            if channels == 1:
                print("⚠️ Grayscale dataset detected, training from scratch (weights=None)")

            # Data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )
            val_datagen = ImageDataGenerator(rescale=1./255)
            test_datagen = ImageDataGenerator(rescale=1./255)

            train_gen = train_datagen.flow_from_directory(
                Path(dataset_dir) / "train",
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical",
                color_mode=color_mode
            )
            val_gen = val_datagen.flow_from_directory(
                Path(dataset_dir) / "val",
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical",
                color_mode=color_mode
            )
            test_gen = test_datagen.flow_from_directory(
                Path(dataset_dir) / "test",
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
                color_mode=color_mode
            )

            # Model
            base_model = EfficientNetB0(weights=None,
                                        include_top=False,
                                        input_shape=(*img_size, channels))
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.3)(x)
            output = Dense(len(classes_order), activation="softmax")(x)
            model = Model(inputs=base_model.input, outputs=output)

            model.compile(optimizer=Adam(1e-4),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

            # Train
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            # Log metrics
            for epoch in range(epochs):
                mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
                mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
                if "accuracy" in history.history:
                    mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
                if "val_accuracy" in history.history:
                    mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

            # Save loss curve
            plt.figure(figsize=(8, 5))
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Val Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            loss_curve_path = eval_dir / "loss_curve.png"
            plt.savefig(loss_curve_path)
            plt.close()
            _safe_log_artifact(loss_curve_path)

            # Evaluate
            loss, acc = model.evaluate(test_gen)
            mlflow.log_metric("test_loss", float(loss))
            mlflow.log_metric("test_accuracy", float(acc))

            # Confusion matrix
            y_true = test_gen.classes
            y_pred = np.argmax(model.predict(test_gen), axis=1)
            cm = confusion_matrix(y_true, y_pred)
            _plot_and_log_confusion(cm, classes=list(classes_order))

            # Classification report
            report_txt = classification_report(y_true, y_pred, target_names=list(classes_order))
            report_path = eval_dir / "classification_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_txt)
            _safe_log_artifact(report_path)

            # Register model
            mlflow.tensorflow.log_model(model=model,
                                        artifact_path="efficientnet_model",
                                        registered_model_name=model_registry_name)

            print(f"🎉 Training complete. Test accuracy: {acc:.4f}")

# --- Main entry point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_train_evaluate_register.py <preprocessing_run_id> [registry_name]")
        sys.exit(1)

    run_id = sys.argv[1]
    registry_name = sys.argv[2] if len(sys.argv) > 2 else "Mushroom-EfficientNet"
    train_evaluate_register(run_id, model_registry_name=registry_name)
