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

DEF_EXPERIMENT = "Mushroom - EfficientNet Training"

# --- Auto detect image channels from dataset ---
def detect_image_channels(dataset_dir: str):
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    
    first_class = next(os.walk(train_dir))[1][0]
    first_image_path = next(os.walk(os.path.join(train_dir, first_class)))[2][0]
    img_path = os.path.join(train_dir, first_class, first_image_path)
    
    img = Image.open(img_path)
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
        return path
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
        from pathlib import Path
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

    label_encoder_path = os.path.join(local_transformers, "label_encoder.pkl")
    transform_config_path = os.path.join(local_preproc, "transforms.json")

    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError("❌ label_encoder.pkl not found in preprocessing artifacts")

    label_encoder_obj = joblib.load(label_encoder_path)

    if os.path.exists(transform_config_path):
        with open(transform_config_path, "r", encoding="utf-8") as f:
            transform_config = json.load(f)
    else:
        print(f"⚠️ transforms.json not found at {transform_config_path}")
        transform_config = {}

    return label_encoder_obj, transform_config

# --- Plot and log confusion matrix ---
def _plot_and_log_confusion(cm: np.ndarray, classes: list, artifact_dir="eval_artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)
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
    path = os.path.join(artifact_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifacts(artifact_dir, artifact_path="evaluation")

# --- Main training function ---
def train_evaluate_register(preprocessing_run_id: str,
                            dataset_dir: str = "mlops_pipeline/dataset",
                            model_registry_name: str = "Mushroom-EfficientNet",
                            batch_size: int = 16,
                            epochs: int = 1):
    gpus = tf.config.list_physical_devices('GPU')
    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"✅ Using device: {device}")

    mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
    mlflow.set_experiment(DEF_EXPERIMENT)

    label_encoder_obj, transform_config = _load_artifacts_from_preprocessing_run(preprocessing_run_id)
    classes_order = label_encoder_obj.get("classes_", [])
    if not classes_order:
        raise ValueError("❌ classes_ is empty in label_encoder.pkl")

    img_size = tuple(transform_config.get("resize", (256, 256)))

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

            # Data pipeline
            datagen = ImageDataGenerator(rescale=1./255)
            train_gen = datagen.flow_from_directory(os.path.join(dataset_dir, "train"),
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    color_mode=color_mode)
            val_gen = datagen.flow_from_directory(os.path.join(dataset_dir, "val"),
                                                  target_size=img_size,
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  color_mode=color_mode)
            test_gen = datagen.flow_from_directory(os.path.join(dataset_dir, "test"),
                                                   target_size=img_size,
                                                   batch_size=batch_size,
                                                   class_mode="categorical",
                                                   shuffle=False,
                                                   color_mode=color_mode)

            # Model
            base_model = EfficientNetB0(weights=weights,
                                        include_top=False,
                                        input_shape=(*img_size, channels))
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.3)(x)
            output = Dense(len(classes_order), activation="softmax")(x)
            model = Model(inputs=base_model.input, outputs=output)

            model.compile(optimizer=Adam(learning_rate=1e-4),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

            # Train
            history = model.fit(train_gen,
                                validation_data=val_gen,
                                epochs=epochs)

            # Log metrics
            for epoch in range(epochs):
                mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
                mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
                if "accuracy" in history.history:
                    mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
                if "val_accuracy" in history.history:
                    mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

            # Save loss curve
            os.makedirs("eval_artifacts", exist_ok=True)
            plt.figure(figsize=(8, 5))
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Val Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig("eval_artifacts/loss_curve.png")
            plt.close()
            mlflow.log_artifact("eval_artifacts/loss_curve.png", artifact_path="evaluation")

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
            with open("eval_artifacts/classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report_txt)
            mlflow.log_artifacts("eval_artifacts", artifact_path="evaluation")

            # Register model
            mlflow.tensorflow.log_model(model=model,
                                        artifact_path="efficientnet_model",
                                        registered_model_name=model_registry_name)

            print(f"🎉 Training complete. Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_train_evaluate_register.py <preprocessing_run_id> [registry_name]")
        sys.exit(1)

    run_id = sys.argv[1]
    registry_name = sys.argv[2] if len(sys.argv) > 2 else "Mushroom-EfficientNet"
    train_evaluate_register(run_id, model_registry_name=registry_name)
