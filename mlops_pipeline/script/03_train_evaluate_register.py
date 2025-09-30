# =============================================
# 03_train_evaluate_register.py — Mushroom EfficientNet Training
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

DEF_EXPERIMENT = "Mushroom - EfficientNet Training"


def _load_artifacts_from_preprocessing_run(run_id: str):
    """โหลด label encoder และ transform config จาก preprocessing run"""
    local_trans = download_artifacts(run_id=run_id, artifact_path="transformers")
    label_encoder_obj = joblib.load(os.path.join(local_trans, "label_encoder.pkl"))

    local_preproc = download_artifacts(run_id=run_id, artifact_path="preprocessing")
    with open(os.path.join(local_preproc, "transforms.json"), "r", encoding="utf-8") as f:
        transform_config = json.load(f)

    return label_encoder_obj, transform_config


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


def train_evaluate_register(preprocessing_run_id: str,
                            dataset_dir: str = "dataset",
                            model_registry_name: str = "Mushroom-EfficientNet",
                            batch_size: int = 16,
                            epochs: int = 1):
    """Train, evaluate and register EfficientNet using preprocessing artifacts."""

    # GPU detection
    gpus = tf.config.list_physical_devices('GPU')
    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"✅ Using device: {device}")

    mlflow.set_experiment(DEF_EXPERIMENT)

    # Load preprocessing artifacts
    label_encoder_obj, transform_config = _load_artifacts_from_preprocessing_run(preprocessing_run_id)
    classes_order = label_encoder_obj.get("classes_", [])
    img_size = tuple(transform_config.get("resize", (256, 256)))

    with tf.device(device):
        with mlflow.start_run(run_name=f"efficientnet_from_{preprocessing_run_id}"):
            mlflow.set_tag("ml.step", "model_training_evaluation")
            mlflow.log_param("preprocessing_run_id", preprocessing_run_id)
            mlflow.log_param("img_size", img_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("num_classes", len(classes_order))

            # Data pipeline
            datagen = ImageDataGenerator(rescale=1./255)

            train_gen = datagen.flow_from_directory(
                os.path.join(dataset_dir, "train"),
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical"
            )
            val_gen = datagen.flow_from_directory(
                os.path.join(dataset_dir, "val"),
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical"
            )
            test_gen = datagen.flow_from_directory(
                os.path.join(dataset_dir, "test"),
                target_size=img_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False
            )

            # Model
            base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.3)(x)
            output = Dense(len(classes_order), activation="softmax")(x)
            model = Model(inputs=base_model.input, outputs=output)

            model.compile(optimizer=Adam(learning_rate=1e-4),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

            # Train
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs
            )

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
            os.makedirs("eval_artifacts", exist_ok=True)
            with open("eval_artifacts/classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report_txt)
            mlflow.log_artifacts("eval_artifacts", artifact_path="evaluation")

            # Log model
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="efficientnet_model",
                registered_model_name=model_registry_name
            )

            print("Training complete. Test accuracy:", acc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_train_evaluate_register.py <preprocessing_run_id> [dataset_dir] [registry_name]")
        sys.exit(1)
    run_id = sys.argv[1]
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    registry_name = sys.argv[3] if len(sys.argv) > 3 else "Mushroom-EfficientNet"
    train_evaluate_register(run_id, dataset_dir, registry_name)
