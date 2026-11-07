"""Train and compare three transfer-learning classifiers for NeuroVision.

The script deliberately keeps the data split and training controls identical for
all backbones. It is intended for an apples-to-apples comparison with the
current classifier, rather than for production inference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


MODEL_NAMES = ("ResNet50", "EfficientNetB0", "MobileNetV2")


def set_seed(seed: int) -> None:
    """Make the split and model initialization repeatable where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Load and validate image paths and binary tumor labels."""
    frame = pd.read_csv(csv_path)
    required = {"patient_id", "image_path", "mask"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    frame = frame[["patient_id", "image_path", "mask"]].copy()
    frame["mask"] = pd.to_numeric(frame["mask"], errors="raise").astype(int)
    if not frame["mask"].isin([0, 1]).all():
        raise ValueError("mask must contain only binary labels 0 and 1")
    frame = frame.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return frame


def split_labels(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 70/15/15 stratified splits without sharing patients between them."""
    patients = frame[["patient_id", "mask"]].drop_duplicates("patient_id")
    train_patients, remainder = train_test_split(
        patients, test_size=0.30, random_state=seed, stratify=patients["mask"]
    )
    val_patients, test_patients = train_test_split(
        remainder, test_size=0.50, random_state=seed, stratify=remainder["mask"]
    )

    train = frame[frame["patient_id"].isin(train_patients["patient_id"])].copy()
    val = frame[frame["patient_id"].isin(val_patients["patient_id"])].copy()
    test = frame[frame["patient_id"].isin(test_patients["patient_id"])].copy()
    return train, val, test


def make_dataset(frame: pd.DataFrame, root: Path, image_size: int, batch_size: int, shuffle: bool):
    paths = [str(root / path) for path in frame["image_path"]]
    labels = tf.keras.utils.to_categorical(frame["mask"].to_numpy(), num_classes=2)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def read_tiff(path):
        image_path = path.decode("utf-8")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read MRI image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (image_size, image_size)).astype(np.float32)

    def read_image(path, label):
        image = tf.numpy_function(read_tiff, [path], tf.float32)
        image.set_shape([image_size, image_size, 3])
        return image, label

    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(len(frame), seed=17, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_classifier(name: str, image_size: int) -> tf.keras.Model:
    """Build the same two-class head on each requested ImageNet backbone."""
    builders = {
        "ResNet50": tf.keras.applications.ResNet50,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
    }
    backbone = builders[name](
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input((image_size, image_size, 3), name="mri")
    if name == "ResNet50":
        processed = tf.keras.applications.resnet50.preprocess_input(inputs)
    elif name == "EfficientNetB0":
        processed = tf.keras.applications.efficientnet.preprocess_input(inputs)
    else:
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    features = backbone(processed, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.30)(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="tumor_class")(features)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model: tf.keras.Model, dataset, labels: pd.Series) -> dict[str, float]:
    probabilities = model.predict(dataset, verbose=0)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    truth = labels.to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "auc": float(roc_auc_score(truth, probabilities)),
    }


def train_and_compare(args: argparse.Namespace) -> list[dict]:
    set_seed(args.seed)
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_labels(root / args.labels)
    missing_images = [root / path for path in frame["image_path"] if not (root / path).is_file()]
    if missing_images:
        sample = ", ".join(str(path) for path in missing_images[:3])
        raise FileNotFoundError(
            f"{len(missing_images)} labeled MRI files are missing. "
            f"Place the TCGA folders under {root}. Examples: {sample}"
        )
    train, val, test = split_labels(frame, args.seed)
    train_ds = make_dataset(train, root, args.image_size, args.batch_size, shuffle=True)
    val_ds = make_dataset(val, root, args.image_size, args.batch_size, shuffle=False)
    test_ds = make_dataset(test, root, args.image_size, args.batch_size, shuffle=False)

    results = []
    for name in MODEL_NAMES:
        print(f"\nTraining {name} ({len(train)} train / {len(val)} val / {len(test)} test images)")
        model = build_classifier(name, args.image_size)
        checkpoint = output_dir / f"{name.lower()}-best.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True),
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
        metrics = evaluate_model(model, test_ds, test["mask"])
        metrics.update({"model": name, "epochs_trained": len(history.history["loss"]), "weights": str(checkpoint)})
        results.append(metrics)
        print(json.dumps(metrics, indent=2))
        tf.keras.backend.clear_session()

    results.sort(key=lambda item: item["auc"], reverse=True)
    report = {
        "dataset": {"labels": str(root / args.labels), "train": len(train), "validation": len(val), "test": len(test)},
        "seed": args.seed,
        "models": results,
        "best_model": results[0]["model"],
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output_dir / "comparison.csv", index=False)
    print(f"\nBest model by AUC: {report['best_model']}")
    print(f"Reports written to {output_dir}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root containing the dataset folders")
    parser.add_argument("--labels", default="data_mask.csv", help="CSV with patient_id, image_path, and mask columns")
    parser.add_argument("--output-dir", default="model_comparison", help="Directory for weights and comparison reports")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_compare(parse_args())