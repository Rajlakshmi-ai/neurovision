import pandas as pd
import pytest

from model_comparison import load_labels, split_labels


def test_split_labels_keeps_patients_isolated():
    rows = []
    for patient_index in range(20):
        label = patient_index % 2
        for image_index in range(2):
            rows.append(
                {
                    "patient_id": f"patient-{patient_index}",
                    "image_path": f"patient-{patient_index}/image-{image_index}.tif",
                    "mask": label,
                }
            )

    train, validation, test = split_labels(pd.DataFrame(rows), seed=42)
    groups = [set(train.patient_id), set(validation.patient_id), set(test.patient_id)]

    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(train) == 28
    assert len(validation) == 6
    assert len(test) == 6


def test_load_labels_requires_binary_training_columns(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image_path": ["image.tif"], "mask": [0]}).to_csv(labels, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_labels(labels)