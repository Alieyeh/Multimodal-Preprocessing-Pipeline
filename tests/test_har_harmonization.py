import pandas as pd

from mmprep.common.windowing import sliding_windows
from mmprep.har.base import harmonize_har_labels, make_har_windows


def test_harmonize_har_labels_maps_unmatched_to_other_and_preserves_null():
    df = pd.DataFrame({"label": ["walking", "weird_activity", "0", "null"]})
    out = harmonize_har_labels(df, dataset_name="wisdm", label_map={"walking": "walking"})
    assert out["original_label"].tolist() == ["walking", "weird_activity", "0", "null"]
    assert out["label"].tolist() == ["walking", "other", "0", "null"]
    assert (out["label_schema_name"] == "har_unified_v1").all()


def test_harmonize_har_labels_allows_configured_schema_name():
    df = pd.DataFrame({"label": ["walking"]})
    out = harmonize_har_labels(
        df,
        dataset_name="wisdm",
        label_map={"walking": "walking"},
        label_schema_name="har_project_custom_v2",
    )
    assert out["label_schema_name"].tolist() == ["har_project_custom_v2"]


def test_sliding_windows_excludes_string_null_labels_from_majority_vote():
    signal = pd.DataFrame(
        {
            "acc_x": [0.0] * 6,
            "acc_y": [0.0] * 6,
        }
    ).to_numpy()
    labels = pd.Series(["0", "walking", "walking", "null", "walking", "walking"]).to_numpy()
    windows, out_labels = sliding_windows(signal, labels, window_size=4, step_size=2, null_labels={0, "null", "transient"})
    assert windows.shape[0] == 2
    assert out_labels == ["walking", "walking"]


def test_make_har_windows_preserves_original_label_for_majority_vote():
    df = pd.DataFrame(
        {
            "acc_x": [0.0] * 6,
            "acc_y": [0.0] * 6,
            "gyro_x": [0.0] * 6,
            "gyro_y": [0.0] * 6,
            "label": ["walking", "walking", "walking", "running", "running", "running"],
            "original_label": ["walk_fast", "walk_fast", "walk_fast", "jog", "jog", "jog"],
            "label_schema_name": ["har_unified_v1"] * 6,
            "signal_qc_summary": [""] * 6,
        }
    )
    windows, rows = make_har_windows(
        df=df,
        dataset_name="wisdm",
        subject_id="s1",
        source_name="record.csv",
        sampling_rate_hz=1,
        channel_schema=["acc_x", "acc_y", "gyro_x", "gyro_y"],
        kind="supervised",
        null_labels={"0", "null", "transient"},
    )
    assert windows.shape[0] == 1
    assert rows[0]["label_or_event"] == "walking"
    assert rows[0]["original_label"] == "walk_fast"
