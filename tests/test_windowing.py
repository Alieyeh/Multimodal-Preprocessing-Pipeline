import numpy as np
from mmprep.common.windowing import sliding_windows


def test_sliding_windows_majority_non_null():
    x = np.random.randn(20, 6).astype(np.float32)
    labels = np.array([0, 0] + ["walk"] * 8 + ["sit"] * 10, dtype=object)
    windows, out = sliding_windows(x, labels, window_size=10, step_size=10, null_labels={0})
    assert windows.shape == (2, 6, 10)
    assert out == ["walk", "sit"]


def test_sliding_windows_drop_null_only_windows_and_return_retained_starts():
    x = np.random.randn(15, 2).astype(np.float32)
    labels = np.array(["0"] * 5 + ["walk"] * 5 + ["sit"] * 5, dtype=object)
    windows, out, starts = sliding_windows(
        x,
        labels,
        window_size=5,
        step_size=5,
        null_labels={"0", 0, "null", "transient"},
        return_starts=True,
    )
    assert windows.shape == (2, 2, 5)
    assert out == ["walk", "sit"]
    assert starts == [5, 10]
