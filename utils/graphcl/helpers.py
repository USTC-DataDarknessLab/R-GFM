import random
from typing import Sequence, Tuple


def make_uid(dataset_name: str, center_id: int) -> Tuple[str, int]:
    return dataset_name, int(center_id)


def assert_uid_alignment(
    uids1: Sequence,
    uids2: Sequence,
    step: int,
    interval: int = 50,
    sample_size: int = 16
):
    if step % interval != 0:
        return

    assert len(uids1) == len(uids2), f"UID length mismatch: {len(uids1)} vs {len(uids2)}"

    if len(uids1) == 0:
        return

    idx = list(range(len(uids1)))
    random.shuffle(idx)
    idx = idx[: min(sample_size, len(idx))]

    for i in idx:
        assert uids1[i] == uids2[i], f"UID mismatch at {i}: {uids1[i]} vs {uids2[i]}"


__all__ = ["make_uid", "assert_uid_alignment"]
