"""Resolve current trainset/testset directories and existing train/test data."""
def split_path(root, split):
    path = root / {"train": "trainset", "test": "testset"}[split]
    legacy = root / split
    return legacy if not (path / "meta/info.json").exists() and (legacy / "meta/info.json").exists() else path


def require_unsplit(report):
    """Refuse old partitions rather than omit their test episodes."""
    if set(report.get("splits", {})) != {"train"}:
        raise ValueError("Dataset has a train/test partition; re-prepare all source episodes in a new directory")
