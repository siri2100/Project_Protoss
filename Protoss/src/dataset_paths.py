"""Resolve current trainset/testset directories and existing train/test data."""
def split_path(root, split):
    path = root / {"train": "trainset", "test": "testset"}[split]
    legacy = root / split
    return legacy if not (path / "meta/info.json").exists() and (legacy / "meta/info.json").exists() else path
