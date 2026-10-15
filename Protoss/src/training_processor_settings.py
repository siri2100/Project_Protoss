"""Use the pretrained processor without changing settings, statistics or history."""
import copy
import hashlib
import json
from pathlib import Path
import shutil


def processor_path(base_model):
    root = Path(base_model)
    return root if (root / "processor_config.json").is_file() else root / "processor"


def load_training_modalities(base_model, tag):
    settings = json.loads((processor_path(base_model) / "processor_config.json").read_text())
    modalities = copy.deepcopy(settings["processor_kwargs"]["modality_configs"])
    if tag not in modalities:
        raise ValueError(f"Pretrained processor does not support {tag}")
    return modalities


def processor_contract(base_model):
    root = processor_path(base_model)
    files = ["processor_config.json", "statistics.json"]
    if (root / "embodiment_id.json").exists():
        files.append("embodiment_id.json")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def save_original_processor(base_model, output):
    """Copy original artifacts byte-for-byte, avoiding serialization overrides."""
    root = processor_path(base_model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in processor_contract(base_model):
        shutil.copy2(root / name, output / name)


def create_training_dataset(pipeline, save_cfg_dir):
    """Bypass upstream processor overrides and reject normalization changes."""
    from transformers import AutoProcessor
    from gr00t.data.dataset.factory import DatasetFactory

    base = pipeline.config.training.start_from_checkpoint
    processor = AutoProcessor.from_pretrained(processor_path(base))
    original_stats = copy.deepcopy(processor.state_action_processor.statistics)
    tags = {spec.embodiment_tag for spec in pipeline.config.data.datasets}
    missing = tags - set(original_stats)
    if missing:
        raise ValueError(f"Pretrained normalization is missing dataset embodiments: {sorted(missing)}")
    if pipeline.config.data.override_pretraining_statistics:
        raise ValueError("Pretrained processor requires override_pretraining_statistics=False")
    pipeline.processor = processor
    datasets = DatasetFactory(config=pipeline.config).build(processor=processor)
    if processor.state_action_processor.statistics != original_stats:
        raise ValueError("Dataset setup changed pretrained normalization statistics")
    save_original_processor(base, Path(save_cfg_dir).parent / "processor")
    return datasets
