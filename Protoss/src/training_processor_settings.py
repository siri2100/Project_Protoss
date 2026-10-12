"""Keep pretrained modality definitions, with v1.3 single-frame DROID video."""
import copy
import json


def load_training_modalities(base_model, tag):
    path = base_model / 'processor_config.json'
    if not path.exists():
        path = base_model / 'processor/processor_config.json'
    settings = json.loads(path.read_text())['processor_kwargs']
    modalities = copy.deepcopy(settings['modality_configs'])
    if tag not in modalities:
        raise ValueError(f'Pretrained processor does not support {tag}')
    modalities[tag]['video']['delta_indices'] = [0]
    return modalities
