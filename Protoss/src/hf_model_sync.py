"""Synchronize complete training runs with Hugging Face using setup-time login."""
from argparse import ArgumentParser, Namespace
from pathlib import Path


def training_destination(args, api=None):
    if getattr(args, "no_hf_upload", False):
        return None
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    account = api.whoami()["name"]
    repo = getattr(args, "hf_repo_id", None) or f"{account}/{args.output_dir.name}"
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo)
    if len(repo.split("/")) != 2:
        raise ValueError("--hf-repo-id must be ACCOUNT/REPOSITORY")
    print(f"Automatic upload destination: https://huggingface.co/{repo}", flush=True)
    return repo


def upload_training_run(folder, repo):
    from .upload_checkpoint_hf import upload
    upload(Namespace(folder=Path(folder), repo_id=repo, workers=2, dry_run=False))


def download_for_server(args, remaining, downloader=None):
    """Download before launching the server; never fall back to stale local weights."""
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    # Explicit local paths avoid relying on absolute paths saved on the training host.
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--base-model-path", required=True, type=Path)
    paths, _ = parser.parse_known_args(remaining)
    model = paths.model_path.resolve()
    base = paths.base_model_path.resolve()
    if model == base or model.is_relative_to(base) or base.is_relative_to(model):
        raise ValueError("Model run and base model must use separate, non-nested folders")
    for repo, revision, destination in (
        (args.hf_repo_id, args.hf_revision, model),
        (args.hf_base_repo_id, args.hf_base_revision, base),
    ):
        print(f"Downloading {repo}@{revision} to {destination}", flush=True)
        downloader(repo_id=repo, revision=revision, local_dir=str(destination))
    from .inference_loading import resolve_inference_paths
    _, adapter, _ = resolve_inference_paths(model, base)
    if adapter is None:
        raise ValueError("Downloaded repository must contain a complete PEFT run")
    return remaining
