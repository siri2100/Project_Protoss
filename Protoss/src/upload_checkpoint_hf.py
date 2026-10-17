"""Upload a complete GR00T checkpoint or PEFT adapter run to a private Hugging Face model repo.

Authenticate with `hf auth login` or HF_TOKEN. No token is accepted on the CLI.
Large uploads keep resumable state under the source folder's .cache directory.
"""
import argparse
import os
from pathlib import Path


IGNORE = [".cache/**", "**/.cache/**", ".git/**", "**/.git/**", ".DS_Store", "**/.DS_Store"]


def inventory(folder):
    if not folder.is_dir():
        raise ValueError(f"Checkpoint folder does not exist: {folder}")
    files = [p for p in folder.rglob("*") if p.is_file()
             and not any(part in (".git", ".cache", ".DS_Store") for part in p.relative_to(folder).parts)]
    if (folder / "adapter_inference.json").is_file():
        raise ValueError(f"inference/ contains only a manifest, not the adapter and processor. "
                         f"Upload the complete run folder instead: {folder.parent}")
    if (folder / "adapter/adapter_config.json").is_file():
        from .inference_loading import resolve_inference_paths
        base, adapter, processor = resolve_inference_paths(folder)
        if adapter is None:
            raise ValueError("Adapter run metadata missing; upload the complete training output "
                             "including lora_settings.json or inference/adapter_inference.json")
        for path in (adapter, processor):
            if not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError(f"Adapter/processor is outside upload folder: {path}")
        if not (processor / "statistics.json").is_file():
            raise ValueError(f"Pretrained normalization statistics missing: {processor / 'statistics.json'}")
    elif not any(p.name == "config.json" for p in files):
        raise ValueError("No GR00T config.json or complete PEFT adapter run found; "
                         "specify the training output folder, not inference/")
    if not any(p.name.endswith(".safetensors") or p.name.startswith("pytorch_model") and p.suffix == ".bin"
               for p in files):
        raise ValueError("No model weights found; checkpoint saving may be incomplete")
    if any(p.stat().st_size == 0 for p in files if p.suffix in (".safetensors", ".bin")):
        raise ValueError("Empty weight file found; wait for checkpoint saving to finish")
    return files


def upload(args, api=None, *, validator=inventory, repo_type="model"):
    folder = args.folder.resolve()
    files = validator(folder)
    total = sum(p.stat().st_size for p in files)
    if len(args.repo_id.split("/")) != 2 or any(not part.strip() for part in args.repo_id.split("/")):
        raise ValueError("repo-id must be ACCOUNT/REPOSITORY")
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{args.repo_id}"
    print(f"Source: {folder}\nDestination: {url}\n"
          f"Files: {len(files)}, size: {total / 1e9:.2f} GB", flush=True)
    if args.dry_run:
        print("Dry run: no network request or upload performed")
        return
    if api is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=repo_type, private=True, exist_ok=True)
    info = api.dataset_info(args.repo_id) if repo_type == "dataset" else api.model_info(args.repo_id)
    if not info.private:
        raise ValueError("Destination already exists and is public. Use a private repository.")
    # Upload everything, including optimizer/scheduler/RNG and processor files.
    # Re-running the same folder/repo reuses the SDK's large-upload cache.
    api.upload_large_folder(repo_id=args.repo_id, repo_type=repo_type, folder_path=str(folder),
                            ignore_patterns=IGNORE + (["source/**"] if repo_type == "dataset" else []),
                            num_workers=args.workers)
    remote = set(api.list_repo_files(repo_id=args.repo_id, repo_type=repo_type))
    missing = sorted(p.relative_to(folder).as_posix() for p in files
                     if p.relative_to(folder).as_posix() not in remote)
    if missing:
        # Large-upload metadata can report files as committed even when the
        # selected repository lacks them (e.g. copied cache / changed repo-id).
        # Upload exact inventory paths directly; do not trust that cached state.
        print(f"Remote verification found {len(missing)} missing files. "
              "Repairing with direct uploads (bypassing large-upload cache).", flush=True)
        for relative_path in missing:
            print(f"Uploading missing file: {relative_path}", flush=True)
            api.upload_file(repo_id=args.repo_id, repo_type=repo_type,
                            path_or_fileobj=str(folder / relative_path), path_in_repo=relative_path)
        remote = set(api.list_repo_files(repo_id=args.repo_id, repo_type=repo_type))
        missing = sorted(p.relative_to(folder).as_posix() for p in files
                         if p.relative_to(folder).as_posix() not in remote)
        if missing:
            raise RuntimeError(f"Upload verification still missing {len(missing)} files "
                               f"after direct upload: {missing[:10]}. Destination: {url}")
    print(f"Upload complete: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    upload(args)


if __name__ == "__main__":
    main()
