"""Command-line entry point; implementation lives in src/."""
if __package__:
    from .src.upload_checkpoint_hf import main
else:
    from src.upload_checkpoint_hf import main


if __name__ == "__main__":
    main()
