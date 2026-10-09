"""Command-line entry point; implementation lives in src/."""
if __package__:
    from .src.prepare_groot_dataset import main
else:
    from src.prepare_groot_dataset import main


if __name__ == "__main__":
    main()
