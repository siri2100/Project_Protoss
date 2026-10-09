"""Command-line entry point; implementation lives in src/."""
if __package__:
    from .src.prepare_droid import main
else:
    from src.prepare_droid import main


if __name__ == "__main__":
    main()
