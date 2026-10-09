"""Command-line entry point; implementation lives in src/."""
if __package__:
    from .src.eval_groot_models import main
else:
    from src.eval_groot_models import main


if __name__ == "__main__":
    main()
