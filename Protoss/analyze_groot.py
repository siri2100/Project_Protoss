"""CLI entry point for GR00T diagnostics."""
if __package__:
    from .src.analyze_groot import main
else:
    from src.analyze_groot import main

if __name__ == '__main__':
    main()
