"""One-line recipes for setup, training, inference and evaluation."""

import sys

sys.dont_write_bytecode = True

from src.run_recipes import main


if __name__ == "__main__":
    sys.exit(main())
