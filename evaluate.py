"""Compatibility entry point for :mod:`scripts.evaluate`."""

from scripts.evaluate import main, make_json_safe, parse_args

__all__ = ["parse_args", "make_json_safe", "main"]

if __name__ == "__main__":
    main()
