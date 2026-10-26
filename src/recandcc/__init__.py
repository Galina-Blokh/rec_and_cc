"""RecAndCC: detect calls/videos, record them locally, transcribe them locally."""


def main() -> None:
    import sys

    from .cli import main as cli_main

    sys.exit(cli_main())
