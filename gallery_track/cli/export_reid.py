"""CLI shim for ReID export.

This module exists so users can run:

  python -m gallery_track.cli.export_reid ...

It delegates all argument parsing and work to `gallery_track.tools.reid_export`.
"""

from __future__ import annotations

from ..tools.reid_export import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()