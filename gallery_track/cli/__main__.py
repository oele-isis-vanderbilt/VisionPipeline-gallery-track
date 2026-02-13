from __future__ import annotations

import argparse
import importlib


def _module_exists(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m gallery_track.cli",
        description=(
            "gallery_track CLI package. Run a subcommand module.\n\n"
            "Examples:\n"
            "  python -m gallery_track.cli.track_video -h\n"
            "  python -m gallery_track.tools.reid_export -h\n"
        ),
    )
    ap.add_argument(
        "--version",
        action="store_true",
        help="Print package version (if available).",
    )

    args, _ = ap.parse_known_args()

    if args.version:
        try:
            import importlib.metadata as im

            print(im.version("gallery-track"))
        except Exception:
            print("unknown")
        return

    # Default: show usage and list available subcommands that are importable.
    ap.print_help()

    candidates = [
        ("track_video", "gallery_track.cli.track_video"),
        ("reid_export", "gallery_track.tools.reid_export"),
        ("build_gallery", "gallery_track.tools.build_gallery"),
    ]

    available = [(name, mod) for name, mod in candidates if _module_exists(mod)]
    if available:
        print("\nAvailable commands:")
        for name, mod in available:
            print(f"  python -m {mod} -h")
    else:
        print("\nNo CLI subcommands found. If you are developing, ensure modules are on PYTHONPATH.")


if __name__ == "__main__":
    main()