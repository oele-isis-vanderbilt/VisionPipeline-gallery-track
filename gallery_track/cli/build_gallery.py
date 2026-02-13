from __future__ import annotations

"""CLI entrypoint: `python -m gallery_track.cli.build_gallery`.

This module is intentionally thin: it delegates to the implementation in
`gallery_track.tools.build_gallery`.

Note:
  The build-gallery tool is optional (GUI dependency). If the underlying tool
  cannot be imported, we raise a clear error message.
"""


def main() -> None:
    try:
        from ..tools.build_gallery import main as _main
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "build_gallery is not available. Install the optional GUI dependencies "
            "(e.g., PyQt5) or add the tool implementation under gallery_track/tools." 
            f"\nImport error: {e}"
        )
    _main()


if __name__ == "__main__":
    main()