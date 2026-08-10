#!/usr/bin/env python3
"""Serve browser test assets with headers required by WASM pthreads."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class IsolatedHandler(SimpleHTTPRequestHandler):
    model_cache: Path | None = None

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        prefix = "/__model_cache__/"
        if self.model_cache is not None and parsed.path.startswith(prefix):
            relative = Path(unquote(parsed.path[len(prefix) :]))
            root = self.model_cache.resolve()
            target = (root / relative).resolve()
            if target.is_relative_to(root):
                return str(target)
            return str(root / "__forbidden_path__")
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-cache",
        type=Path,
        help="Optional read-only directory exposed below /__model_cache__/.",
    )
    args = parser.parse_args()
    browser_dir = Path(__file__).resolve().parents[1] / "browser"
    IsolatedHandler.model_cache = args.model_cache.resolve() if args.model_cache else None
    handler = lambda *inner_args, **kwargs: IsolatedHandler(  # noqa: E731
        *inner_args, directory=str(browser_dir), **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {browser_dir} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
