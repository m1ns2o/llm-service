#!/usr/bin/env python3
"""Serve browser test assets with headers required by WASM pthreads."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class IsolatedHandler(SimpleHTTPRequestHandler):
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
    args = parser.parse_args()
    browser_dir = Path(__file__).resolve().parents[1] / "browser"
    handler = lambda *inner_args, **kwargs: IsolatedHandler(  # noqa: E731
        *inner_args, directory=str(browser_dir), **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {browser_dir} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
