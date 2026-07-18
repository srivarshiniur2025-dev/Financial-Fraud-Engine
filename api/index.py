"""
api/index.py
------------
Vercel Python entrypoint for this Streamlit project.

Vercel auto-detects a root-level app.py and expects a FastAPI/Flask
`app` / `application` / `handler` export. Our Streamlit UI lives in
app.py and must stay unchanged, so this module is the deployment
entrypoint instead.

It launches Streamlit against ../app.py and exposes a small HTTP
handler that Vercel recognizes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Project root (parent of /api)
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_FILE = ROOT_DIR / "app.py"

# Prefer Vercel's PORT when present; fall back for local checks
PORT = int(os.environ.get("PORT", os.environ.get("STREAMLIT_PORT", "8501")))

_streamlit_process: subprocess.Popen | None = None


def _streamlit_command() -> list[str]:
    """Build the Streamlit CLI command that starts app.py."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(PORT),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]


def start_streamlit() -> None:
    """Start Streamlit once per warm serverless instance."""
    global _streamlit_process

    if _streamlit_process is not None and _streamlit_process.poll() is None:
        return

    if not APP_FILE.exists():
        raise FileNotFoundError(f"Streamlit app not found at {APP_FILE}")

    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(PORT)
    env["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    _streamlit_process = subprocess.Popen(
        _streamlit_command(),
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Give Streamlit a moment to bind the port on cold start
    time.sleep(2)


class handler(BaseHTTPRequestHandler):
    """
    Vercel Python Function handler.

    Vercel looks for this top-level `handler` class (not inside app.py).
    """

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Keep Vercel logs quieter
        return

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        try:
            start_streamlit()
            body = (
                "Financial Fraud Intelligence Engine\n\n"
                "Streamlit is starting from app.py via api/index.py.\n"
                f"Configured port: {PORT}\n\n"
                "If you are deploying on Vercel, prefer Dockerfile.vercel "
                "(container mode) so the full Streamlit UI is served on $PORT.\n"
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to launch Streamlit: {exc}\n".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(message)
