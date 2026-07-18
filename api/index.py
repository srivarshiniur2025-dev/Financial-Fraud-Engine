"""
api/index.py (legacy serverless stub)

This file is NOT used for the production Vercel deployment anymore.

Why it existed:
  Vercel treated root app.py as FastAPI/Flask and expected an `app` export.
  This handler was a workaround that only returned plain text while starting
  Streamlit in the background — browsers never received the Streamlit UI.

Current deployment:
  Dockerfile.vercel runs `streamlit run app.py` on $PORT and Vercel routes
  all traffic to that container. Visiting the Vercel URL loads Streamlit.

Kept in the repo only so older docs/links do not break. Safe to ignore.
"""

from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            b"This serverless stub is unused.\n"
            b"Deploy with Dockerfile.vercel so Streamlit serves the UI on $PORT.\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return
