"""No-cache static server for the web/ frontend (Cloud Run).

GitHub Pages forces a fixed ~10-minute HTML cache we can't change, which is why
edits kept appearing stale. Serving the same static site here lets us set
Cache-Control: no-store on every response, so the browser NEVER serves an old
shell — each load fetches the current files. Trade-off vs Pages: a ~2-3s cold
start when idle (scale-to-zero); in exchange, zero cache confusion.
"""

import http.server
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Force revalidation of everything, HTML included.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, *args):  # quiet
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    http.server.HTTPServer(("0.0.0.0", port), NoCacheHandler).serve_forever()
