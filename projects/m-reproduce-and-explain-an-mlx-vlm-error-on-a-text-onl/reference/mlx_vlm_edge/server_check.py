import http.server
import threading
import time
import urllib.request


class DummyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_headless_smoke_test(port=8080):
    server = http.server.HTTPServer(("127.0.0.1", port), DummyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
        status = req.status
        content = req.read()
        success = (status == 200 and content == b"OK")
    except Exception:
        success = False
    finally:
        server.shutdown()
        server.server_close()
    return success
