from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading


class HttpServer:
    def __init__(self, port):
        self.server = HTTPServer(('', port), SimpleHTTPRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        print(f"HTTP server started on port {self.server.server_port}")

    def stop(self):
        self.server.shutdown()
        self.thread.join()
        print("HTTP server stopped")
