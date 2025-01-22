import threading
import http.server
import socketserver

class HttpServer:

    def singleStaticPage(self):
        self.handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", self.port), self.handler) as httpd:
            self.httpd = httpd
            print("Starting http server on port", self.port)
            self.httpd.serve_forever()

    def __init__(self, port):
        self.httpserver = None
        self.port = port

    def __del__(self):
        self.httpd.shutdown()

    def start(self):
        if self.httpserver is None:
            self.httpserver = threading.Thread(target=self.singleStaticPage)
            self.httpserver.start()

    def stop(self):
        if(self.httpd != None):
            self.httpd.shutdown()
            print("httpd shut down")
        self.httpserver.join()
        self.httpserver = None
