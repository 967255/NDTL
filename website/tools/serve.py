#!/usr/bin/env python3
"""Serve the static preview without installing JavaScript dependencies."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=8765)
args = parser.parse_args()
directory = Path(__file__).resolve().parents[1] / 'dist'
handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
print(f'Local: http://127.0.0.1:{args.port}', flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
