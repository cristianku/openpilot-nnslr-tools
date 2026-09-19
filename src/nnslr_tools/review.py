# [preannotation] - START
"""Self-contained review page and a loopback-only-by-default report server."""
from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside, resolve_data_root


def write_review(directory: Path, rows: list[dict], run_id: str) -> None:
    template=files('nnslr_tools').joinpath('review.html').read_text()
    payload=json.dumps({'run_id':run_id,'frames':rows},ensure_ascii=True,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (directory/'index.html').write_text(template.replace('__NNSLR_DATA__',payload),encoding='utf-8')


def public_file(directory: Path, url: str) -> Path | None:
    name=unquote(urlsplit(url).path)
    if name in {'/','/index.html'}:
        relative=Path('index.html')
    elif re.fullmatch(r'/images/[0-9]{8}\.jpg',name):
        relative=Path(name.lstrip('/'))
    else:
        return None
    path=directory/relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        return None
    return path


def latest_report(root: Path, route: str) -> Path:
    parse_route_id(route)
    base=_inside(root,root/'derived/preannotations'/route)
    pointer=_inside(base,base/'latest.json')
    if not pointer.is_file():
        raise PreannotationError(f'no report; run: nnslr preannotate --route {route}')
    run_id=json.loads(pointer.read_text()).get('run_id','')
    if not isinstance(run_id,str) or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}',run_id):
        raise PreannotationError('invalid latest report pointer')
    directory=_inside(base,base/run_id)
    if public_file(directory,'/') is None:
        raise PreannotationError('report index is missing')
    return directory


def serve_review(route: str, *, data_root: Path | None=None, host: str='127.0.0.1', port: int=8765) -> None:
    directory=latest_report(resolve_data_root(data_root),route)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=public_file(directory,self.path)
            if path is None:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(path.stat().st_size))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.end_headers()
            with path.open('rb') as source:
                while block:=source.read(65536): self.wfile.write(block)
        def log_message(self,*args):
            pass
    try:
        with ThreadingHTTPServer((host,port),Handler) as server:
            print(f'Review: http://{host}:{server.server_port}/',flush=True)
            print(f'Report: {directory / "index.html"}',flush=True)
            print('Ctrl+C to stop. Review edits stay in your browser until you export the JSONL.',flush=True)
            try: server.serve_forever()
            except KeyboardInterrupt: pass
    except OSError as exc:
        raise PreannotationError(f'cannot start review server: {exc}') from exc
# [preannotation] - END
