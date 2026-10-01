"""Bounded-memory archive reading, atomic outputs and reproducible fingerprints."""
from contextlib import contextmanager
import codecs
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import zlib
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=json_default).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, default=json_default, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_directory(path):
    """Publish only a complete new directory; never overwrite an existing analysis."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=path.name + ".partial-", dir=path.parent))
    try:
        yield temp
        temp.rename(path)
    except BaseException:
        import shutil
        shutil.rmtree(temp)
        raise


def byte_chunks(path, chunk_size=65536):
    """Detect real ZIP/gzip/zlib/plain JSON by content, including legacy .json.zip."""
    with open(path, "rb") as f:
        header = f.read(4)
        f.seek(0)
        if header[:2] == b"PK":
            with zipfile.ZipFile(f) as archive:
                names = [n for n in archive.namelist() if not n.endswith("/")]
                if len(names) != 1:
                    raise ValueError("ZIP must contain exactly one sample-array file")
                with archive.open(names[0]) as member:
                    yield from iter(lambda: member.read(chunk_size), b"")
        elif header[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=f) as stream:
                yield from iter(lambda: stream.read(chunk_size), b"")
        elif len(header) >= 2 and header[0] & 15 == 8 and int.from_bytes(header[:2], "big") % 31 == 0:
            decoder = zlib.decompressobj()
            for data in iter(lambda: f.read(chunk_size), b""):
                while data:
                    chunk = decoder.decompress(data, chunk_size)
                    data = decoder.unconsumed_tail
                    if chunk:
                        yield chunk
                if decoder.unused_data:
                    raise ValueError("Unexpected bytes after zlib stream")
            tail = decoder.flush()
            if tail:
                yield tail
            if not decoder.eof:
                raise ValueError("Truncated zlib archive")
        else:
            yield from iter(lambda: f.read(chunk_size), b"")


def iter_sample_rows(path):
    """Incrementally parse a top-level JSON array; at most one physical row is decoded."""
    decoder, utf8 = json.JSONDecoder(), codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    state = "start"
    for chunk in byte_chunks(path):
        buffer += utf8.decode(chunk)
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if state == "end":
                raise ValueError("Trailing data after sample array")
            if state == "start":
                if buffer[0] != "[":
                    raise ValueError("Sample archive must be a top-level JSON array")
                buffer, state = buffer[1:], "first"
                continue
            if state in ("first", "comma") and buffer[0] == "]":
                buffer, state = buffer[1:], "end"
                continue
            if state == "comma":
                if buffer[0] != ",":
                    raise ValueError("Missing comma in sample array")
                buffer, state = buffer[1:], "value"
                continue
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if len(buffer) > 8 << 20:
                    raise ValueError("Malformed archive or individual row larger than 8 MiB")
                break
            if not isinstance(row, (list, dict)):
                raise ValueError("Each physical sample must be a list or mapping")
            yield row
            buffer, state = buffer[end:], "comma"
    buffer += utf8.decode(b"", final=True)
    if state != "end" or buffer.strip():
        raise ValueError("Truncated or malformed sample array")


def write_sample_rows(path, rows, compression_level=1):
    """One zlib array without constructing a giant JSON string or accumulated sample list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    encoder = zlib.compressobj(compression_level)
    try:
        with temp.open("wb") as f:
            f.write(encoder.compress(b"["))
            first = True
            for row in rows:
                payload = json.dumps(row, separators=(",", ":"), default=json_default, allow_nan=False).encode()
                f.write(encoder.compress((b"" if first else b",") + payload))
                first = False
            f.write(encoder.compress(b"]"))
            f.write(encoder.flush())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
