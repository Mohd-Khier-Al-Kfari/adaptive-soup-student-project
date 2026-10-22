"""Explicit acquisition, adapted from HAR run_datasets.py:269.
Copyright (C) 2024 Moh'd Khier Al Kfari; adaptations 2026. GPL-3.0-or-later.
No network access or filesystem writes occur during import.
Author: Moh'd Khier Al Kfari.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]

def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must remain inside student project: {path}")
    return path

def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_archive(path, expected=""):
    actual = checksum(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
    return actual

def extract_archive(path, destination):
    """Reject traversal and symlinks; extract only into the student project."""
    destination = inside(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlink unsupported: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)

def acquire(dataset, settings, *, allowed=False):
    if not allowed:
        raise ValueError("Download requires explicit --download or interactive consent.")
    folder = inside(ROOT / "data" / "original_datasets" / dataset)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / settings["archive"]
    if not path.exists():
        partial = path.with_suffix(".part")
        try:
            request = urllib.request.Request(settings["url"], headers={"User-Agent": "Student-HAR/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                if getattr(response, "status", 200) != 200:
                    raise ValueError(f"Download status: {response.status}")
                shutil.copyfileobj(response, out)
            verify_archive(partial, settings.get("sha256", ""))
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
    digest = verify_archive(path, settings.get("sha256", ""))
    extracted = folder / "extracted"
    extract_archive(path, extracted)
    # PAMAP2/MHEALTH distribute a nested dataset ZIP. RealWorld sensor ZIPs stay intact.
    for nested in list(extracted.rglob("*.zip")):
        if dataset != "RealWorld" and ("pamap2_dataset" in nested.stem.lower() or "mhealth" in nested.stem.lower()):
            verify_archive(nested)
            extract_archive(nested, nested.parent)
    info = {"dataset": dataset, "url": settings["url"], "archive": str(path),
            "sha256": digest, "trusted_checksum_supplied": bool(settings.get("sha256"))}
    (folder / "download_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return extracted

def extract_existing(dataset, settings):
    """Extract an archive already present locally, without network access."""
    folder = inside(ROOT / "data" / "original_datasets" / dataset)
    path = folder / settings["archive"]
    if not path.is_file():
        raise FileNotFoundError(f"Local archive not found: {path}")
    digest = verify_archive(path, settings.get("sha256", ""))
    extracted = folder / "extracted"
    extract_archive(path, extracted)
    for nested in list(extracted.rglob("*.zip")):
        if dataset != "RealWorld" and ("pamap2_dataset" in nested.stem.lower() or "mhealth" in nested.stem.lower()):
            verify_archive(nested)
            extract_archive(nested, nested.parent)
    info = {"dataset": dataset, "url": settings["url"], "archive": str(path),
            "sha256": digest, "trusted_checksum_supplied": bool(settings.get("sha256")),
            "source": "existing_local_archive"}
    (folder / "download_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return extracted

