"""Fetch the original Middlebury frames, authors' source and SceneFlow checkpoint."""
import hashlib
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "6e93ed2169bd858dbb43033988563f3b0bb49506"
WEIGHTS = "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
SCENES = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets"
CONFIG = json.loads((ROOT/"configs/benchmark.json").read_text())


def checksum(file):
    with file.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def fetch(url, dest):
    if dest.is_file():
        return
    dest.parent.mkdir(exist_ok=True, parents=True)
    part = dest.with_suffix(dest.suffix+".part")
    with urllib.request.urlopen(url, timeout=150) as remote, part.open("wb") as local:
        shutil.copyfileobj(remote, local)
    part.replace(dest)


def prepare(cache, names):
    cache.mkdir(parents=True, exist_ok=True)
    upstream = cache/"raft"
    if not upstream.exists():
        subprocess.run(["git","clone","https://github.com/princeton-vl/RAFT-Stereo.git",str(upstream)], check=True)
        subprocess.run(["git","-C",str(upstream),"checkout","--detach",PIN],check=True)
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=PIN:
        raise RuntimeError(f"Expected upstream commit {PIN} in {upstream}")
    ckpt = cache/"raftstereo-sceneflow.pth"
    if not ckpt.is_file():
        archive = cache/"raft-models.zip"
        fetch(WEIGHTS, archive)
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith("raftstereo-sceneflow.pth"))
            part = ckpt.with_suffix(".part")
            with z.open(member) as src, part.open("wb") as dst:
                shutil.copyfileobj(src,dst)
            part.replace(ckpt)
    assets = [ckpt]
    for name in names:
        for file in ("im0.png","im1.png","disp0.pfm","disp1.pfm"):
            dst = cache/"stereodata"/name/file
            fetch(f"{SCENES}/{name}-perfect/{file}", dst)
            assets.append(dst)
    actual = {p.relative_to(cache).as_posix():checksum(p) for p in assets}
    manifest = ROOT/"configs/assets-sha256.json"
    if manifest.is_file():
        expected = json.loads(manifest.read_text())
        for path, digest in actual.items():
            if expected.get(path) != digest:
                raise ValueError(f"Asset content differs from recorded run: {path}")
    return actual
