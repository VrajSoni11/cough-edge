"""Extract Coswara split archives one date-folder at a time.
Run from project root:  python src\data\extract_coswara.py
"""
import io
import re
import tarfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/coswara"
DST = ROOT / "data/raw/coswara_extracted"
PART = re.compile(r"\.a[a-z]$")


class MultiFile(io.RawIOBase):
    """Reads several files back-to-back as one stream (like `cat part*`)."""
    def __init__(self, paths):
        self.paths, self.i = list(paths), 0
        self.f = open(self.paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.f = open(self.paths[self.i], "rb")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    failed = []
    for folder in tqdm(sorted(p for p in SRC.iterdir() if p.is_dir())):
        marker = DST / f".{folder.name}.done"
        if marker.exists():
            continue
        parts = sorted(f for f in folder.iterdir() if f.is_file() and PART.search(f.name))
        if not parts:
            print(f"\n[skip] no archive parts in {folder.name}")
            continue
        try:
            with tarfile.open(fileobj=io.BufferedReader(MultiFile(parts)), mode="r|gz") as tar:
                tar.extractall(DST, filter="data")
            marker.touch()
        except Exception as e:
            failed.append((folder.name, repr(e)))
    print("\nFailed folders:", failed if failed else "none")


if __name__ == "__main__":
    main()