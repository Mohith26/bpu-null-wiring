"""Download Winding et al. 2023 Supplementary Data S1 (larval connectome
adjacency matrices + annotations) from a public GitHub mirror into data/."""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/brain-networks/larval-drosophila-connectome/raw/main/Supplementary-Data-S1.zip"
DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    z = DATA / "Supplementary-Data-S1.zip"
    if not z.exists():
        urlretrieve(URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA, members=[m for m in f.namelist() if not m.startswith("__MACOSX")])
    print("extracted to", DATA / "Supplementary-Data-S1")


if __name__ == "__main__":
    main()
