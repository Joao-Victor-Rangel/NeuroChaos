"""Extrai do zip do MICA-MICs (micapipe) só as matrizes das parcelações escolhidas.

  python scripts/prepare_mica.py --parc schaefer200 aparc

Origem: osf.io/j532r, arquivo micapipe_MICs_v1.1.zip (CC0 1.0; citar Royer et al., Sci Data 2022).
"""

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from neurochaos.config import resolve

ZIP_SHA256 = "ac7047195401cdd3d85d4f4cd431ba6b25dce2cda4a51f50e5c8caebd20ab348"
KINDS = ("sc", "edgeLength", "fc", "timeseries", "gd", "mpc")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/derivatives/mica_mics/_download/micapipe_MICs_v1.1.zip")
    ap.add_argument("--out", default="data/derivatives/mica_mics")
    ap.add_argument("--parc", nargs="+", default=["schaefer200", "aparc"])
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    zp, out = resolve(a.zip), resolve(a.out)
    if not a.skip_hash:
        h = sha256(zp)
        if h != ZIP_SHA256:
            raise SystemExit(f"SHA-256 diferente do publicado no OSF: {h}")
        print("SHA-256 confere.")
    pat = re.compile(r"^micapipe/sub-HC\d+/ses-\d+/(anat/geo_dist|anat/micro_profiles|dwi|func)/"
                     r"sub-HC\d+_ses-\d+_space-\w+_atlas-(" + "|".join(map(re.escape, a.parc)) +
                     r")_desc-(" + "|".join(KINDS) + r")\.txt$")
    n = 0
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if not pat.match(info.filename):
                continue
            dst = (out / info.filename).resolve()
            if out.resolve() not in dst.parents:  # proteção contra caminhos fora do destino
                raise SystemExit(f"caminho suspeito no zip: {info.filename}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    print(f"{n} arquivos extraídos em {out / 'micapipe'}")


if __name__ == "__main__":
    main()