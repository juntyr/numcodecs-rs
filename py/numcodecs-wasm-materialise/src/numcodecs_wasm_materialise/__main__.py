import re
from pathlib import Path

import toml

CODEC = "identity"  # TODO: obtain from argparse

template_pattern = re.compile(r"(%[^%]+%)")

repo_path = Path(__file__).parent.parent.parent.parent.parent
template_path = repo_path / "py" / "numcodecs-wasm-template"
staging_path = (
    repo_path
    / "py"
    / "numcodecs-wasm-materialise"
    / "staging"
    / f"numcodecs-wasm-{CODEC}"
)

codec_crate_path = repo_path / "codecs" / CODEC

templates = {
    "package-suffix": CODEC,
    "package_suffix": CODEC.replace("-", "_"),
    "crate-suffix": CODEC,
    "crate-version": toml.load(codec_crate_path / "Cargo.toml")["package"]["version"],
    "CodecName": "".join(c.title() for c in CODEC.split("-")),
    "wasm-file": CODEC,
}

for p in template_path.glob("**/*"):
    if not p.is_file():
        continue

    np = str(staging_path / p.relative_to(template_path))
    for t, r in templates.items():
        np = np.replace(f"%{t}%", r)
    np = Path(np)

    for m in template_pattern.finditer(str(np)):
        raise Exception(f"unknown template {m.group(0)}")

    with p.open() as f:
        c = f.read()

    for t, r in templates.items():
        c = c.replace(f"%{t}%", r)

    for l in c.splitlines():
        for m in template_pattern.finditer(l):
            raise Exception(f"unknown template {m.group(0)}")

    np.parent.mkdir(parents=True, exist_ok=True)

    with np.open("w") as f:
        f.write(c)
