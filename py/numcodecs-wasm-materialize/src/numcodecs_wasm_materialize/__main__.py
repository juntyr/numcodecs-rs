import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import toml

template_pattern = re.compile(r"(%[^%]+%)")

repo_path = Path(__file__).parent.parent.parent.parent.parent
template_path = repo_path / "py" / "numcodecs-wasm-template"
dist_path = repo_path / "py" / "numcodecs-wasm-materialize" / "dist"

codec_pattern = sys.argv[1]

for c in (repo_path / "codecs").iterdir():
    codec = c.name

    if not codec_pattern or re.match(codec_pattern, codec) is None:
        continue

    codec_crate_path = repo_path / "codecs" / codec
    staging_path = (
        repo_path
        / "py"
        / "numcodecs-wasm-materialize"
        / "staging"
        / f"numcodecs-wasm-{codec}"
    )

    templates = {
        "package-suffix": codec,
        "package_suffix": codec.replace("-", "_"),
        "crate-suffix": codec,
        "crate-version": toml.load(codec_crate_path / "Cargo.toml")["package"][
            "version"
        ],
        "codec-id": f"{codec}.rs",
        "codec-path": "".join(c.title() for c in codec.split("-")) + "Codec",
        "CodecName": "".join(c.title() for c in codec.split("-")),
        "wasm-file": codec,
    }

    if codec == "pco":
        templates["codec-path"] = "Pcodec"
        templates["CodecName"] = "Pco"

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

    subprocess.run(
        shlex.split(
            "cargo run -p numcodecs-wasm-builder --"
            f" --crate numcodecs-{templates['crate-suffix']}"
            f" --version {templates['crate-version']}"
            f" --codec {templates['codec-path']}"
            f" --output {staging_path / 'src' / ('numcodecs_wasm_' + templates['package_suffix']) / 'codec.wasm'}"
        ),
        check=True,
    )

    subprocess.run(
        shlex.split("uv sync"),
        check=True,
        cwd=staging_path,
    )
    subprocess.run(
        shlex.split("uv pip install ."),
        check=True,
        cwd=staging_path,
    )
    subprocess.run(
        shlex.split(
            f"uv run python3 {Path(__file__).parent / 'stub.py'} "
            f"{'numcodecs_wasm_' + templates['package_suffix']} src"
        ),
        check=True,
        cwd=staging_path,
    )
    subprocess.run(
        shlex.split(
            f'uv run python3 -c "from {"numcodecs_wasm_" + templates["package_suffix"]} '
            f'import {templates["CodecName"]} as Codec; '
            f'assert Codec.codec_id == {templates["codec-id"]!r}"'
        ),
        check=True,
        cwd=staging_path,
    )
    shutil.rmtree(staging_path / ".venv", ignore_errors=True)

    subprocess.run(
        shlex.split(f"uv build --directory {staging_path} --out-dir {dist_path}"),
        check=True,
    )
