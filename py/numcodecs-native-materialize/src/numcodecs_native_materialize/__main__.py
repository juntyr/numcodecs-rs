import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import toml

template_pattern = re.compile(r"(%[^%]+%)")

repo_path = Path(__file__).parent.parent.parent.parent.parent
template_path = repo_path / "py" / "numcodecs-native-template"
dist_path = repo_path / "py" / "numcodecs-native-materialize" / "dist"

parser = argparse.ArgumentParser(prog="numcodecs_native_materialize")
parser.add_argument(
    "codec",
    type=re.compile,
    help="regex pattern for the codecs to materialize",
)
args = parser.parse_args()

for c in (repo_path / "codecs").iterdir():
    codec = c.name

    if not args.codec or re.match(args.codec, codec) is None:
        continue

    codec_crate_path = repo_path / "codecs" / codec
    staging_path = (
        repo_path
        / "py"
        / "numcodecs-native-materialize"
        / "staging"
        / f"numcodecs-native-{codec}"
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

        for line in c.splitlines():
            for m in template_pattern.finditer(line):
                raise Exception(f"unknown template {m.group(0)}")

        np.parent.mkdir(parents=True, exist_ok=True)

        with np.open("w") as f:
            f.write(c)

    shutil.rmtree(staging_path / ".venv", ignore_errors=True)
    subprocess.run(
        shlex.split("uv venv"),
        check=True,
        cwd=staging_path,
    )
    subprocess.run(
        shlex.split("uv pip install ."),
        check=True,
        cwd=staging_path,
        env={**os.environ, "VIRTUAL_ENV": str(staging_path / ".venv")},
    )
    subprocess.run(
        shlex.split(
            f"uv run --no-sync python3 {Path(__file__).parent / 'stub.py'} "
            + f"{'numcodecs_native_' + templates['package_suffix']} py"
        ),
        check=True,
        cwd=staging_path,
        env={**os.environ, "VIRTUAL_ENV": str(staging_path / ".venv")},
    )
    subprocess.run(
        shlex.split(
            f'uv run --no-sync python3 -c "from {"numcodecs_native_" + templates["package_suffix"]} '
            + f"import {templates['CodecName']} as Codec; "
            + f'assert Codec.codec_id == {templates["codec-id"]!r}, Codec.codec_id"'
        ),
        check=True,
        cwd=staging_path,
        env={**os.environ, "VIRTUAL_ENV": str(staging_path / ".venv")},
    )
    shutil.rmtree(staging_path / ".venv", ignore_errors=True)

    subprocess.run(
        shlex.split(f"uv build --directory {staging_path} --out-dir {dist_path}"),
        check=True,
    )
