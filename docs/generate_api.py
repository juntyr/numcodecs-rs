"""Generate the code reference pages and navigation.
Adapted from: https://github.com/proxystore/proxystore/blob/main/docs/generate_api.py # noqa: E501
Licensed under the MIT license
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

import numcodecs_wasm

nav = mkdocs_gen_files.Nav()

src = Path(numcodecs_wasm.__file__).parent.parent

print(src)

for path in sorted(src.rglob("numcodecs_wasm_*/**/*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)
    parts = tuple(".".join(parts[: i + 1]) for i in range(len(parts)))

    if parts[-1].endswith("__init__"):
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].endswith("__main__"):
        continue

    if len(parts) == 1:
        nav_parts = parts
    elif len(parts) == 2:
        nav_parts = (parts[1],)
    else:
        nav_parts = tuple([parts[1]] + [p.split(".")[-1] for p in parts[2:]])
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {parts[-1]}\n\n::: {parts[-1]}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
