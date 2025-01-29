# From: https://github.com/huggingface/candle/blob/main/candle-pyo3/stub.py
# See: https://raw.githubusercontent.com/huggingface/tokenizers/main/bindings/python/stub.py  # noqa: E501

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import (
    Any,
    Optional,
)

INDENT = " " * 4

LINK_USE_PATTERN = re.compile(r"\[([^]]+)\]")
LINK_DEF_PATTERN = re.compile(r"^\[([^]]+)\]: (.+)$")
INLINE_MATH_PATTERN = re.compile(r"(:?`\$)|(:?\$`)")


def do_indent(text: str, indent: str) -> str:
    return text.replace("\n", f"\n{indent}")


def extract_docstring(
    obj: Any,
    cls: Optional[type] = None,
) -> Optional[str]:
    docstring = getattr(obj, "__doc__", None)

    if docstring is not None or cls is None:
        return docstring

    for m in cls.mro():
        entry = getattr(m, obj.__name__, None)

        if entry is None:
            continue

        docstring = getattr(entry, "__doc__", None)

        if docstring is not None:
            break

    return docstring


def process_docstring(
    docstring: str,
    module: str,
) -> str:
    doc_lines = docstring.splitlines()

    link_defs = dict()
    i = 0
    while i < len(doc_lines):
        match = LINK_DEF_PATTERN.fullmatch(doc_lines[i])
        if match is not None and not match.group(1).startswith("^"):
            link_defs[match.group(1)] = match.group(2)
            doc_lines.pop(i)
        else:
            i += 1

    math_block = False

    for i, line in enumerate(doc_lines):
        if line.strip() == "```math" and not math_block:
            line = line.replace("```math", "$$")
            math_block = True

        if line.strip() == "```" and math_block:
            line = line.replace("```", "$$")
            math_block = False

        if not math_block:
            line = INLINE_MATH_PATTERN.sub("$", line)

        matches = list(LINK_USE_PATTERN.finditer(line))
        for match in matches[::-1]:
            next = None if match.end() >= len(line) else line[match.end()]
            if next == "[" or next == "(":
                continue
            source = match.group(1)
            if source.startswith("^"):
                continue
            target = link_defs.get(source)
            if target is not None:
                line = (
                    line[: match.start()]
                    + f"[{source}]({target})"
                    + line[match.end() :]  # noqa: E203
                )
            elif not source.startswith("Accessed: ") and source.startswith("`"):
                source = source.replace("::", ".")
                target = source.strip("`")
                line = (
                    line[: match.start()]
                    + f"[{source}][{module}.{target}]"
                    + line[match.end() :]  # noqa: E203
                )

        doc_lines[i] = line

    while len(doc_lines) > 0 and len(doc_lines[-1].strip()) == 0:
        doc_lines.pop()

    return "\n".join(doc_lines)


def extract_and_process_docstring(
    obj: Any,
    cls: Optional[type] = None,
    module: Optional[str] = None,
) -> Optional[str]:
    docstring = extract_docstring(obj, cls=cls)

    if docstring is None:
        return None

    if module is None:
        module = obj.__module__

    docstring = process_docstring(docstring, module)

    if len(docstring.strip()) == 0:
        return None

    return docstring


def stub_function(
    func: Any,
    indent: str,
    module: str,
    cls: Optional[type] = None,
    name: Optional[str] = None,
    text_signature: Optional[str] = None,
) -> str:
    if name is None:
        name = func.__name__

    if text_signature is None:
        text_signature = func.__text_signature__
    if text_signature is not None:
        text_signature = (
            text_signature.replace("$self", "self")
            .replace("$cls", "cls")
            .lstrip()
            .rstrip()
        )

    docstring = extract_and_process_docstring(func, cls=cls, module=module)

    string = f"{indent}def {name}{text_signature}:"

    if docstring is not None:
        string += "\n"
        string += f'{indent + INDENT}r"""\n'
        string += f"{indent + INDENT}{do_indent(docstring, indent + INDENT)}\n"
        string += f'{indent + INDENT}"""\n'
        string += f"{indent + INDENT}...\n\n"
    else:
        string += " ...\n\n"

    return string


def stub_module(
    obj: Any,
    indent: str,
) -> str:
    string = ""

    imports = set()

    for name, member in inspect.getmembers(obj):
        if inspect.ismodule(member):
            continue
        if name.startswith("_"):
            continue
        if getattr(obj, "__all__", None) is None or name in obj.__all__:
            string += stub_member(
                member, indent, module=obj.__name__, imports=imports, name=name
            )

    if len(imports) > 0:
        string = "\n".join(f"import {i}" for i in sorted(imports)) + "\n\n" + string

    if getattr(obj, "__all__", None) is not None:
        string = f"__all__ = {sorted(set(obj.__all__))!r}\n\n" + string

    return string


def stub_class(
    obj: type,
    indent: str,
    imports: set,
) -> str:
    bases = [base for base in obj.__bases__ if base is not object]

    if len(bases) > 0:
        inherit = []
        for base in bases:
            if base.__module__ not in sys.modules:
                continue
            if getattr(sys.modules[base.__module__], base.__name__, None) is None:
                continue
            imports.add(base.__module__)
            inherit.append(f"{base.__module__}.{base.__name__}")
        inherit = f"({', '.join(inherit)})"
    else:
        inherit = ""

    string = f"class {obj.__name__}{inherit}:"
    indent += INDENT

    body = ""

    docstring = extract_and_process_docstring(obj, cls=obj)
    if docstring is not None:
        body += f'{indent}r"""\n{indent}{do_indent(docstring, indent)}\n{indent}"""\n\n'

    # __init__ signature
    if obj.__text_signature__:
        text_signature = (
            obj.__text_signature__.replace("$self", "self")
            .replace("$cls", "cls")
            .lstrip()
            .rstrip()
        )
        body += f"{indent}def __init__{text_signature}: ...\n\n"
    elif getattr(obj, "__init__", None) is not None:
        body += stub_member(
            obj.__init__,
            indent=indent,
            module=obj.__module__,
            imports=imports,
            name="__init__",
            cls=obj,
        )

    for name, member in inspect.getmembers(obj):
        if not name.startswith("_"):
            body += stub_member(
                member,
                indent=indent,
                module=obj.__module__,
                imports=imports,
                name=name,
                cls=obj,
            )

    if len(body) == 0:
        string += " ...\n\n"
    else:
        string += f"\n{body}\n\n"

    return string


def stub_member(
    obj: Any,
    indent: str,
    module: str,
    imports: set,
    name: Optional[str] = None,
    cls: Optional[type] = None,
) -> str:
    if inspect.ismodule(obj):
        return stub_module(obj, indent)

    if inspect.isclass(obj):
        return stub_class(obj, indent, imports)

    if inspect.isbuiltin(obj):
        stub = stub_function(obj, indent, module, cls=cls)
        return f"{indent}@staticmethod\n{stub}"

    if inspect.ismethoddescriptor(obj):
        return stub_function(obj, indent, module, cls=cls)

    if name is not None and isinstance(obj, type(lambda x: x)):
        return stub_function(
            obj,
            indent,
            module,
            name=name,
            text_signature=str(inspect.signature(obj)),
            cls=cls,
        )

    if name is not None:
        return f"{indent}{name} = {obj!r}\n\n"

    raise Exception(f"Object {obj} is not supported")


def write_module_stubs(module: Any, directory: Path) -> None:
    pyi_content = (
        stub_module(
            module,
            indent="",
        ).strip()
        + "\n"
    )

    directory.mkdir(parents=True, exist_ok=True)

    filename = directory / "__init__.pyi"

    with open(filename, "w") as f:
        f.write(pyi_content)

    for name, member in inspect.getmembers(module):
        if not inspect.ismodule(member):
            continue

        if member.__name__ != f"{module.__name__}.{name}":
            continue

        if name.startswith("_"):
            continue

        write_module_stubs(member, directory / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("output")
    args = parser.parse_args()

    module = importlib.import_module(args.module)

    write_module_stubs(module, Path(args.output).joinpath(*args.module.split(".")))
