#!/usr/bin/env python3
"""Patch basicsr for torchvision >= 0.17 (idempotent).

basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, a module that
was removed from torchvision 0.17+. The symbol now lives in
`torchvision.transforms.functional`. Rewrite the import in place; safe to re-run.
"""
import importlib.util
import sys

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def main() -> int:
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.submodule_search_locations:
        print("basicsr not installed; nothing to patch.")
        return 0
    root = next(iter(spec.submodule_search_locations))
    target = f"{root}/data/degradations.py"
    try:
        with open(target, encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        print(f"{target} not found; skipping.")
        return 0

    if OLD not in src:
        print(f"basicsr already patched (no '{OLD}' in degradations.py).")
        return 0

    with open(target, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW))
    print(f"Patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
