#!/usr/bin/env python

# workaround for pipreqs

import subprocess
import re
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def shell(cmd):
    return subprocess.check_output(cmd, shell=True).decode("utf-8")


PKG_PYPI_MAP = {
    "fastchat": "fschat",
    "cv2": "opencv_python",
    "skimage": "scikit-image",
}


def get_installed_pkgs():
    pip_list = shell("pip list").split("\n")
    for line in pip_list:
        line = line.strip()
        if line != "":
            pkg, version = line.split()[:2]
            pkg = pkg.replace("-", "_")
            yield pkg, version


PATTERNS = [
    re.compile(r"^\s*import\s+(\w+)"),
    re.compile(r"^\s*from\s+(\w+)(.\w+)*\s+import"),
]


def get_import_pkgs(file_path):
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()

            for pattern in PATTERNS:
                matches = pattern.match(line)
                if matches is not None:
                    pkg = matches.group(1)
                    if pkg in PKG_PYPI_MAP:
                        eprint(f"[resolve] {pkg} => {PKG_PYPI_MAP[pkg]}")
                        pkg = PKG_PYPI_MAP[pkg]
                    yield pkg
                    break


if __name__ == "__main__":
    installed_pkg_map = {pkg: version for pkg, version in get_installed_pkgs()}

    vis = set()
    import_pkg_list = []
    for file in shell("find . -name '*.py'").split():
        eprint(f"[scan]: {file}")
        for pkg in get_import_pkgs(file):
            if pkg in vis:
                continue
            vis.add(pkg)

            version = installed_pkg_map.get(pkg)
            if version is not None:
                import_pkg_list.append((pkg, version))
            else:
                eprint(f"[ignore]: {pkg}")

    import_pkg_list.sort()
    for pkg, version in import_pkg_list:
        print(f"{pkg}~={version}")
