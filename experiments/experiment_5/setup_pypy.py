"""Install PyPy and build experiment 5's virtual environment. Windows and Linux.

    conda run -n lndp python setup_pypy.py           # install + verify
    conda run -n lndp python setup_pypy.py --check   # verify only, install nothing

Run it with ANY Python; it only downloads and unpacks, then hands off to PyPy itself.
Nothing is installed into the venv -- experiment 5's search imports nothing outside
the standard library, which is the property that makes running it under PyPy possible
at all. The venv exists so the interpreter has a stable path (`.venv-pypy/`) that
`bench.py`, `test_equivalence.py` and the docs can all name.

WHY THIS FILE EXISTS RATHER THAN A LINE IN THE README. The remote GPU box
(`shoveler-l.cs.ucl.ac.uk`) only ever sees the git repo, has no conda channel for
PyPy, and runs csh. A script that picks the right archive for the platform, verifies
the download and prints the exact command to run afterwards is the difference between
reproducing this in one command and reproducing it in a support thread.

⚠️ PyPy IS NOT ALWAYS THE FAST CHOICE. Above ~14 program inputs CPython wins, by up to
5x -- the truth-table integers get big enough that CPython's hand-written C big-int
code beats anything the JIT can do about the interpreter around it. `bench.py
--crossover` measures the line on the machine you are actually on; `train.py` prints
a one-line reminder at startup. Install this, then check which side of it you are on.
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
VENV = HERE / ".venv-pypy"

# Pinned. An unpinned "latest" would silently change the interpreter under a running
# experiment, and `test_equivalence.py`'s CPython-equals-PyPy guarantee is a claim
# about a specific build until it is re-checked.
PYPY_VERSION = "7.3.23"
PYPY_PYTHON = "3.11"

_ARCHIVES = {
    ("Windows", "AMD64"): (f"pypy{PYPY_PYTHON}-v{PYPY_VERSION}-win64.zip", "zip"),
    ("Linux", "x86_64"): (f"pypy{PYPY_PYTHON}-v{PYPY_VERSION}-linux64.tar.bz2", "tar"),
    ("Linux", "aarch64"): (f"pypy{PYPY_PYTHON}-v{PYPY_VERSION}-aarch64.tar.bz2", "tar"),
    ("Darwin", "arm64"): (f"pypy{PYPY_PYTHON}-v{PYPY_VERSION}-macos_arm64.tar.bz2", "tar"),
    ("Darwin", "x86_64"): (f"pypy{PYPY_PYTHON}-v{PYPY_VERSION}-macos_x86_64.tar.bz2", "tar"),
}
BASE_URL = "https://downloads.python.org/pypy/"


def venv_python() -> pathlib.Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _runtime_dir() -> pathlib.Path:
    """Where the unpacked interpreter lives: the user's home, not the repo.

    Deliberately outside the working tree. It is ~300 MB of regenerable binary that
    no `.gitignore` should ever have to be trusted to catch, and one install serves
    every clone of the repo on the machine.
    """
    return pathlib.Path.home() / f"pypy{PYPY_PYTHON}-{PYPY_VERSION}"


def _download(url: str, dest: pathlib.Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"  {dest.stat().st_size // (1024 * 1024)} MB, sha256 {digest[:16]}...")


def install() -> pathlib.Path:
    key = (platform.system(), platform.machine())
    if key not in _ARCHIVES:
        raise SystemExit(f"no pinned PyPy build for {key}. Known: {sorted(_ARCHIVES)}. "
                         f"Install PyPy {PYPY_PYTHON} by hand, then run "
                         f"`<pypy> -m venv {VENV}`.")
    name, kind = _ARCHIVES[key]
    runtime = _runtime_dir()
    exe = runtime / ("pypy.exe" if sys.platform == "win32" else "bin/pypy")
    if exe.exists():
        print(f"  PyPy already unpacked at {runtime}")
    else:
        archive = pathlib.Path.home() / name
        if not archive.exists():
            _download(BASE_URL + name, archive)
        print(f"  unpacking -> {runtime}")
        staging = pathlib.Path.home() / f"_pypy_staging_{PYPY_VERSION}"
        shutil.rmtree(staging, ignore_errors=True)
        if kind == "zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(staging)
        else:
            with tarfile.open(archive, "r:bz2") as t:
                t.extractall(staging)
        # every archive contains exactly one top-level directory
        (inner,) = list(staging.iterdir())
        shutil.rmtree(runtime, ignore_errors=True)
        shutil.move(str(inner), str(runtime))
        shutil.rmtree(staging, ignore_errors=True)
        if sys.platform != "win32":
            exe.chmod(0o755)
    return exe


def make_venv(exe: pathlib.Path) -> None:
    if venv_python().exists():
        print(f"  venv already present at {VENV}")
        return
    print(f"  creating venv -> {VENV}")
    subprocess.run([str(exe), "-m", "venv", str(VENV)], check=True)


def check() -> int:
    """Verify the venv is a working PyPy that can import the experiment."""
    py = venv_python()
    if not py.exists():
        print(f"  MISSING: {py}\n  run: python setup_pypy.py")
        return 1
    probe = ("import sys, cgp, ecgp, tasks, train; "
             "print('  ' + sys.version.split()[0], "
             "'PyPy' if hasattr(sys,'pypy_version_info') else 'NOT PyPy', "
             "'| bit_count', (255).bit_count(), "
             "'| retina patterns', tasks.n_patterns('retina_ka2005'))")
    r = subprocess.run([str(py), "-c", probe], cwd=str(HERE),
                       capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode != 0:
        return 1
    if "NOT PyPy" in r.stdout:
        print("  ERROR: that venv is not PyPy.")
        return 1
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print(f"setup_pypy.py | PyPy {PYPY_VERSION} (Python {PYPY_PYTHON}) | "
          f"{platform.system()} {platform.machine()}")
    if "--check" not in argv:
        make_venv(install())
    rc = check()
    if rc == 0:
        py = venv_python()
        print("\n  ready. The search runs headless under PyPy:")
        print(f"      {py} train.py --task retina_ka2005 --no-viz --save-best")
        print("  and the diagrams are drawn afterwards under CPython:")
        print("      conda run -n lndp python render.py <run-dir>")
        print("\n  Check which interpreter your task actually wants first:")
        print("      conda run -n lndp python bench.py --crossover")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
