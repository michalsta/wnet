"""Print this package's requirement on NAME (pylmcf, wnet) as _wnet_metadata.py declares it.

Between releases that may be a pin on a git revision (PEP 508 direct reference),
at a release it is a version range.  CI jobs that build a dependency from source
with their own flags (sanitizer lanes, header-only lanes) install it from this
requirement, so they build the exact code the package will run against and the
package's own install then finds it satisfied.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _wnet_metadata import BASE_DEPENDENCIES  # noqa: E402

name = sys.argv[1]
reqs = [d for d in BASE_DEPENDENCIES if re.match(r"[A-Za-z0-9_.-]+", d).group() == name]
if len(reqs) != 1:
    sys.exit(f"{name}: expected one requirement in BASE_DEPENDENCIES, found {reqs}")
if len(sys.argv) > 2 and sys.argv[2] == "--ref":
    # The git revision to check out for header-only use: the pinned one, or the
    # release tag of the lowest version a version range allows.
    m = re.search(r"@\s*git\+\S+@([0-9a-f]{7,40})\s*$", reqs[0])
    v = re.search(r">=\s*([0-9][0-9A-Za-z.]*)", reqs[0])
    if not (m or v):
        sys.exit(f"{name}: cannot derive a git revision from {reqs[0]!r}")
    print(m.group(1) if m else "v" + v.group(1))
else:
    print(reqs[0])
