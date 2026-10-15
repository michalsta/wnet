"""Full-fixture native benchmark runner; logs belong outside the repository."""

import argparse
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--work", type=Path, required=True)
p.add_argument("--fixtures", type=Path, required=True)
p.add_argument(
    "--stage",
    choices=["small", "ordinary", "heavy", "refinement", "auto"],
    required=True,
)
p.add_argument("--binary", type=Path)
p.add_argument("--rss-gib", type=float, default=8)
p.add_argument("--pair-limit", type=int, default=8_000_000)
p.add_argument("--cpu", type=int, default=4)
a = p.parse_args()
root = a.work
logs = root / a.stage
logs.mkdir(exist_ok=True)
path = logs / "results.json"
rows = json.loads(path.read_text()) if path.exists() else []
done = {(r["fixture"], r["variant"], r["phase"]) for r in rows}
all_variants = [
    "baseline",
    "flow",
    "virtual",
    "quadratic",
    "groupdual",
    "checkpoint64",
    "checkpoint256",
    "checkpoint1024",
    "checkpoint256-groupdual",
    "virtual-checkpoint256",
    "quadratic-checkpoint256",
]
cases = []
if a.stage == "small":
    for fixture in [
        "hemoglobin-p2-matched",
        "hemoglobin-p2-auto",
        "hemoglobin-p1-matched",
    ]:
        for phase in ["solve", "dual", "gains"]:
            vs = (
                ["baseline", "flow", "virtual", "quadratic"]
                if phase == "solve"
                else (
                    all_variants
                    if phase == "dual"
                    else [
                        "baseline",
                        "flow",
                        "virtual",
                        "quadratic",
                        "checkpoint64",
                        "checkpoint256",
                        "checkpoint1024",
                        "virtual-checkpoint256",
                        "quadratic-checkpoint256",
                    ]
                )
            )
            for v in vs:
                if "-p1-" in fixture and "quadratic" in v:
                    continue
                cases.append((fixture, v, phase, 3, 1, 20))
elif a.stage == "ordinary":
    for fixture in [
        "perfumes-p2-matched",
        "shim-p2-matched",
        "pinene-p2-matched",
        "benzyl-anisaldehyde-p2-matched",
    ]:
        for v in ["baseline", "flow", "virtual", "quadratic"]:
            cases.append(
                (
                    fixture,
                    v,
                    "solve",
                    1,
                    0,
                    90 if fixture.startswith("perfumes") else 30,
                )
            )
elif a.stage == "auto":
    for v, phase, budget in [
        ("baseline", "solve", 90),
        ("flow", "solve", 90),
        ("flow", "dual", 90),
        ("flow", "both", 90),
        ("checkpoint256-groupdual", "dual-compare", 180),
        ("checkpoint256", "gains", 240),
    ]:
        cases.append(("perfumes-p2-auto", v, phase, 1, 0, budget))
    for ds in ["shim", "pinene", "benzyl-anisaldehyde"]:
        cases.append((ds + "-p2-auto", "flow", "solve", 1, 0, 30))
elif a.stage == "refinement":
    for fixture in ["hemoglobin-p2-matched", "hemoglobin-p2-auto"]:
        for phase in ["solve", "dual", "gains"]:
            for v in ["baseline", "virtual-cached", "quadratic-cached"]:
                cases.append((fixture, v, phase, 3, 1, 20))
    for v in ["baseline", "virtual-cached", "quadratic-cached"]:
        cases.append(("perfumes-p2-matched", v, "solve", 1, 0, 90))
else:
    fixture = "perfumes-p2-matched"
    for v, phase, timeout in [
        ("baseline", "dual", 120),
        ("checkpoint256-groupdual", "dual-compare", 180),
        ("flow", "dual", 90),
        ("baseline", "gains", 120),
        ("checkpoint256", "gains", 180),
        ("flow", "gains", 90),
        ("flow", "both", 90),
    ]:
        cases.append((fixture, v, phase, 1, 0, timeout))
# Reproducible ordering, avoiding a systematic optimization-last bias.
random.Random(
    1251 + ["small", "ordinary", "heavy", "refinement", "auto"].index(a.stage)
).shuffle(cases)


def child_setup():
    # No RLIMIT_AS: unused address-space reservations must not trigger failure.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.sched_setaffinity(0, {a.cpu})


# Run cached-view follow-ups before the costly full tracebacks, on the same
# CPU and sequentially. Refinement resumes completed rows on subsequent runs.
if a.stage == "heavy":
    import sys

    with (root / "progress-refinement.log").open("w") as progress:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--work",
                str(root),
                "--fixtures",
                str(a.fixtures),
                "--stage",
                "refinement",
                "--binary",
                str(root / "bench-v2"),
                "--rss-gib",
                str(a.rss_gib),
                "--pair-limit",
                str(a.pair_limit),
                "--cpu",
                str(a.cpu),
            ],
            stdout=progress,
            stderr=subprocess.STDOUT,
            check=True,
        )

for fixture, v, phase, reps, warmup, budget in cases:
    if (fixture, v, phase) in done:
        continue
    stem = f"{fixture}-{v}-{phase}"
    row = dict(
        stage=a.stage,
        fixture=fixture,
        variant=v,
        phase=phase,
        repetitions=reps,
        warmup=warmup,
        sample_budget_seconds=budget,
        rss_budget_gib=a.rss_gib,
        address_space_limit=None,
        pair_limit=a.pair_limit,
    )
    args = [
        str(a.binary or (root / "bench")),
        str(a.fixtures / (fixture + ".cst")),
        v,
        phase,
        str(reps),
        str(a.pair_limit),
        str(warmup),
    ]
    rssfile = logs / (stem + ".rss")
    args = ["/usr/bin/time", "-f", "%M", "-o", str(rssfile)] + args
    started = time.monotonic()
    with (
        (logs / (stem + ".jsonl")).open("w") as out,
        (logs / (stem + ".err")).open("w") as err,
    ):
        child = subprocess.Popen(
            args,
            stdout=out,
            stderr=err,
            preexec_fn=child_setup,
            start_new_session=True,
            env=dict(
                os.environ,
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            ),
        )
        rss = virtual = 0
        reason = None
        sample_start = started
        last_size = 0
        while child.poll() is None:
            try:
                offspring = (
                    Path(f"/proc/{child.pid}/task/{child.pid}/children")
                    .read_text()
                    .split()
                )
                monitored = int(offspring[0]) if offspring else child.pid
                status = Path(f"/proc/{monitored}/status").read_text()
                fields = {
                    s.split(":", 1)[0]: s.split(":", 1)[1].strip()
                    for s in status.splitlines()
                    if ":" in s
                }
                rss = max(
                    rss,
                    int(fields.get("VmRSS", "0 kB").split()[0]),
                    int(fields.get("VmHWM", "0 kB").split()[0]),
                )
                virtual = max(virtual, int(fields.get("VmSize", "0 kB").split()[0]))
                size = (logs / (stem + ".jsonl")).stat().st_size
                if size != last_size:
                    messages = [
                        json.loads(s)
                        for s in (logs / (stem + ".jsonl")).read_text().splitlines()
                        if s.startswith("{")
                    ]
                    if messages and messages[-1].get("event") == "sample_start":
                        sample_start = time.monotonic()
                    last_size = size
            except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError):
                pass
            if rss > a.rss_gib * 1024**2:
                reason = "rss_limit"
            elif time.monotonic() - sample_start > budget:
                reason = "timeout"
            # If startup/model construction never reaches sample_start, the
            # same budget bounds it; setup times are retained in ready output.
            if reason:
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(0.02)
        child.wait()
    row.update(
        status=reason or ("ok" if child.returncode == 0 else "error"),
        returncode=child.returncode,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_kib=rss,
        peak_virtual_kib=virtual,
    )
    row["rss_method"] = "sampled VmRSS/VmHWM"
    try:
        exact = int(rssfile.read_text().splitlines()[-1])
        row["peak_rss_kib"] = max(row["peak_rss_kib"], exact)
        row["rss_method"] = "GNU time peak RSS"
    except (FileNotFoundError, ValueError, IndexError):
        pass
    error = (logs / (stem + ".err")).read_text()
    if "profitable-pair budget exceeded" in error:
        row["status"] = "pair_budget"
    messages = []
    for line in (logs / (stem + ".jsonl")).read_text().splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    row["log_events"] = messages
    for msg in messages:
        if msg.get("event") in ["ready", "result"]:
            row.update({k: val for k, val in msg.items() if k != "event"})
        if msg.get("event") == "sample_done":
            for key in ["trace_ms", "original_dual_ms", "grouped_dual_ms", "pairs"]:
                if key in msg:
                    row[key] = msg[key]
    if error:
        row["stderr"] = error[-3000:]
    rows.append(row)
    path.write_text(json.dumps(rows, indent=2))
    print(
        json.dumps({k: val for k, val in row.items() if k != "log_events"}), flush=True
    )
