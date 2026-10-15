"""Resumeable full-data native comparison; all outputs belong outside Git."""

import argparse
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import time

DATASETS = ["hemoglobin", "perfumes", "shim", "pinene", "benzyl-anisaldehyde"]
SCALES = {
    "hemoglobin": [43939219, 460508],
    "perfumes": [768759760, 64075434],
    "shim": [2329251521, 705869301],
    "pinene": [1796660943, 209987661],
    "benzyl-anisaldehyde": [964175678, 60474720],
}
VARIANTS = [
    "baseline",
    "group",
    "linear",
    "fused",
    "moments",
    "group-linear",
    "group-fused",
    "linear-fused",
    "group-linear-fused",
    "all",
]
p = argparse.ArgumentParser()
p.add_argument("--work", type=Path, required=True)
p.add_argument("--raw-dir", type=Path, required=True)
p.add_argument("--sample-seconds", type=float, default=15)
p.add_argument("--repetitions", type=int, default=3)
p.add_argument("--memory-gib", type=int, default=4)
p.add_argument("--cpu", type=int, default=4)
p.add_argument("--phases", nargs="+", default=["solve", "dual", "gains"])
p.add_argument("--datasets", nargs="+", default=DATASETS)
p.add_argument("--variants", nargs="+", default=VARIANTS)
p.add_argument("--bench", type=Path)
p.add_argument("--results-name", default="results")
p.add_argument("--orders", nargs="+", type=int, choices=[1, 2], default=[1, 2])
a = p.parse_args()
work = a.work.resolve()
fixtures = work / "fixtures"
results = work / a.results_name
fixtures.mkdir(exist_ok=True, parents=True)
results.mkdir(exist_ok=True, parents=True)
rows = (
    json.loads((results / "results.json").read_text())
    if (results / "results.json").exists()
    else []
)
models = [
    (ds, order, "matched", SCALES[ds][order - 1])
    for ds in a.datasets
    for order in a.orders
]
if "hemoglobin" in a.datasets and 2 in a.orders:
    models += [("hemoglobin", 2, "auto", 0)]
for ds, order, model, scale in models:
    name = f"{ds}-p{order}-{model}"
    path = fixtures / (name + ".cst")
    if path.exists():
        continue
    cmd = [
        str(work / "export"),
        str(a.raw_dir / (ds + ".raw")),
        str(path),
        str(order),
        str(scale),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    (fixtures / (name + ".json")).write_text(r.stdout)
    print(
        json.dumps(dict(event="exported", fixture=name, metadata=json.loads(r.stdout))),
        flush=True,
    )


def save():
    tmp = results / "results.tmp"
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(results / "results.json")


def limit():
    resource.setrlimit(
        resource.RLIMIT_AS, (a.memory_gib * 1024**3, a.memory_gib * 1024**3)
    )


env = dict(
    os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
)
completed = {(r["fixture"], r["variant"], r["phase"]) for r in rows}
status = {
    (r["fixture"], r["variant"]): r["status"] for r in rows if r["phase"] == "solve"
}
for phase in a.phases:
    for ds, order, model, scale in models:
        fixture = f"{ds}-p{order}-{model}"
        variants = [
            v
            for v in a.variants
            if order == 2 or v not in ["moments", "all", "moment-stream", "all-stream"]
        ]
        # Baseline first, then reproducibly shuffle variants to spread order effects.
        random.Random(fixture).shuffle(variants)
        if "baseline" in variants:
            variants.remove("baseline")
            variants.insert(0, "baseline")
        for variant in variants:
            if (fixture, variant, phase) in completed:
                continue
            if phase != "solve" and status.get((fixture, variant)) != "ok":
                continue
            stem = f"{fixture}-{variant}-{phase}"
            outpath = results / (stem + ".jsonl")
            errpath = results / (stem + ".err")
            rsspath = results / (stem + ".rss")
            row = dict(
                fixture=fixture,
                dataset=ds,
                p=order,
                model=model,
                variant=variant,
                phase=phase,
                repetitions=a.repetitions,
                sample_budget_seconds=a.sample_seconds,
            )
            cmd = [
                "/usr/bin/time",
                "-f",
                "%M",
                "-o",
                str(rsspath),
                "taskset",
                "-c",
                str(a.cpu),
                str(a.bench or (work / "bench")),
                str(fixtures / (fixture + ".cst")),
                variant,
                phase,
                str(a.repetitions),
            ]
            with outpath.open("w") as out, errpath.open("w") as err:
                started = time.monotonic()
                proc = subprocess.Popen(
                    cmd,
                    stdout=out,
                    stderr=err,
                    env=env,
                    start_new_session=True,
                    preexec_fn=limit,
                )
                last_count = 0
                deadline = started + 30  # fixture/model setup has its own budget
                while True:
                    try:
                        code = proc.wait(timeout=0.15)
                        row.update(
                            status="ok" if code == 0 else "error", returncode=code
                        )
                        break
                    except subprocess.TimeoutExpired:
                        lines = outpath.read_text().splitlines()
                        valid = []
                        for line in lines:
                            try:
                                valid.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                        starts = [e for e in valid if e.get("event") == "sample_start"]
                        if len(starts) > last_count:
                            last_count = len(starts)
                            deadline = time.monotonic() + a.sample_seconds
                        if time.monotonic() > deadline:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            row.update(status="timeout", timeout_sample=last_count - 1)
                            break
                row["elapsed_seconds"] = time.monotonic() - started
            events = []
            for line in outpath.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            row["log_events"] = events
            for e in events:
                if e.get("event") == "ready":
                    row.update(
                        {
                            ("input_events" if k == "events" else k): v
                            for k, v in e.items()
                            if k != "event"
                        }
                    )
                if e.get("event") == "result":
                    row.update(
                        {
                            ("input_events" if k == "events" else k): v
                            for k, v in e.items()
                            if k != "event"
                        }
                    )
            if row["status"] == "ok" and "median_ms" not in row:
                row.update(status="error", error="missing final measurement")
            if row["status"] != "ok":
                row["stderr"] = errpath.read_text()[-2000:]
            try:
                row["peak_rss_kib"] = int(rsspath.read_text().strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass
            if phase == "solve":
                status[fixture, variant] = row["status"]
            rows.append(row)
            save()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ["log_events", "stderr"]}
                ),
                flush=True,
            )
