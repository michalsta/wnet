"""Sequential, pinned, resource-bounded full-data prototype benchmark matrix."""

import argparse
import json
import os
from pathlib import Path
import queue
import resource
import signal
import subprocess
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("exe", type=Path)
parser.add_argument("data", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--seconds", type=float, default=30)
parser.add_argument("--repetitions", type=int, default=3)
parser.add_argument("--models-only", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
results_path = args.output / "results.json"
rows = json.loads(results_path.read_text()) if results_path.exists() else []
completed = {(r["dataset"], r["p"], r["label"]) for r in rows}
env = dict(
    os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
)


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


for dataset in ["hemoglobin", "perfumes", "shim", "pinene", "benzyl-anisaldehyde"]:
    for p in [1, 2]:
        modes = [
            (m, m, [])
            for m in ["baseline", "cached", "cost-scaling", "active", "approx"]
        ]
        modes += [("grid-coarse", "grid", ["1e-5"]), ("grid-fine", "grid", ["1e-7"])]
        if p == 1:
            modes += [("slope", "slope", [])]
        if args.models_only:
            modes = [("grid-model-fine", "grid-model", ["1e-7"])]
            if p == 1:
                modes.append(("slope-native", "slope-native", []))
                modes.append(("grid-fast-fine", "grid-fast", ["1e-7"]))
                modes.append(("slope-fast", "slope-fast", []))
                modes.append(("grid-sweep-model-fine", "grid-sweep-model", ["1e-7"]))
        for label, mode, extra in modes:
            if (dataset, p, label) in completed:
                continue
            stem = f"{dataset}-p{p}-{label}"
            logpath = args.output / f"{stem}.jsonl"
            errpath = args.output / f"{stem}.err"
            rsspath = args.output / f"{stem}.rss"
            command = [
                "/usr/bin/time",
                "-f",
                "%M",
                "-o",
                str(rsspath),
                "taskset",
                "-c",
                "4",
                str(args.exe),
                str(args.data / f"{dataset}-p{p}.bin"),
                mode,
                str(args.repetitions),
                *extra,
            ]
            row = dict(
                dataset=dataset,
                p=p,
                label=label,
                timeout_per_sample_seconds=args.seconds,
                memory_limit_gib=4,
            )
            events = []
            started = time.monotonic()
            with errpath.open("w") as err, logpath.open("w") as out:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    text=True,
                    env=env,
                    start_new_session=True,
                    preexec_fn=limits,
                )
                updates = queue.Queue()

                def reader(stdout=proc.stdout, updates=updates):
                    for line in stdout:
                        updates.put(line)
                    updates.put(None)

                thread = threading.Thread(target=reader, daemon=True)
                thread.start()
                deadline = time.monotonic() + args.seconds
                timed_out = False
                while True:
                    try:
                        line = updates.get(
                            timeout=max(0.01, min(0.2, deadline - time.monotonic()))
                        )
                        if line is None:
                            break
                        out.write(line)
                        out.flush()
                        event = json.loads(line)
                        events.append(event)
                        if event.get("event") == "sample":
                            deadline = time.monotonic() + args.seconds
                    except queue.Empty:
                        if time.monotonic() > deadline:
                            timed_out = True
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                            break
                proc.wait()
                thread.join(timeout=2)
                row.update(
                    status=(
                        "timeout"
                        if timed_out
                        else "ok" if proc.returncode == 0 else "error"
                    ),
                    returncode=proc.returncode,
                )
            row["elapsed_seconds"] = time.monotonic() - started
            row["events"] = events
            for event in events:
                if event.get("event") == "result":
                    row.update({k: v for k, v in event.items() if k != "event"})
            if rsspath.exists():
                try:
                    row["peak_rss_kib"] = int(
                        rsspath.read_text().strip().splitlines()[-1]
                    )
                except (ValueError, IndexError):
                    pass
            if row["status"] != "ok":
                row["error"] = errpath.read_text()[-1200:]
            rows.append(row)
            results_path.write_text(json.dumps(rows, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k not in ["events"]}),
                flush=True,
            )
