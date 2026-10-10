"""Build, validate and time the standalone direct-DP dual prototypes.

Run from the workspace with numpy/scipy/pylmcf installed:
    .venv-rounding/bin/python wnet/prototypes/run_slope_dp_dual.py \
        --output review_runs/dp_dual_prototype --data-root wnet_smietnik/deconvbench/data

Outputs are external artifacts, not repository files. This does not install or
modify the production backend. The C++ harness uses the production ConvexPL
kernel and a standalone copy of its primal-recovery/cluster-expansion wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import pylmcf


def fractional_checks(path):
    rng = np.random.default_rng(94312)
    cases = probes = cuts = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        e, t = np.asarray(row['e'], float), np.asarray(row['t'], float)
        n, tau = len(e), row['tau']
        if n == 0:
            continue
        positions = np.r_[0, np.cumsum(row['gap'])]
        objective = (np.abs(positions[:, None] - positions[None, :]) - tau).ravel()
        # Direct continuous matching LP: each m_ij uses empirical capacity i
        # and theoretical capacity j. It shares no DP/recovery code.
        constraints = sparse.vstack([
            sparse.kron(sparse.eye(n), np.ones((1, n))),
            sparse.kron(np.ones((1, n)), sparse.eye(n)),
        ], format='csr')
        E, T = e.sum(), t.sum()
        # A(E,T) is the analytical trash bracket. Each entry also provides an
        # affine supporting piece at the original supplies (valid at ties).
        models = [
            (lambda a, b: .3*tau*a + .7*tau*b, .3*tau, .7*tau),
            (lambda a, b: tau*max(a, b), tau if E >= T else 0, tau if E < T else 0),
            (lambda a, b: tau*a + 2*tau*max(b-a, 0), -tau if T >= E else tau, 2*tau if T >= E else 0),
            (lambda a, b: tau*b + 2*tau*max(a-b, 0), 2*tau if E >= T else 0, -tau if E >= T else tau),
            (lambda a, b: tau*a, tau, 0),
            (lambda a, b: tau*b, 0, tau),
        ]
        for k in range(3):
            ep = e if k == 0 else rng.uniform(0, 30, n)
            tp = t if k == 0 else rng.uniform(0, 30, n)
            result = linprog(objective, A_ub=constraints, b_ub=np.r_[ep, tp],
                             bounds=(0, None), method='highs')
            if not result.success:
                raise AssertionError(result.message)
            if k == 0:
                np.testing.assert_allclose(result.fun, row['cost'], rtol=0, atol=1e-7)
            for key in ['pi_lazy', 'pi_cluster']:
                pi = np.asarray(row[key], float)
                matching_lower = pi @ ep + (-tau-pi) @ tp
                if matching_lower > result.fun + 1e-7:
                    raise AssertionError('invalid continuous matching cut')
                for A, dE, dT in models:
                    lower = matching_lower + A(E, T) + dE*(ep.sum()-E) + dT*(tp.sum()-T)
                    optimum = result.fun + A(ep.sum(), tp.sum())
                    if lower > optimum + 1e-7:
                        raise AssertionError('invalid full-objective cut')
                    cuts += 1
            probes += 1
        cases += 1
    return dict(cases=cases, continuous_LPs=probes, full_objective_cuts=cuts)


def make_chain(mixture, components, output, tau=.47):
    intensity_scale, cost_scale = 100_000_000, 100_000_000
    events = []
    metadata = []
    for side, path in enumerate([mixture, *components]):
        arr = np.loadtxt(path, delimiter=',')
        if not np.isfinite(arr).all() or (arr[:, 1] < 0).any():
            raise ValueError(f'invalid spectrum: {path}')
        weight = 1.0 if side == 0 else 1.0 / len(components)
        intensities = (arr[:, 1] / arr[:, 1].sum() * weight * intensity_scale).astype(np.int64)
        events.extend((float(x), int(m) if side == 0 else 0,
                       int(m) if side else 0) for x, m in zip(arr[:, 0], intensities) if m > 0)
        metadata.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             weight=weight))
    events.sort(key=lambda r: r[0])
    prev = events[0][0]; xq = 0
    with output.open('w') as f:
        f.write(f'{len(events)} {round(tau*cost_scale)}\n')
        for x, e, t in events:
            xq += round((x-prev)*cost_scale); prev = x
            f.write(f'{xq} {e} {t}\n')
    return dict(file=str(output), spectra=metadata, nodes=len(events),
                intensity_scale=intensity_scale, cost_scale=cost_scale, tau=tau,
                note='one unsplit chain, positive quantized peaks; fixed uniform component weights')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    include = here.parent/'src'/'wnet'/'cpp'
    pyl_include = Path(pylmcf.__file__).resolve().parent/'cpp'
    binary = out/'test_slope_dp_dual'
    compile_cmd = [args.cxx, '-std=c++20', '-O3', '-DNDEBUG', '-Wall', '-Wextra',
                   f'-I{include}', f'-I{pyl_include}', str(here/'test_slope_dp_dual.cpp'), '-o', str(binary)]
    subprocess.run(compile_cmd, check=True)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {args.cpu})
    checks = subprocess.check_output([str(binary), '--test', str(out/'cases.jsonl')], text=True)
    (out/'integer_checks.txt').write_text(checks)
    print(checks.strip(), flush=True)
    fractional = fractional_checks(out/'cases.jsonl')
    (out/'fractional_checks.json').write_text(json.dumps(fractional, indent=2)+'\n')
    print(json.dumps(fractional), flush=True)
    data = []
    if args.data_root:
        root = args.data_root.resolve()
        e10 = root/'magnetstein_rescue'/'e10'
        if e10.is_dir():
            data.append(make_chain(e10/'preprocessed_mix_variant_4.csv',
                                   sorted(e10.glob('preprocessed_variant_4_comp*.csv')),
                                   out/'nmr_e10.chain'))
        urine = root/'urine_interlab'
        if urine.is_dir():
            data.append(make_chain(urine/'mixture.csv',
                                   [urine/f'component_{i:02d}.csv' for i in range(3)],
                                   out/'urine_three_components.chain'))
    command = [str(binary), '--bench', *[d['file'] for d in data]]
    with (out/'benchmark.jsonl').open('w') as f:
        subprocess.run(command, check=True, stdout=f)
    meta = dict(cpu=args.cpu, platform=platform.platform(), python=sys.version,
                compiler=subprocess.check_output([args.cxx, '--version'], text=True).splitlines()[0],
                compile_command=compile_cmd, data=data,
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [here/'slope_dp_dual.hpp', here/'test_slope_dp_dual.cpp',
                                         include/'wnet'/'decompositable_graph.hpp']},
                scope='Standalone DP matching kernel/recovery; not full graph build, outer optimization, or Python API')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    for line in (out/'benchmark.jsonl').read_text().splitlines():
        row = json.loads(line)
        print(f"{row['name']}: solve {row['solve_us']/1000:.3f} ms; "
              f"lazy +{row['lazy_recovery_validation_us']/1000:.3f} ms; "
              f"cluster +{row['cluster_recovery_validation_us']/1000:.3f} ms", flush=True)


if __name__ == '__main__':
    main()
