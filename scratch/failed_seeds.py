"""Diagnose the long-budget noisy runs: is a failed seed a persistent failure or an unlucky final snapshot?

Re-trains (p_train, seed) runs exactly as the notebook does (same code, same seeding), snapshots the Q-table every
10k episodes over the last 100k, and evaluates each snapshot greedily.
Usage: .venv/bin/python scratch/failed_seeds.py  (run from the repo root)
"""
import json
import multiprocessing as mp

import numpy as np

nb = json.load(open('submission/AI6101_Code_HUANGTING_G2605438E.ipynb'))
ns = {}
exec('import copy\nimport numpy as np\n', ns)
for i in (4, 5, 10):  # compat shim, environment, agent
    exec(''.join(nb['cells'][i]['source']), ns)
CliffBoxGridWorld, RLAgent, apply_action_noise = ns['CliffBoxGridWorld'], ns['RLAgent'], ns['apply_action_noise']

BUDGET = 300_000
SNAPS = list(range(200_000, BUDGET + 1, 10_000))


def greedy_q(q, state):
    best = max(q[state].values())
    cands = [a for a, v in q[state].items() if v == best]
    return cands[np.random.randint(0, len(cands))]


def run(q, p, n):
    env = CliffBoxGridWorld()
    succ, rets = 0, []
    for _ in range(n):
        s, done, tot = env.reset(), False, 0
        while not done:
            s, r, done, _ = env.step(apply_action_noise(greedy_q(q, s), p))
            tot += r
        succ += bool(all(env.box_pos == env.goal_pos)); rets.append(tot)
    return succ / n, float(np.mean(rets))


def job(cfg):
    p, seed = cfg
    np.random.seed(seed)
    agent = RLAgent(CliffBoxGridWorld(), BUDGET)
    snaps, count, reset = {}, [0], agent.env.reset

    def hooked_reset():
        if count[0] in SNAPS:
            snaps[count[0]] = {s: dict(v) for s, v in agent.q_table.items()}
        count[0] += 1
        return reset()

    agent.env.reset = hooked_reset
    agent.learn(action_noise=p)
    snaps[BUDGET] = agent.q_table
    np.random.seed(123)
    rows = []
    for k in SNAPS:
        sr0, r0 = run(snaps[k], 0.0, 1)
        srp, rp = run(snaps[k], p, 500)
        rows.append((k, sr0, r0, srp, rp))
    last = agent.successes[-10_000:]
    out = [f'p_train={p} seed={seed}: training success over last 10k episodes = {np.mean(last):.3f}']
    out += [f'  after {k:>7,} ep: noiseless greedy success {sr0:.0f} (return {r0:6.0f}) | '
            f'p_test={p}: SR {srp:.3f}, return {rp:7.1f}' for k, sr0, r0, srp, rp in rows]
    return '\n'.join(out)


if __name__ == '__main__':
    cfgs = [(0.1, 1), (0.2, 1), (0.1, 0), (0.2, 0)]
    with mp.get_context('fork').Pool(len(cfgs)) as pool:
        for text in pool.imap(job, cfgs):
            print(text, flush=True)
