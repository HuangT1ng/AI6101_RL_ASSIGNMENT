import sys; sys.path.insert(0, 'scratch')
from proto import Env, greedy_rollout, np
import time
from multiprocessing import Pool
env0 = Env()
CELLS = [(r, c) for r in range(6) for c in range(14) if not env0._check_off_cliff((r, c))]
BOX_CELLS = [(r, c) for (r, c) in CELLS if 1 <= r <= 4 and c >= 1 and (r, c) != tuple(env0.goal_pos)]
def train(p=0.0, episodes=15000, alpha=0.1, gamma=0.99, seed=0, rho=0.0, decay_frac=0.5):
    rng = np.random.default_rng(seed); env = Env(); Q = {}; first = None; succ = 0
    for ep in range(episodes):
        eps = max(0.01, 1 - 0.99 * ep / (decay_frac * episodes))
        s = env.reset()
        if rng.random() < rho:
            b = BOX_CELLS[rng.integers(len(BOX_CELLS))]
            while True:
                a_ = CELLS[rng.integers(len(CELLS))]
                if a_ != b: break
            env.agent_pos, env.box_pos = np.array(a_), np.array(b); s = (*a_, *b)
        done = False; fixed = s == (4, 1, 1, 1)
        while not done:
            q = Q.setdefault(s, np.zeros(4))
            a = rng.integers(4) if rng.random() < eps else rng.choice(np.flatnonzero(q == q.max()))
            ex = a if rng.random() >= p else rng.choice([b for b in range(4) if b != a])
            s2, r, done, _ = env.step(int(ex) + 1)
            terminal = done and env.timesteps < env.episode_length
            q2 = Q.setdefault(s2, np.zeros(4))
            q[a] += alpha * ((r if terminal else r + gamma * q2.max()) - q[a])
            s = s2
        if fixed and all(env.box_pos == env.goal_pos):
            succ += 1
            if first is None: first = ep
    return Q, first, succ
def job(cfg):
    name, kw = cfg; t = time.time(); Q, first, succ = train(**kw)
    out = []
    for pt in [0.0, 0.1, 0.2]:
        res = greedy_rollout(Q, p=pt, n=1000, seed=7)
        out.append(f'pt={pt}:SR={np.mean([o[1] for o in res]):.2f},R={np.mean([o[0] for o in res]):.0f}')
    return f'{name:28s} t={time.time()-t:4.0f}s first_fixedstart_succ={first} ' + ' '.join(out)
if __name__ == '__main__':
    cfgs = []
    for p in [0.0, 0.1, 0.2]:
        for rho in [0.5, 1.0]:
            for E in [15000, 30000]:
                cfgs.append((f'p={p} rho={rho} E={E}', dict(p=p, rho=rho, episodes=E)))
        cfgs.append((f'p={p} plain E=300k', dict(p=p, episodes=300000)))
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
