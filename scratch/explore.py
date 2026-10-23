import sys; sys.path.insert(0, 'scratch')
from proto import Env, greedy_rollout, np
import time
from multiprocessing import Pool
def train(p=0.0, episodes=15000, alpha=0.1, gamma=0.99, seed=0, mode='eps', q0=0.0, c=0.0, beta=0.0, eps_const=None):
    rng = np.random.default_rng(seed); env = Env(); Q = {}; N = {}; first = None; rets = []
    for ep in range(episodes):
        eps = eps_const if eps_const is not None else max(0.01, 1 - 0.99 * ep / (0.5 * episodes))
        s = env.reset(); done = False; R = 0
        while not done:
            q = Q.setdefault(s, np.full(4, q0)); n = N.setdefault(s, np.zeros(4))
            if mode == 'ucb':
                u = q + c * np.sqrt(np.log(n.sum() + 1) / np.maximum(n, 1e-9)) * (n > 0) + 1e9 * (n == 0)
                a = rng.choice(np.flatnonzero(u == u.max()))
            elif mode == 'greedy' or rng.random() >= eps:
                a = rng.choice(np.flatnonzero(q == q.max()))
            else:
                a = rng.integers(4)
            ex = a if rng.random() >= p else rng.choice([b for b in range(4) if b != a])
            s2, r, done, _ = env.step(int(ex) + 1)
            n[a] += 1
            terminal = done and env.timesteps < env.episode_length
            q2 = Q.setdefault(s2, np.full(4, q0)); N.setdefault(s2, np.zeros(4))
            rr = r + beta / np.sqrt(n[a])
            q[a] += alpha * ((rr if terminal else rr + gamma * q2.max()) - q[a])
            s = s2; R += r
        if first is None and all(env.box_pos == env.goal_pos): first = ep
        rets.append(R)
    return Q, rets, first
def job(cfg):
    name, kw = cfg; t = time.time(); Q, rets, first = train(**kw)
    out = []
    for pt in [0.0, 0.1, 0.2]:
        res = greedy_rollout(Q, p=pt, n=1000, seed=7)
        out.append(f'pt={pt}:SR={np.mean([o[1] for o in res]):.2f},R={np.mean([o[0] for o in res]):.0f}')
    return f'{name:30s} t={time.time()-t:4.0f}s first={first} ' + ' '.join(out)
if __name__ == '__main__':
    cfgs = []
    for p in [0.0, 0.1, 0.2]:
        cfgs += [(f'p={p} q0=300', dict(p=p, q0=300)),
                 (f'p={p} q0=1000', dict(p=p, q0=1000)),
                 (f'p={p} ucb c=100', dict(p=p, mode='ucb', c=100)),
                 (f'p={p} ucb c=500', dict(p=p, mode='ucb', c=500)),
                 (f'p={p} bonus b=50', dict(p=p, beta=50)),
                 (f'p={p} bonus b=200', dict(p=p, beta=200)),
                 (f'p={p} bonus b=200 greedy', dict(p=p, beta=200, mode='greedy')),
                 (f'p={p} greedy q0=0', dict(p=p, mode='greedy'))]
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
