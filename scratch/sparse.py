import sys; sys.path.insert(0, 'scratch')
from proto import Env, greedy_rollout, np
import time
from multiprocessing import Pool
def train(episodes=15000, alpha=0.1, gamma=0.99, seed=0, mode='eps', q0=0.0, beta=0.0, eps_const=None):
    rng = np.random.default_rng(seed); env = Env(); Q = {}; N = {}; first = None
    for ep in range(episodes):
        eps = eps_const if eps_const is not None else max(0.01, 1 - 0.99 * ep / (0.5 * episodes))
        s = env.reset(); done = False
        while not done:
            q = Q.setdefault(s, np.full(4, q0)); n = N.setdefault(s, np.zeros(4))
            a = rng.integers(4) if (mode == 'eps' and rng.random() < eps) else rng.choice(np.flatnonzero(q == q.max()))
            s2, r, done, _ = env.step(int(a) + 1)
            success = all(env.box_pos == env.goal_pos)
            r = 1000.0 if success else 0.0                    # sparse reward
            n[a] += 1
            terminal = done and env.timesteps < env.episode_length
            q2 = Q.setdefault(s2, np.full(4, q0)); N.setdefault(s2, np.zeros(4))
            rr = r + beta / np.sqrt(n[a])
            q[a] += alpha * ((rr if terminal else rr + gamma * q2.max()) - q[a])
            s = s2
        if first is None and success: first = ep
    return Q, first
def job(cfg):
    name, kw = cfg; t = time.time(); Q, first = train(**kw)
    res = greedy_rollout(Q, p=0, n=1, seed=7)
    return f'{name:32s} t={time.time()-t:4.0f}s first={first} greedy(R_orig,succ,steps)={res[0]}'
if __name__ == '__main__':
    cfgs = []
    for s in [0, 1]:
        cfgs += [(f'eps-greedy q0=0 s{s}', dict(seed=s)),
                 (f'greedy q0=1000 s{s}', dict(mode='greedy', q0=1000, seed=s)),
                 (f'greedy q0=100 s{s}', dict(mode='greedy', q0=100, seed=s)),
                 (f'eps q0=1000 s{s}', dict(q0=1000, seed=s)),
                 (f'greedy bonus b=1 s{s}', dict(mode='greedy', beta=1, seed=s)),
                 (f'greedy bonus b=10 s{s}', dict(mode='greedy', beta=10, seed=s)),
                 (f'eps bonus b=10 s{s}', dict(beta=10, seed=s)),
                 (f'greedy q0=1000 g.95 s{s}', dict(mode='greedy', q0=1000, gamma=0.95, seed=s))]
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
