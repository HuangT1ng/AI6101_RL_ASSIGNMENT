import sys; sys.path.insert(0, 'scratch')
from proto import Env, greedy_rollout, np
import time, itertools
from multiprocessing import Pool

def q_learning(episodes=15000, alpha=0.1, gamma=0.99, eps0=1.0, eps_min=0.01, decay_frac=0.5, p=0.0, seed=0, replay=0):
    rng = np.random.default_rng(seed); env = Env(); Q = {}; rets = []; buf = []; succ = 0
    for ep in range(episodes):
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / (decay_frac * episodes))
        s = env.reset(); done = False; R = 0
        while not done:
            q = Q.setdefault(s, np.zeros(4))
            a = rng.integers(4) if rng.random() < eps else rng.choice(np.flatnonzero(q == q.max()))
            ex = a if rng.random() >= p else rng.choice([b for b in range(4) if b != a])
            s2, r, done, _ = env.step(int(ex) + 1)
            terminal = done and env.timesteps < env.episode_length
            q2 = Q.setdefault(s2, np.zeros(4))
            q[a] += alpha * ((r if terminal else r + gamma * q2.max()) - q[a])
            if replay:
                buf.append((s, a, r, s2, terminal))
                for i in rng.integers(len(buf), size=replay):
                    ps, pa, pr, ps2, pt = buf[i]
                    Q[ps][pa] += alpha * ((pr if pt else pr + gamma * Q[ps2].max()) - Q[ps][pa])
            s = s2; R += r
        succ += all(env.box_pos == env.goal_pos)
        rets.append(R)
    return Q, rets, succ

def job(cfg):
    name, kw = cfg; t = time.time()
    Q, rets, succ = q_learning(**kw)
    out = []
    for pt in [0.0, 0.1, 0.2]:
        res = greedy_rollout(Q, p=pt, n=1000, seed=7)
        out.append(f'pt={pt}:SR={np.mean([o[1] for o in res]):.2f},R={np.mean([o[0] for o in res]):.0f}')
    return f'{name:28s} t={time.time()-t:5.0f}s train_succ={succ} ' + ' '.join(out)

if __name__ == '__main__':
    cfgs = []
    for p in [0.1, 0.2]:
        cfgs += [(f'p={p} E=100k', dict(p=p, episodes=100000)),
                 (f'p={p} E=50k decay.9', dict(p=p, episodes=50000, decay_frac=0.9)),
                 (f'p={p} E=50k epsconst.1', dict(p=p, episodes=50000, eps0=0.1, eps_min=0.1)),
                 (f'p={p} E=15k replay5', dict(p=p, episodes=15000, replay=5)),
                 (f'p={p} E=15k replay20', dict(p=p, episodes=15000, replay=20)),
                 (f'p={p} E=15k decay.9 a.3', dict(p=p, episodes=15000, decay_frac=0.9, alpha=0.3))]
        for s in [1, 2]:
            cfgs.append((f'p={p} E=15k replay5 s{s}', dict(p=p, episodes=15000, replay=5, seed=s)))
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
