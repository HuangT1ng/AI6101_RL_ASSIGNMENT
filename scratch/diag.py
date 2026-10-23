import sys; sys.path.insert(0, 'scratch')
from proto import Env, np
from multiprocessing import Pool
def run(p, episodes=15000, alpha=0.1, gamma=0.99, seed=0):
    rng = np.random.default_rng(seed); env = Env(); Q = {}; first = None; boxes = set(); maxcol = []
    for ep in range(episodes):
        eps = max(0.01, 1 - 0.99 * ep / (0.5 * episodes))
        s = env.reset(); done = False; mc = 0
        while not done:
            q = Q.setdefault(s, np.zeros(4))
            a = rng.integers(4) if rng.random() < eps else rng.choice(np.flatnonzero(q == q.max()))
            ex = a if rng.random() >= p else rng.choice([b for b in range(4) if b != a])
            s2, r, done, _ = env.step(int(ex) + 1)
            terminal = done and env.timesteps < env.episode_length
            q2 = Q.setdefault(s2, np.zeros(4))
            q[a] += alpha * ((r if terminal else r + gamma * q2.max()) - q[a])
            s = s2; boxes.add(s[2:]); mc = max(mc, s[3])
        if first is None and all(env.box_pos == env.goal_pos): first = ep
        maxcol.append(mc)
    mc = np.array(maxcol)
    return p, first, len(Q), len(boxes), [int(np.percentile(mc[i:i+1500], 90)) for i in range(0, episodes, 1500)], Q[(4,1,1,1)].round(0)
with Pool(3) as pool:
    for r in pool.map(run, [0.0, 0.1, 0.2]): print('p=%s first_success_ep=%s states=%d box_positions=%d p90_max_box_col_per_1500ep=%s Q(start)=%s' % r)
