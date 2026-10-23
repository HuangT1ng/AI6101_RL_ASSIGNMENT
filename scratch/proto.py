import json, time, copy, sys
import numpy as np
nb = json.load(open('RLAgent_2026Fall.ipynb'))
ns = {'np': np, 'copy': copy}
exec(''.join(nb['cells'][2]['source']), ns)
Env = ns['CliffBoxGridWorld']

def q_learning(episodes=15000, alpha=0.1, gamma=0.99, eps0=1.0, eps_min=0.01, decay_frac=0.5, p=0.0, seed=0):
    rng = np.random.default_rng(seed)
    env = Env()
    Q = {}
    rets = []
    for ep in range(episodes):
        eps = max(eps_min, eps0 - (eps0 - eps_min) * ep / (decay_frac * episodes))
        s = env.reset(); done = False; R = 0
        while not done:
            q = Q.setdefault(s, np.zeros(4))
            if rng.random() < eps: a = rng.integers(4)
            else: a = rng.choice(np.flatnonzero(q == q.max()))
            ex = a
            if rng.random() < p: ex = rng.choice([b for b in range(4) if b != a])
            s2, r, done, _ = env.step(int(ex) + 1)
            q2 = Q.setdefault(s2, np.zeros(4))
            terminal = done and env.timesteps < env.episode_length  # true terminal vs timeout
            target = r if terminal else r + gamma * q2.max()
            q[a] += alpha * (target - q[a])
            s = s2; R += r
        rets.append(R)
    return Q, rets

def greedy_rollout(Q, p=0.0, n=1, seed=1):
    rng = np.random.default_rng(seed); env = Env(); out = []
    for _ in range(n):
        s = env.reset(); done = False; R = 0
        while not done:
            q = Q.get(s, np.zeros(4)); a = int(np.argmax(q))
            if rng.random() < p: a = rng.choice([b for b in range(4) if b != a])
            s, r, done, _ = env.step(a + 1); R += r
        out.append((R, all(env.box_pos == env.goal_pos), env.timesteps))
    return out

if __name__ == '__main__':
    for kw in [dict(), dict(eps0=0.1, eps_min=0.1)]:
        t = time.time(); Q, rets = q_learning(**kw)
        print(kw, 'time', round(time.time()-t,1), 'last1000 mean', np.mean(rets[-1000:]), 'greedy', greedy_rollout(Q))
