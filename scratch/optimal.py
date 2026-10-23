import sys; sys.path.insert(0, '.')
import numpy as np
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('exp', '_experiment.py'); exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
states, index, nxt, rew, term = exp.model()
print('reachable non-terminal states', len(states))
# finite-horizon undiscounted DP: V_h(s) = max_a r + V_{h-1}(s')
H = 100
V = np.zeros(len(states))
for h in range(H):
    Vn = np.where(term, 0, V[np.maximum(nxt, 0)])
    Q = rew + np.where(term, 0, Vn)
    V = Q.max(1)
print('optimal 100-step undiscounted return from start:', V[0])
# discounted VI
g = 0.99; V = np.zeros(len(states))
for _ in range(3000):
    V = (rew + g * np.where(term, 0, V[np.maximum(nxt, 0)])).max(1)
print('optimal discounted V(start) gamma=.99:', V[0])

def noisy_dp(p, H=100):
    # executed action = a w.p. 1-p, else each other action w.p. p/3
    M = np.full((4, 4), p / 3); np.fill_diagonal(M, 1 - p)
    V = np.zeros(len(states)); succ = np.zeros(len(states))
    goal = term & (rew > 0)
    pol_hist = []
    for h in range(H):
        cont = np.where(term, 0, V[np.maximum(nxt, 0)])            # (S,4) per executed action
        sc = np.where(term, goal.astype(float), succ[np.maximum(nxt, 0)])
        Qe = rew + cont                                             # value of executed action
        Q = Qe @ M.T                                                # value of selected action
        a = Q.argmax(1)
        V = Q[np.arange(len(states)), a]
        succ = (sc @ M.T)[np.arange(len(states)), a]
    return V[0], succ[0]
for p in [0.0, 0.1, 0.2]:
    print('p', p, 'optimal expected return & success prob', noisy_dp(p))

def discounted_noisy(p, g, iters=4000):
    M = np.full((4, 4), p / 3); np.fill_diagonal(M, 1 - p)
    V = np.zeros(len(states))
    for _ in range(iters):
        Qe = rew + g * np.where(term, 0, V[np.maximum(nxt, 0)])
        V = (Qe @ M.T).max(1)
    pol = (Qe @ M.T).argmax(1)
    # Monte-Carlo SR of this policy for 100-step episodes
    rng = np.random.default_rng(0); succ = 0; R = 0
    for _ in range(4000):
        s = 0
        for t in range(100):
            a = pol[s]
            if rng.random() < p: a = rng.choice([b for b in range(4) if b != a])
            R += rew[s, a]
            if term[s, a]:
                succ += rew[s, a] > 0; break
            s = nxt[s, a]
    return V[0], succ / 4000, R / 4000
print('\ndiscounted noisy optimum: (V(start), SR, mean undiscounted return over 100 steps)')
for g in [0.99, 0.995, 1.0 - 1e-9]:
    for p in [0.0, 0.1, 0.2]:
        if g > 0.999 and p == 0: pass
        print(f'gamma={g:.4f} p={p}', [round(float(x), 3) for x in discounted_noisy(p, g, iters=4000 if g < 0.999 else 3000)])
