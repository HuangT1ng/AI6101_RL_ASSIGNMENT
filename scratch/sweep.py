import sys, itertools; sys.path.insert(0, 'scratch')
from proto import *
from multiprocessing import Pool
def job(cfg):
    p, E, a, g, d = cfg
    Q, rets = q_learning(p=p, episodes=E, alpha=a, gamma=g, decay_frac=d, seed=0)
    res = greedy_rollout(Q, p=p, n=1000, seed=7)
    return cfg, np.mean([o[1] for o in res]), np.mean([o[0] for o in res])
if __name__ == '__main__':
    cfgs = list(itertools.product([0.1, 0.2], [15000, 50000], [0.05, 0.1, 0.3], [0.99, 1.0], [0.5]))
    with Pool(24) as pool:
        for cfg, sr, R in pool.imap_unordered(job, cfgs):
            print(f'p={cfg[0]} E={cfg[1]} alpha={cfg[2]} gamma={cfg[3]}  SR={sr:.2f} R={R:.0f}', flush=True)
