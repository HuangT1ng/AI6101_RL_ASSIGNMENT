import sys; sys.path.insert(0, 'scratch')
from es import train, greedy_rollout, np
import time
from multiprocessing import Pool
def job(cfg):
    p, E, seed = cfg; t = time.time(); Q, first, succ = train(p=p, episodes=E, seed=seed)
    out = []
    for pt in [0.0, 0.1, 0.2]:
        res = greedy_rollout(Q, p=pt, n=1000, seed=7)
        out.append(f'pt={pt}:SR={np.mean([o[1] for o in res]):.2f},R={np.mean([o[0] for o in res]):.0f}')
    return f'p={p} E={E} seed={seed} t={time.time()-t:4.0f}s first={first} ' + ' '.join(out)
if __name__ == '__main__':
    cfgs = [(p, E, s) for p in [0.0, 0.1, 0.2] for E in [150000, 300000] for s in [1, 2, 3]]
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
