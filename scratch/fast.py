import sys; sys.path.insert(0, 'scratch')
from es import train as es_train, greedy_rollout, np
from explore import train as ex_train
import time
from multiprocessing import Pool
def job(cfg):
    name, fn, kw = cfg; t = time.time()
    if fn == 'es': Q, first, _ = es_train(**kw)
    else: Q, _, first = ex_train(**kw)
    out = []
    for pt in [0.0, 0.1, 0.2]:
        res = greedy_rollout(Q, p=pt, n=1000, seed=7)
        out.append(f'pt={pt}:SR={np.mean([o[1] for o in res]):.2f},R={np.mean([o[0] for o in res]):.0f}')
    return f'{name:30s} t={time.time()-t:4.0f}s first={first} ' + ' '.join(out)
if __name__ == '__main__':
    cfgs = []
    for p in [0.1, 0.2]:
        for E in [50000, 100000]:
            for s in [0, 1, 2]:
                cfgs.append((f'p={p} E={E} decay.1 s{s}', 'es', dict(p=p, episodes=E, decay_frac=0.1, seed=s)))
                cfgs.append((f'p={p} E={E} greedy s{s}', 'ex', dict(p=p, episodes=E, mode='greedy', seed=s)))
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
