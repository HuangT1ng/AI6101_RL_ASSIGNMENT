import sys; sys.path.insert(0, 'scratch')
from explore import train, greedy_rollout, np
import time
from multiprocessing import Pool
def job(cfg):
    name, kw = cfg; t = time.time(); Q, rets, first = train(**kw)
    res = greedy_rollout(Q, p=0, n=1, seed=7)[0]
    return f'{name:28s} t={time.time()-t:4.0f}s first={first} mean_train={np.mean(rets):.0f} greedy_R={res[0]} succ={res[1]}'
if __name__ == '__main__':
    cfgs = []
    for s in [1, 2]:
        for c in [10, 30, 100]:
            cfgs.append((f'ucb c={c} s{s}', dict(mode='ucb', c=c, seed=s)))
        cfgs.append((f'greedy q0=-3000 s{s}', dict(mode='greedy', q0=-3000, seed=s)))
        cfgs.append((f'eps-const .1 s{s}', dict(eps_const=0.1, seed=s)))
    with Pool(len(cfgs)) as pool:
        for line in pool.imap_unordered(job, cfgs): print(line, flush=True)
