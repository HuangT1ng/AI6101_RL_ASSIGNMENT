import sys; sys.path.insert(0, 'scratch')
from proto import *
from multiprocessing import Pool
P = [0.0, 0.1, 0.2]
def job(args):
    p, seed, eps = args
    Q, rets = q_learning(p=p, seed=seed, episodes=eps)
    return [[ (np.mean([o[1] for o in res]), np.mean([o[0] for o in res])) for res in [greedy_rollout(Q, p=pt, n=500, seed=7)]][0] for pt in P], np.mean(rets[-1000:])
if __name__ == '__main__':
    eps = int(sys.argv[1])
    jobs = [(p, s, eps) for p in P for s in range(3)]
    with Pool(9) as pool: out = pool.map(job, jobs)
    for (p, s, _), (m, tr) in zip(jobs, out):
        print(f'p_train={p} seed={s} train_last1000={tr:.0f} ', ' '.join(f'pt={pt}:SR={a:.2f},R={b:.0f}' for pt, (a, b) in zip(P, m)))
