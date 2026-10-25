"""Prototype: bootstrap at time-outs (main agent) vs. treating the 100-step time-out as terminal. 3 seeds x 15k episodes."""
import json
import multiprocessing as mp
import numpy as np

nb = json.load(open('submission/AI6101_Code_YourName_YourMatriculationNumber.ipynb'))
ns = {}
exec('import copy\nimport numpy as np\n', ns)
for i in (4, 5, 10):
    exec(''.join(nb['cells'][i]['source']), ns)
CliffBoxGridWorld, RLAgent = ns['CliffBoxGridWorld'], ns['RLAgent']
true_terminal = ns['is_true_terminal']


def job(cfg):
    variant, seed = cfg
    ns['is_true_terminal'] = true_terminal if variant == 'bootstrap' else (lambda env: True)
    np.random.seed(seed)
    agent = RLAgent(CliffBoxGridWorld(), 15_000)
    rewards = agent.learn()
    first = int(np.argmax(agent.successes)) if any(agent.successes) else None
    env, s, done, tot, steps = CliffBoxGridWorld(), None, False, 0, 0
    s = env.reset()
    while not done:
        s, r, done, _ = env.step(agent.act(s, is_training=False)); tot += r; steps += 1
    return (f'{variant:9s} seed={seed} first={first} meanR={np.mean(rewards):7.0f} '
            f'last1k_SR={np.mean(agent.successes[-1000:]):.3f} greedy: steps={steps} return={tot} '
            f'success={bool(all(env.box_pos == env.goal_pos))}')


if __name__ == '__main__':
    cfgs = [(v, s) for v in ['bootstrap', 'terminal'] for s in [0, 1, 2]]
    with mp.get_context('fork').Pool(4) as pool:
        for line in pool.imap(job, cfgs):
            print(line, flush=True)
