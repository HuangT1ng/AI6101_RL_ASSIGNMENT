import json
from collections import deque
import numpy as np

notebook = json.load(open('RLAgent_2026Fall.ipynb', encoding='utf-8'))
source = ''.join(notebook['cells'][2]['source'])
namespace = {'np': np, 'copy': __import__('copy')}
exec(source, namespace)
CliffBoxGridWorld = namespace['CliffBoxGridWorld']


def model():
    sim = CliffBoxGridWorld()
    start = sim.reset()
    states = [start]
    index = {start: 0}
    next_states, rewards, terminals = [], [], []
    pending = deque([start])
    while pending:
        state = pending.popleft()
        row_next, row_reward, row_terminal = [], [], []
        for action in sim.action_space:
            sim.agent_pos = np.array(state[:2])
            sim.box_pos = np.array(state[2:])
            sim.timesteps = 0
            sim.action_history = []
            nxt, reward, done, _ = sim.step(action)
            if not done and nxt not in index:
                index[nxt] = len(states)
                states.append(nxt)
                pending.append(nxt)
            row_next.append(-1 if done else index[nxt])
            row_reward.append(reward)
            row_terminal.append(done)
        next_states.append(row_next)
        rewards.append(row_reward)
        terminals.append(row_terminal)
    return states, index, np.array(next_states), np.array(rewards), np.array(terminals)


if __name__ == '__main__':
    states, index, next_states, rewards, terminals = model()
    print('states', len(states), 'transitions', next_states.size,
          'goals', np.count_nonzero((terminals) & (rewards > 0)))
    np.savez('_model.npz', states=np.array(states), next_states=next_states,
             rewards=rewards, terminals=terminals)
