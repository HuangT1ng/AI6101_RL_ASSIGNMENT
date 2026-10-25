"""Build the solution notebook from the course template.

Usage: python scratch/build_nb.py OUT.ipynb [--quick]
--quick shrinks every training budget so the whole notebook can be smoke-tested in a minute.
"""
import copy
import json
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

TEMPLATE = 'RLAgent_2026Fall.ipynb'
out_path = sys.argv[1]
quick = '--quick' in sys.argv

template = nbformat.read(TEMPLATE, as_version=4)
tcells = template.cells


def src(i):
    return tcells[i].source


cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(new_code_cell(s.strip('\n')))

# ---------------------------------------------------------------- header
md(r"""
# AI6101 Reinforcement Learning Assignment — CliffBoxPushing

**Name:** HUANGTING &nbsp;&nbsp; **Matriculation number:** G2605438E

**Algorithm:** tabular **Q-learning** (off-policy TD control) with ε-greedy exploration.

Contents
1. Environment (unchanged from the template) and the random-agent baseline
2. Q-learning agent (`RLAgent`) — implementation of the TODOs
3. Training and learning progress (episode reward vs. episode)
4. Learned V-table and policy visualisation
5. Evaluation without exploration
6. Action-noise robustness study ($p_{train}, p_{test} \in \{0, 0.1, 0.2\}$)
7. Bonus A — comparison of exploration strategies (ε-greedy, UCB, no exploration)
8. Bonus B — sparse reward (only +1000 at the goal)

Nothing about the game itself (map, reward function, `step()`) is modified. All changes are in the
agent and in the training/evaluation loops, and are listed in Section 2.
""")

code(r"""
from typing import List, Tuple
import copy
import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
%matplotlib inline

SEED = 6101
np.random.seed(SEED)
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Experiment budgets
MAIN_EPISODES = 15_000          # main agent (same budget as the template)
NOISE_SHORT_EPISODES = 15_000   # robustness study, template budget
NOISE_LONG_EPISODES = 300_000   # robustness study, long budget (action noise slows learning a lot)
NOISE_SEEDS = [0, 1, 2, 3, 4]   # independent training runs per p_train
EVAL_EPISODES = 1_000           # evaluation episodes per (agent, p_test)
BONUS_EPISODES = 15_000
BONUS_SEEDS = [0, 1, 2]
ABLATION_SEEDS = [0, 1, 2]       # time-out handling ablation (Section 5)
""")

code(r"""
# Plot style: recessive axes, hairline grid, fixed categorical colour order.
SURFACE, INK, INK_2, GRID_INK = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
CLIFF_COLOR, GOAL_COLOR, BOX_COLOR = '#3b3a37', '#008300', '#eb6834'
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
BLUES.set_bad('#f1f0ec')
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID_INK, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.titlecolor': INK, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID_INK,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2, 'font.size': 10, 'axes.titlesize': 11, 'legend.frameon': False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches='tight')


def moving_average(x, window):
    # Trailing moving average (expanding mean for the first window-1 points).
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:window] / np.arange(1, window)  # expanding mean until a full window is available
    return np.concatenate([head, (c[window:] - c[:-window]) / window])
""")

md(src(1))
code(r"""
# Compatibility: NumPy >= 2.5 removed the top-level alias np.chararray used by the renderer below.
# The environment code itself is left unchanged.
import warnings
warnings.filterwarnings('ignore', message='The chararray class is deprecated')
if not hasattr(np, 'chararray'):
    np.chararray = np.char.chararray
""")
code(src(2))

md(r"""
**Notes on the dynamics that matter for learning** (read off `step()`):
* The per-step reward is $r_t = -1 - \lVert a_t - b_t\rVert_1 - \lVert b_t - g\rVert_1$ (+1000 on reaching the goal, −1000 on the cliff),
  so every non-terminal reward is negative.
* The box can never be pushed onto row 0, row 5 or column 0 (`box_hard_boundary`). If a push is blocked, the agent doesn't move either.
* The episode is cut off after 100 steps. The state has no time component, so this cut-off is a *truncation*, not a terminal state of the MDP.
""")

md(src(3))
code(src(4))

# ---------------------------------------------------------------- agent
md(r"""
## 2. Q-learning agent

Q-learning learns the optimal action-value function directly from experience:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big],$$

where the target is just $r$ when $s'$ is a true terminal state (cliff or goal). The behaviour policy is ε-greedy with respect to $Q$,
with random tie-breaking.

Implementation choices (all of them in the agent / training loop, none in the game):

| Choice | Why |
|---|---|
| ε decays linearly from 1.0 to 0.01 over the first half of training | explore broadly early, exploit late. Constant ε = 0.1 is compared in Bonus A |
| Time-outs are **not** treated as terminal: we bootstrap from $s'$ | time isn't part of the state, so treating the time limit as terminal would teach the agent that ordinary states have value ≈ 0 (*partial-episode bootstrapping*, Pardo et al. 2018). Ablation in Section 5 |
| $Q$ initialised to 0 | all non-terminal rewards are negative, so 0 is an *optimistic* initial value that already drives systematic exploration (see Bonus A) |
| Action noise is applied **after** the agent chooses $a$; the update is made for the **chosen** action $a$ | the noise is part of the environment's dynamics, so $Q(s,a)$ should learn the expected value of *choosing* $a$ under noise, and the agent can then learn noise-aware (safer) behaviour |
| `sparse_reward=True` replaces the learning signal by $1000\cdot\mathbb{1}[\text{box at goal}]$ | needed for Bonus B. The logged episode return is always the original reward, so results are comparable |

Hyper-parameters: $\alpha = 0.1$, $\gamma = 0.99$.
""")

code(r"""
def apply_action_noise(action, p):
    # With probability p, replace the selected action by one of the other three actions (uniformly).
    if p > 0 and np.random.uniform(0, 1) < p:
        others = [a for a in CliffBoxGridWorld.action_space if a != action]
        return others[np.random.randint(0, len(others))]
    return action


def is_true_terminal(env):
    # True if the episode ended at the cliff or the goal; False for a time-out (truncation).
    return bool(env._check_off_cliff(env.agent_pos) or env._check_off_cliff(env.box_pos)
                or all(env.box_pos == env.goal_pos))


class RLAgent:
    # Tabular Q-learning with epsilon-greedy exploration.
    #   epsilon decays linearly from `epsilon` to `epsilon_min` over the first `decay_ratio` of the episodes;
    #   epsilon == epsilon_min gives a constant epsilon, and epsilon = epsilon_min = 0 a purely greedy agent.
    def __init__(self, env, num_episodes, epsilon=1.0, alpha=0.1, gamma=0.99,
                 epsilon_min=0.01, decay_ratio=0.5, q_init=0.0, bootstrap_timeouts=True, snapshot_episodes=()):
        self.action_space = env.action_space
        self.q_table = dict() # Store all Q-values in a dictionary
        # Loop through all possible grid spaces, create sub-dictionary for each
        for agent_x in range(env.world_height):
            for agent_y in range(env.world_width):
                for box_x in range(env.world_height):
                    for box_y in range(env.world_width):
                        # Populate sub-dictionary with initial values for possible moves
                        self.q_table[(agent_x, agent_y, box_x, box_y)] = {k: q_init for k in self.action_space}

        self.env = env
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.decay_ratio = decay_ratio
        self.alpha = alpha
        self.gamma = gamma
        self.q_init = q_init
        self.bootstrap_timeouts = bootstrap_timeouts   # False: treat the 100-step time-out as terminal (ablation only)
        self.successes = []   # per training episode: did the box reach the goal?
        self.snapshot_episodes = set(snapshot_episodes)   # store a copy of Q before these episodes (stability study)
        self.snapshots = {}

    def epsilon_schedule(self, episode):
        decay_episodes = max(1, int(self.decay_ratio * self.num_episodes))
        frac = min(1.0, episode / decay_episodes)
        return self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    def greedy_action(self, state):
        # argmax_a Q(state, a) with random tie-breaking.
        q_values_of_state = self.q_table[state]
        max_value = max(q_values_of_state.values())
        best = [k for k, v in q_values_of_state.items() if v == max_value]
        return best[0] if len(best) == 1 else best[np.random.randint(0, len(best))]

    def act(self, state, is_training=True):
        # Returns the (epsilon-greedy) optimal action from the Q-value table.
        if is_training and np.random.uniform(0, 1) < self.epsilon:
            return self.action_space[np.random.randint(0, len(self.action_space))]
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, terminal):
        # One-step Q-learning update.
        q_sa = self.q_table[state][action]
        target = reward if terminal else reward + self.gamma * max(self.q_table[next_state].values())
        self.q_table[state][action] = q_sa + self.alpha * (target - q_sa)

    def q_array(self):
        # Compact copy of the Q-table: one row per state (in q_table order), one column per action.
        return np.array([list(q.values()) for q in self.q_table.values()])

    def learn(self, action_noise=0.0, sparse_reward=False):
        # Updates Q-values iteratively. Returns the (original) reward of every training episode.
        rewards = []
        self.successes = []

        for episode in range(self.num_episodes):
            self.epsilon = self.epsilon_schedule(episode)
            if episode in self.snapshot_episodes:
                self.snapshots[episode] = self.q_array()
            cumulative_reward = 0 # Initialise values of each game
            state = self.env.reset()
            done = False
            while not done: # Run until game terminated
                action = self.act(state)                                  # agent's choice
                executed_action = apply_action_noise(action, action_noise)   # environment-side noise
                next_state, reward, done, info = self.env.step(executed_action)
                reached_goal = all(self.env.box_pos == self.env.goal_pos)
                learning_reward = (1000 if reached_goal else 0) if sparse_reward else reward
                terminal = done and (is_true_terminal(self.env) or not self.bootstrap_timeouts)  # time-outs are bootstrapped
                self.update(state, action, learning_reward, next_state, terminal)
                cumulative_reward += reward
                state = next_state

            rewards.append(cumulative_reward)
            self.successes.append(reached_goal)

        if self.snapshot_episodes:
            self.snapshots[self.num_episodes] = self.q_array()
        return rewards
""")

code(r"""
def train_agent(config):
    # Train one agent from a config dict. Kept at module level so it can run in worker processes.
    config = dict(config)
    seed = config.pop('seed', SEED)
    agent_cls = config.pop('agent_cls', RLAgent)
    action_noise = config.pop('action_noise', 0.0)
    sparse_reward = config.pop('sparse_reward', False)
    num_episodes = config.pop('num_episodes')
    np.random.seed(seed)
    agent = agent_cls(CliffBoxGridWorld(), num_episodes, **config)
    agent.training_rewards = agent.learn(action_noise=action_noise, sparse_reward=sparse_reward)
    return agent


def parallel_map(fn, items):
    # Apply fn to independent items in parallel (fork-based worker pool), falling back to sequential execution.
    try:
        ctx = mp.get_context('fork')
        with ctx.Pool(min(len(items), os.cpu_count() or 1)) as pool:
            return pool.map(fn, items)
    except (ValueError, OSError):
        return [fn(x) for x in items]


def train_many(configs):
    # Train independent agents in parallel.
    return parallel_map(train_agent, configs)
""")

md(src(7) + "\n\n(Set `MANUAL_PLAY = True` to play; it is disabled so the notebook can run top-to-bottom.)")
manual = src(8).replace('\n', '\n    ')
code("MANUAL_PLAY = False\n\nif MANUAL_PLAY:\n    " + manual)

md(src(9))
code(src(10).replace("plt.plot(rewards)", """fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(rewards, color=SERIES[0], linewidth=1)
ax.set_xlabel('Episode'); ax.set_ylabel('Episode reward'); ax.set_title('Random agent')
print(f'Random agent: mean reward {np.mean(rewards):.0f}, successes {sum(r > 0 for r in rewards)} / {len(rewards)}')"""))

# ---------------------------------------------------------------- training
md(r"""
## 3. Train the Q-learning agent

The success rate below is computed over the last 2000 *training* episodes, so it includes ε-exploration (ε = 0.01 by then).
""")
code(r"""
np.random.seed(SEED)
env = CliffBoxGridWorld()
agent = RLAgent(env, num_episodes=MAIN_EPISODES)
start_time = time.time()
rewards = agent.learn()
print(f"Training time: {time.time() - start_time:.1f}s")

# Calculate success rate
final_rewards = np.array(rewards[-2000:])
success_rate = np.sum(final_rewards > 0) / len(final_rewards)
print(f"Success rate: {success_rate}")
first_success = int(np.argmax(agent.successes)) if any(agent.successes) else None
print(f"First successful episode: {first_success}")

# Smooth plot
weight=0.95
last = rewards[0]
smoothed = []
for v in rewards:
    smoothed_val = last * weight + (1 - weight) * v
    smoothed.append(smoothed_val)
    last = smoothed_val

# Plot the learning curve
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(rewards, color='#cde2fb', linewidth=0.3, label='episode reward (raw)')
ax.plot(smoothed, color=SERIES[0], label='smoothed (EMA, weight 0.95)')
ax.axhline(577, color=INK_2, linewidth=1)
ax.text(len(rewards), 600, 'optimal return 577', ha='right', va='bottom', color=INK_2, fontsize=9)
ax.set_xlabel('Episode'); ax.set_ylabel('Episode reward')
ax.set_title('Q-learning: learning progress (ε decays 1.0 → 0.01 over the first 7,500 episodes)')
ax.set_ylim(-2600, 900)
ax.legend(loc='upper left')
save(fig, 'learning_curve.png')
plt.show()
""")

md(r"""
**Reading the curve.** For the first few thousand episodes, ε is large and almost every episode ends at the cliff
(≈ −1000 plus the distance penalties) or in a time-out (≈ −1500 to −2500). As ε shrinks, the optimistic zero-initialised
Q-table pulls the agent into unexplored state-actions. The box is gradually pushed further along the corridor until the goal is
found for the first time, after which the +1000 propagates back through the bootstrapped targets and the curve
climbs to the optimal return of **577** (39 steps; Section 5 verifies optimality exactly by dynamic programming). The remaining dips come from the ε = 0.01 exploration
during training.
""")

# ---------------------------------------------------------------- visualisation
md(r"""
## 4. V-table and policy

$V(s) = \max_a Q(s,a)$ and $\pi(s) = \arg\max_a Q(s,a)$. The state is (agent row, agent col, box row, box col), which is
4-dimensional, so we show it in four ways:
1. the template's text output: the 20 largest state values and their greedy actions,
2. the greedy trajectory of the agent and the box,
3. a **policy map for every box position the greedy trajectory visits**. For a fixed box position, each agent cell is coloured by
   $V$ and the arrow is the greedy action (grey cells were never visited during training),
4. an overview of $V$ over box positions: $\max_{\text{agent cell}} V(\text{agent}, \text{box})$.
""")
code(r"""
ACTION_NAMES = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}


def visualize(q_table):
    v_table = {}
    policy = {}
    for key, v in q_table.items():
        v_table[key] = max(v.values())   # V(s) = max_a Q(s, a)
        policy[key] = max(v, key=v.get)  # pi(s) = argmax_a Q(s, a)
    state_num = len(q_table.keys())
    print(f"State space: {state_num}")

    # Print the largest 20 state values in v_table and the corresponding policy
    for k, val in sorted(v_table.items(), key=lambda x: x[1], reverse=True)[:20]:
      print("v_table", k, val / state_num)
      print("policy", k, policy[k], ACTION_NAMES[policy[k]])
    return v_table, policy

v_table, policy = visualize(agent.q_table)
""")

code(r"""
H, W = CliffBoxGridWorld.world_height, CliffBoxGridWorld.world_width
_probe = CliffBoxGridWorld()
CLIFF_CELLS = {(r, c) for r in range(H) for c in range(W) if _probe._check_off_cliff((r, c))}
GOAL_CELL = tuple(CliffBoxGridWorld.goal_pos.tolist())
ARROWS = {1: '↑', 2: '↓', 3: '←', 4: '→'}


def greedy_rollout(agent, action_noise=0.0):
    # Run one episode without exploration; return the visited states, actions, total reward and success flag.
    env = CliffBoxGridWorld()
    state = env.reset()
    states, actions, total, done = [state], [], 0, False
    while not done:
        action = agent.act(state, is_training=False)
        state, reward, done, _ = env.step(apply_action_noise(action, action_noise))
        states.append(state); actions.append(action); total += reward
    return states, actions, total, bool(all(env.box_pos == env.goal_pos))


def draw_world(ax, cell_fontsize=9):
    # Cliffs, goal, grid lines and axis ticks on a (row, col) grid.
    for (r, c) in CLIFF_CELLS:
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color=CLIFF_COLOR, zorder=2))
    ax.add_patch(Rectangle((GOAL_CELL[1] - .5, GOAL_CELL[0] - .5), 1, 1, fill=False, edgecolor=GOAL_COLOR, linewidth=2.5, zorder=4))
    ax.text(GOAL_CELL[1] + .32, GOAL_CELL[0] - .28, 'G', color=GOAL_COLOR, fontsize=cell_fontsize - 2, fontweight='bold', ha='center', va='center', zorder=5)
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticks(np.arange(-.5, W), minor=True); ax.set_yticks(np.arange(-.5, H), minor=True)
    ax.grid(False); ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='both', length=0, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)


def policy_grid(agent, box):
    # V and greedy action for every agent cell, given a fixed box position (NaN / None if never visited).
    V = np.full((H, W), np.nan)
    A = np.full((H, W), None)
    for r in range(H):
        for c in range(W):
            if (r, c) in CLIFF_CELLS or (r, c) == box:
                continue
            q = agent.q_table[(r, c, *box)]
            if all(v == agent.q_init for v in q.values()):
                continue
            V[r, c] = max(q.values())
            A[r, c] = max(q, key=q.get)
    return V, A


def plot_policy_for_box(ax, agent, box, norm, agent_cell=None, title=None):
    V, A = policy_grid(agent, box)
    ax.imshow(np.ma.masked_invalid(V), cmap=BLUES, norm=norm, zorder=1)
    draw_world(ax)
    for r in range(H):
        for c in range(W):
            if A[r, c] is not None:
                light = norm(V[r, c]) > 0.55
                ax.text(c, r, ARROWS[A[r, c]], ha='center', va='center', fontsize=10,
                        color='white' if light else INK, zorder=3)
    ax.add_patch(Rectangle((box[1] - .42, box[0] - .42), .84, .84, color=BOX_COLOR, zorder=3))
    ax.text(box[1], box[0], 'B', ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=4)
    if agent_cell is not None:
        ax.add_patch(Rectangle((agent_cell[1] - .5, agent_cell[0] - .5), 1, 1, fill=False, edgecolor=SERIES[3], linewidth=2.5, zorder=5))
    if title:
        ax.set_title(title, fontsize=9)
""")

code(r"""
states, actions, total, ok = greedy_rollout(agent)
print(f"Greedy rollout: {len(actions)} steps, return {total}, success {ok}")
print("Actions:", ' '.join(ARROWS[a] for a in actions))

fig, ax = plt.subplots(figsize=(10, 4.6))
draw_world(ax, cell_fontsize=12)
agent_path = np.array([s[:2] for s in states]); box_path = np.array([s[2:] for s in states])
ax.plot(agent_path[:, 1], agent_path[:, 0], color=SERIES[0], marker='o', markersize=4, label='agent path', zorder=6)
ax.plot(box_path[:, 1], box_path[:, 0], color=BOX_COLOR, marker='s', markersize=7, linewidth=3, alpha=.85, label='box path', zorder=7)
ax.text(agent_path[0, 1], agent_path[0, 0] + .32, 'A start', ha='center', va='top', fontsize=8, color=SERIES[0], zorder=8)
ax.text(box_path[0, 1], box_path[0, 0] - .32, 'B start', ha='center', va='bottom', fontsize=8, color=BOX_COLOR, zorder=8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2)
ax.set_title(f'Greedy trajectory of the learned policy: {len(actions)} steps, return {total}')
save(fig, 'trajectory.png')
plt.show()
""")

code(r"""
# One policy map per distinct box position on the greedy trajectory.
box_positions, agent_at_box = [], []
for s in states[:-1]:
    if not box_positions or box_positions[-1] != s[2:]:
        box_positions.append(s[2:]); agent_at_box.append(s[:2])

all_V = np.concatenate([policy_grid(agent, b)[0].ravel() for b in box_positions])
norm = plt.Normalize(np.nanpercentile(all_V, 2), np.nanmax(all_V))
ncols = 4
nrows = int(np.ceil(len(box_positions) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.1 * nrows), constrained_layout=True)
for k, ax in enumerate(axes.ravel()):
    if k >= len(box_positions):
        ax.axis('off'); continue
    plot_policy_for_box(ax, agent, box_positions[k], norm, agent_cell=agent_at_box[k],
                        title=f'({k + 1}) box at {box_positions[k]}')
sm = plt.cm.ScalarMappable(norm=norm, cmap=BLUES)
fig.colorbar(sm, ax=axes, shrink=0.6, label='V(s) = max_a Q(s, a)')
fig.suptitle('Learned policy (arrows) and V-table (colour) for each box position on the greedy path; '
             'yellow frame = where the agent is when the box first reaches that cell', fontsize=10)
save(fig, 'policy_maps.png')
plt.show()
""")

code(r"""
# Value of each box position, taking the best agent cell.
V_box = np.full((H, W), np.nan)
for (r, c) in [(r, c) for r in range(H) for c in range(W)]:
    if (r, c) in CLIFF_CELLS or (r, c) == GOAL_CELL:
        continue
    V, _ = policy_grid(agent, (r, c))
    if not np.all(np.isnan(V)):
        V_box[r, c] = np.nanmax(V)
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(np.ma.masked_invalid(V_box), cmap=BLUES, zorder=1)
draw_world(ax, cell_fontsize=12)
for r in range(H):
    for c in range(W):
        if not np.isnan(V_box[r, c]):
            light = im.norm(V_box[r, c]) > 0.55
            ax.text(c, r, f'{V_box[r, c]:.0f}', ha='center', va='center', fontsize=7, color='white' if light else INK, zorder=3)
fig.colorbar(im, ax=ax, shrink=0.8, label='max over agent cells of V')
ax.set_title('Value of the box position (grey: box position never reached, or unreachable for the box)')
save(fig, 'v_box.png')
plt.show()
""")

md(r"""
**Interpretation.** The agent brings the box down through the only gap in column 3 (row 4), lifts it above the column-6
cliff, slides it along row 1 past the column-7 cliff and through the row 1–2 corridor between the column-11/12 cliffs, then
moves around to push it down column 13 into the goal. Within each panel, the arrows send the agent to the correct side of the
box to make the next push. The values increase steadily along the path towards the goal.
""")

# ---------------------------------------------------------------- evaluation
md("## 5. Evaluation\n\n" + src(15).strip() + " (ε = 0, deterministic environment).")
code(src(16))

md(r"""
**Verification against the exact optimum.** The environment is small and deterministic, so we can enumerate all reachable
states and solve the 100-step problem exactly by finite-horizon dynamic programming. This is **only used as a reference**
(never for training). With action noise $p$ the executed action is the chosen one with probability $1-p$ and each other
action with probability $p/3$, so the same DP also gives the best achievable expected return under noise (an upper bound
for the robustness study) and the success probability of that return-maximising policy (a reference value: a policy that
only maximised the success probability could do slightly better).
""")
code(r"""
def build_model():
    # Enumerate reachable non-terminal states (BFS from the start) and their deterministic transitions.
    sim = CliffBoxGridWorld()
    start = sim.reset()
    index, states_list, frontier = {start: 0}, [start], [start]
    nxt, rew, term = [], [], []
    while frontier:
        new_frontier = []
        for s in frontier:
            row_n, row_r, row_t = [], [], []
            for a in sim.action_space:
                sim.agent_pos, sim.box_pos, sim.timesteps, sim.action_history = np.array(s[:2]), np.array(s[2:]), 0, []
                s2, r, done, _ = sim.step(a)
                if not done and s2 not in index:
                    index[s2] = len(states_list); states_list.append(s2); new_frontier.append(s2)
                row_n.append(index.get(s2, 0) if not done else 0); row_r.append(r); row_t.append(done)
            nxt.append(row_n); rew.append(row_r); term.append(row_t)
        frontier = new_frontier
    rew, term = np.array(rew, float), np.array(term)
    return np.array(nxt), rew, term, term & (rew > 0)


def optimal_reference(p, horizon=100):
    # Finite-horizon DP: maximal expected return and the success probability of that optimal policy.
    nxt, rew, term, goal = MODEL
    M = np.full((4, 4), p / 3); np.fill_diagonal(M, 1 - p)  # M[chosen, executed]
    V = np.zeros(len(nxt)); P = np.zeros(len(nxt))
    for _ in range(horizon):
        Qe = rew + np.where(term, 0, V[nxt])
        Pe = np.where(term, goal, P[nxt])
        Q = Qe @ M.T
        a = Q.argmax(1)
        V, P = Q[np.arange(len(nxt)), a], (Pe @ M.T)[np.arange(len(nxt)), a]
    return V[0], P[0]


MODEL = build_model()
print(f"Reachable non-terminal states: {len(MODEL[0])}")
OPTIMAL = {p: optimal_reference(p) for p in [0.0, 0.1, 0.2]}
for p, (v, s) in OPTIMAL.items():
    print(f"p = {p}: optimal expected return {v:8.1f}, success probability {s:.3f}")
print(f"Learned greedy policy (p = 0): return {total}")
""")

md(src(17))
code(src(18))

md(r"""
**Ablation: time-outs as truncation vs. terminal.** The main agent bootstraps from $s'$ when an episode is cut off after
100 steps (`bootstrap_timeouts=True`). Here the same agent is trained with the time-out treated as a terminal state instead
(target $= r$), 3 seeds each, 15,000 episodes, no action noise.
""")
code(r"""
configs = [dict(num_episodes=MAIN_EPISODES, seed=s, bootstrap_timeouts=b) for b in [True, False] for s in ABLATION_SEEDS]
agents = train_many(configs)
print(f"{'time-out handling':22s} {'first success per seed':>24s} {'last-1k train SR':>16s} {'greedy return':>14s}")
for k, name in enumerate(['bootstrap (main agent)', 'terminal']):
    runs = agents[k * len(ABLATION_SEEDS):(k + 1) * len(ABLATION_SEEDS)]
    firsts = [int(np.argmax(a.successes)) if any(a.successes) else None for a in runs]
    last_sr = np.mean([np.mean(a.successes[-1000:]) for a in runs])
    greedy = [int(greedy_rollout(a)[2]) for a in runs]
    print(f"{name:22s} {str(firsts):>24s} {last_sr:>16.3f} {str(greedy):>14s}")
""")
md(r"""
Treating the time-out as terminal makes **no measurable difference** here: both variants first reach the goal after ≈9k–11k
episodes, succeed in ≈95% of the last 1,000 training episodes and end with the optimal greedy policy (return 577). Once the
goal has been found, the greedy route takes 39 steps and never reaches the time limit, so the biased targets only affect
states visited at step 100 of failed episodes. We keep the bootstrapped target because it is the correct one for a
time-limited task, not because it improves the results.
""")

# ---------------------------------------------------------------- robustness
md(r"""
## 6. Action-noise robustness study

**Protocol.** In the training/evaluation loop (not in `step()`), with probability $p$ the selected action is replaced by
one of the other three actions uniformly at random. For each $p_{train} \in \{0, 0.1, 0.2\}$ we train **5 independent agents**
(seeds 0–4) with exactly the same agent and hyper-parameters as the main agent, under two budgets:
* **15,000 episodes**, the template's budget, and
* **300,000 episodes**, because the short budget turns out to be far too small once noise is present.

Every trained policy is then evaluated greedily (ε = 0, only the specified action noise) for 1,000 episodes under each
$p_{test} \in \{0, 0.1, 0.2\}$. Tables report the mean over the 5 seeds (± std), and the per-seed results are printed as well.
The training runs are independent, so they are executed in parallel worker processes.
""")
md(src(19))
code(r"""
noise_levels = [0.0, 0.1, 0.2]
noise_runs = {}  # (budget, p_train) -> list of agents (one per seed)
# Long runs also store a Q-table snapshot every 1/30 of the budget during the last third (used for the stability study).
SNAPSHOT_EPISODES = tuple(range(2 * NOISE_LONG_EPISODES // 3, NOISE_LONG_EPISODES, NOISE_LONG_EPISODES // 30))

for budget in [NOISE_SHORT_EPISODES, NOISE_LONG_EPISODES]:
    snapshots = SNAPSHOT_EPISODES if budget == NOISE_LONG_EPISODES else ()
    configs = [dict(num_episodes=budget, action_noise=p, seed=s, snapshot_episodes=snapshots)
               for p in noise_levels for s in NOISE_SEEDS]
    start_time = time.time()
    agents = train_many(configs)
    print(f"budget {budget}: trained {len(agents)} agents in {time.time() - start_time:.0f}s")
    for cfg, a in zip(configs, agents):
        noise_runs.setdefault((budget, cfg['action_noise']), []).append(a)

# Template variables: one agent per p_train (seed 0, long budget) and its training rewards.
trained_agents = {p: noise_runs[(NOISE_LONG_EPISODES, p)][0] for p in noise_levels}
training_rewards = {p: trained_agents[p].training_rewards for p in noise_levels}

for budget in [NOISE_SHORT_EPISODES, NOISE_LONG_EPISODES]:
    for p in noise_levels:
        firsts = [int(np.argmax(a.successes)) if any(a.successes) else None for a in noise_runs[(budget, p)]]
        print(f"budget {budget:>7}, p_train={p}: first successful training episode per seed = {firsts}")
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharey='row', constrained_layout=True)
for col, budget in enumerate([NOISE_SHORT_EPISODES, NOISE_LONG_EPISODES]):
    window = budget // 50
    for k, p in enumerate(noise_levels):
        runs = noise_runs[(budget, p)]
        x = np.arange(budget)
        R = np.array([moving_average(a.training_rewards, window) for a in runs])
        S = np.array([moving_average(a.successes, window) for a in runs])
        for row, Y in enumerate([R, S]):
            ax = axes[row, col]
            ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=SERIES[k], alpha=0.15, linewidth=0)
            ax.plot(x, np.nanmean(Y, 0), color=SERIES[k], label=f'p_train = {p}')
    axes[0, col].set_title(f'{budget:,} training episodes')
    axes[1, col].set_xlabel('Episode')
axes[0, 0].set_ylabel(f'Episode reward\n(moving average)')
axes[1, 0].set_ylabel('Training success rate\n(moving average)')
axes[1, 0].set_ylim(-0.02, 1.02)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3)
fig.suptitle(f'Training under action noise: mean over {len(NOISE_SEEDS)} seeds, band = min/max across seeds; '
             'moving-average window = 2% of the budget', fontsize=10)
save(fig, 'noise_training_curves.png')
plt.show()
""")

md(src(21))
code(r"""
def evaluate(agent, p_test, total_run=100):
    env = CliffBoxGridWorld()
    success = 0
    episode_returns = []

    for _ in range(total_run):
        state = env.reset()
        done = False
        rewards = []

        while not done:
            action = agent.act(state, is_training=False)    # greedy: no exploration
            action = apply_action_noise(action, p_test)     # only the specified action noise
            state, reward, done, info = env.step(action)
            rewards.append(reward)

        if all(env.box_pos == env.goal_pos):
            success += 1
        episode_returns.append(sum(rewards))

    return success / total_run, np.mean(episode_returns)


np.random.seed(SEED)
results = {}  # budget -> (success array [seed, p_train, p_test], return array)
for budget in [NOISE_SHORT_EPISODES, NOISE_LONG_EPISODES]:
    sr = np.zeros((len(NOISE_SEEDS), 3, 3)); ret = np.zeros((len(NOISE_SEEDS), 3, 3))
    for i, p_train in enumerate(noise_levels):
        for k, a in enumerate(noise_runs[(budget, p_train)]):
            for j, p_test in enumerate(noise_levels):
                sr[k, i, j], ret[k, i, j] = evaluate(a, p_test, total_run=EVAL_EPISODES)
    results[budget] = (sr, ret)

success_rate = results[NOISE_LONG_EPISODES][0].mean(0)
average_return = results[NOISE_LONG_EPISODES][1].mean(0)

for budget, (sr, ret) in results.items():
    print(f"\n=== Budget: {budget:,} episodes (mean over {len(NOISE_SEEDS)} seeds) ===")
    print("Rows: p_train =", noise_levels)
    print("Columns: p_test =", noise_levels)
    print("Success rate:")
    print(sr.mean(0).round(3))
    print("Average episode return:")
    print(ret.mean(0).round(1))
    print("Per-seed success rate at p_test = p_train:",
          {p: sr[:, i, i].round(2).tolist() for i, p in enumerate(noise_levels)})
print("\nDP return-maximising policy for p_test = 0 / 0.1 / 0.2: success",
      ' / '.join(f'{OPTIMAL[p][1]:.3f}' for p in noise_levels), "; return", ' / '.join(f'{OPTIMAL[p][0]:.1f}' for p in noise_levels))

print("\nPer-seed results, long budget (success rate at p_test = 0 / 0.1 / 0.2 | return at p_test = 0.1):")
sr, ret = results[NOISE_LONG_EPISODES]
for i, p in enumerate(noise_levels):
    for k, s in enumerate(NOISE_SEEDS):
        print(f"  p_train={p} seed={s}: SR " + ' / '.join(f'{v:.3f}' for v in sr[k, i]) + f" | R {ret[k, i, 1]:7.1f}")
""")

code(r"""
def heatmap(ax, mean, std, title, fmt, vmin, vmax):
    im = ax.imshow(mean, cmap=BLUES, vmin=vmin, vmax=vmax)
    for i in range(3):
        for j in range(3):
            light = im.norm(mean[i, j]) > 0.55
            ax.text(j, i, f'{mean[i, j]:{fmt}}\n±{std[i, j]:{fmt}}', ha='center', va='center',
                    fontsize=9, color='white' if light else INK)
    ax.set_xticks(range(3), [f'{p}' for p in noise_levels]); ax.set_yticks(range(3), [f'{p}' for p in noise_levels])
    ax.set_xlabel('p_test'); ax.set_ylabel('p_train'); ax.set_title(title, fontsize=10); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.4), constrained_layout=True)
for row, budget in enumerate([NOISE_SHORT_EPISODES, NOISE_LONG_EPISODES]):
    sr, ret = results[budget]
    im0 = heatmap(axes[row, 0], sr.mean(0), sr.std(0), f'Success rate — {budget:,} episodes', '.2f', 0, 1)
    im1 = heatmap(axes[row, 1], ret.mean(0), ret.std(0), f'Average return — {budget:,} episodes', '.0f', -2000, 600)
fig.colorbar(im0, ax=axes[:, 0], shrink=0.5, location='bottom', label='success rate')
fig.colorbar(im1, ax=axes[:, 1], shrink=0.5, location='bottom', label='average episode return')
opt_sr = ', '.join(f'{OPTIMAL[p][1]:.2f}' for p in noise_levels)
opt_ret = ', '.join(f'{OPTIMAL[p][0]:.0f}' for p in noise_levels)
fig.suptitle(f'Cross-noise evaluation (greedy policy, {EVAL_EPISODES} episodes per cell, mean ± std over {len(NOISE_SEEDS)} seeds)\n'
             f'DP optimum for p_test = 0 / 0.1 / 0.2: success {opt_sr}; return {opt_ret}', fontsize=10)
save(fig, 'noise_heatmaps.png')
plt.show()
""")

code(r"""
# Greedy noiseless path of every long-budget agent: does training with noise change the route?
for p in noise_levels:
    for k, a in enumerate(noise_runs[(NOISE_LONG_EPISODES, p)]):
        st, ac, tot, ok = greedy_rollout(a)
        print(f"p_train={p} seed={k}: noiseless greedy return {tot:5d}, steps {len(ac):3d}, success {ok}")
""")

md(r"""
**Stability of the greedy policy.** Two long-budget noisy runs (seed 1 for $p_{train}=0.1$ and $0.2$) fail the noise-free task
in the final evaluation even though they learned (see the training curves). To check whether this is a persistent failure, the long
runs store a copy of the Q-table every 10,000 episodes during the last third of training, and every snapshot is evaluated greedily
(ε = 0) under each $p_{test}$.
""")
code(r"""
SNAPSHOT_EVAL_EPISODES = 200
long_agents = [(p, k, a) for p in noise_levels for k, a in enumerate(noise_runs[(NOISE_LONG_EPISODES, p)])]
snap_x = sorted(long_agents[0][2].snapshots)


def snapshot_agent(agent, q_array):
    # A copy of `agent` that acts with a stored Q-table snapshot.
    snap = copy.copy(agent)
    snap.q_table = {s: dict(zip(agent.action_space, row)) for s, row in zip(agent.q_table, q_array.tolist())}
    return snap


def evaluate_snapshots(i):
    # Greedy success rate of every snapshot of long_agents[i] under each p_test: array [snapshot, p_test].
    agent = long_agents[i][2]
    np.random.seed(SEED + i)
    return np.array([[evaluate(snapshot_agent(agent, agent.snapshots[x]), p, total_run=SNAPSHOT_EVAL_EPISODES)[0]
                      for p in noise_levels] for x in snap_x])


start_time = time.time()
snapshot_sr = parallel_map(evaluate_snapshots, list(range(len(long_agents))))
SNAP = {p: np.array([sr for (q, _, _), sr in zip(long_agents, snapshot_sr) if q == p]) for p in noise_levels}  # [seed, snapshot, p_test]
print(f"Evaluated {len(snap_x)} snapshots x {len(long_agents)} runs in {time.time() - start_time:.0f}s "
      f"(episodes {snap_x[0]:,} ... {snap_x[-1]:,}, {SNAPSHOT_EVAL_EPISODES} evaluation episodes per p_test)")
for i, p in enumerate(noise_levels):
    S = SNAP[p]
    print(f"\np_train={p}: {np.mean(S[:, :, 0] == 1):.0%} of the snapshots solve the noise-free task; "
          f"mean success over all snapshots at p_test = 0 / 0.1 / 0.2: " + ' / '.join(f'{v:.3f}' for v in S.mean((0, 1))))
    print(f"  training success over the last third, per seed: {[round(float(np.mean(a.successes[snap_x[0]:])), 3) for q, _, a in long_agents if q == p]}")
    print(f"  success at p_test = {p} per seed (rows) and snapshot (columns):")
    print(S[:, :, i].round(2))
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, constrained_layout=True)
for i, p in enumerate(noise_levels):
    ax = axes[i]
    for k in range(len(NOISE_SEEDS)):
        ax.plot(snap_x, SNAP[p][k, :, i], color=SERIES[i], alpha=0.8, linewidth=1.2, marker='o', markersize=3,
                label='one line per seed' if k == 0 else None)
    ax.axhline(OPTIMAL[p][1], color=INK_2, linewidth=1, linestyle='--', label='DP optimum')
    ax.set_title(f'p_train = p_test = {p}')
    ax.set_xlabel('Training episode of the snapshot')
    ax.legend(loc='lower left', fontsize=8)
axes[0].set_ylabel('Greedy success rate')
axes[0].set_ylim(-0.03, 1.05)
fig.suptitle(f'Greedy success rate of the Q-table snapshots taken during the last third of the {NOISE_LONG_EPISODES:,}-episode runs '
             f'({SNAPSHOT_EVAL_EPISODES} episodes per point)', fontsize=10)
save(fig, 'noise_snapshots.png')
plt.show()
""")

md(r"""
### Discussion: how action noise affects learning, and whether noisy training improves robustness

**Effect on learning.** With the template's 15,000 episodes, all five noise-free agents converge (first success after
10.7k–11.1k episodes), but **none of the ten noisy agents reaches the goal even once during training**. Their greedy policies
therefore have a success rate of 0 under every $p_{test}$. Noise breaks the optimism-driven exploration that solves the
noise-free task. The first success requires pushing the box ≈15 cells along a cliff-lined corridor. Slips near the corridor
often send the agent or the box into a cliff, and the −1000 is credited to the action that was *chosen*. Corridor states
therefore start to look worse than the safe start region before the +1000 has ever been found. With 300,000 episodes all
settings learn, but noise delays the first success from 110.8k–113.5k episodes ($p=0$) to 140.9k–161.0k ($p=0.1$) and
160.0k–219.1k ($p=0.2$). Even the optimal policy fails 41% ($p=0.1$) or 75% ($p=0.2$) of the time, because 39 consecutive
moves next to cliffs have to survive the random slips. Low success rates under noise are therefore mostly a property of the
task, not of the learner.

**Cross-noise generalisation (final policies, 1,000 episodes per cell).**
* $p_{train}=0$ is perfect at $p_{test}=0$ but reaches only 0.494 / 0.168 success at $p_{test}$ = 0.1 / 0.2. That is 84% / 67%
  of the DP policy's 0.587 / 0.252. It only knows the shortest route, which hugs the cliffs, and it rarely visited the
  off-path states that a slip leads to.
* $p_{train}=0.1$ gives the most robust policies. Four of the five seeds reach 0.575–0.586 success at $p_{test}=0.1$ and
  0.246–0.257 at $p_{test}=0.2$ (≈98–102% of the DP values, within the ±0.016 standard error), with returns of −307 to −283
  at $p_{test}=0.1$ (DP: −279). They still solve the noise-free task (return 571–573). The learning target includes the
  consequences of slips, so actions whose slips lead into a cliff are valued lower. The slips also make the agent visit, and
  learn to recover from, off-path states.
* $p_{train}=0.2$ brings no further benefit. Its median seed is worse than $p_{train}=0.1$ even when tested at $p=0.2$ (0.23 vs
  0.25), because it finds the goal later (on average at episode 184k vs 151k) and its value estimates are noisier.

**Why one seed per noisy setting fails (snapshot study above).** Seed 1 fails the noise-free task at the end of training for
both $p_{train}=0.1$ and $0.2$, but this is not a failure to learn:
1. Its training success over the last third (0.496 at $p=0.1$, 0.162 at $p=0.2$) is the same as that of the other seeds
   (0.485–0.503 and 0.091–0.156).
2. Most of its earlier snapshots are good: 8 of its 11 snapshots reach ≈0.50–0.62 success at $p_{test}=0.1$.
3. The seeds that look good at the end have bad snapshots too. With $p_{train}=0.1$, 18% of all 55 snapshots fail the
   noise-free task, and every seed has at least one snapshot with ≤ 0.40 success at $p_{test}=0.1$ (seed 0 drops to 0.00 at
   280k episodes). With $p_{train}=0.2$, only 56% of the snapshots solve the noise-free task. Without noise, 100% do.

The cause is the **constant step size** $\alpha=0.1$. Under noise, the TD target of the same $(s,a)$ is random: a slip into a
cliff gives a target near −1000, far below the target after a normal step, so a single update moves $Q(s,a)$ by
$\alpha\cdot 1000 \approx 100$ or more. This can exceed the gap between the best and the second-best action at a critical state. The greedy action at a
critical state therefore keeps flipping, and when it flips to a bad action, the deterministic noise-free rollout gets stuck
until the 100-step time-out (return −679 / −840 instead of ≈570). Noise-free targets are deterministic, so those Q-values
converge and every snapshot behaves identically. Two standard remedies, not applied here so that all $p_{train}$ settings
use the main agent's exact configuration, are a decaying step size (e.g. $\alpha \propto 1/N(s,a)$, which satisfies the
Robbins–Monro conditions) and choosing the final policy by evaluating checkpoints on separate validation episodes.

**Answer.** Action noise makes learning much harder. It needs about 20× the template's budget, the goal is found later, and
the final greedy policy is unstable from one checkpoint to the next. Training with **moderate noise ($p_{train}=0.1$) does
improve robustness**. A converged policy closes almost the whole gap to the optimal noisy policy and loses almost nothing on
the noise-free task. Averaged over *all* 55 snapshots, including the bad ones, it still beats $p_{train}=0$ at $p_{test}=0.1$
(0.512 vs 0.495) and $p_{test}=0.2$ (0.209 vs 0.173), at the price of 18% of snapshots failing the noise-free task. Heavier
noise ($p_{train}=0.2$) brings no additional benefit within the budget (0.564 / 0.322 / 0.124 averaged over its snapshots).
""")

# ---------------------------------------------------------------- bonus A
md(r"""
## 7. Bonus A — exploration strategies

All agents use the same Q-learning update ($\alpha=0.1, \gamma=0.99$), 15,000 episodes, no action noise, 3 seeds. Only the
behaviour policy (and, for the last baseline, the initial Q-values) changes:

| Strategy | Behaviour policy |
|---|---|
| ε-greedy, decaying (main agent) | ε linearly 1.0 → 0.01 over the first half |
| ε-greedy, constant ε = 0.1 | the template's default |
| UCB1 | $\arg\max_a Q(s,a) + c\sqrt{\ln N(s)/N(s,a)}$ with $c = 10$, untried actions first |
| No exploration, $Q_0 = 0$ | always greedy (random tie-breaking) |
| No exploration, $Q_0 = -3000$ | always greedy, pessimistic initial values |
""")
code(r"""
class UCBAgent(RLAgent):
    # Q-learning with UCB1 action selection: argmax_a Q(s,a) + c * sqrt(ln N(s) / N(s,a)).
    def __init__(self, env, num_episodes, c=10.0, **kwargs):
        kwargs.update(epsilon=0.0, epsilon_min=0.0)
        super().__init__(env, num_episodes, **kwargs)
        self.c = c
        self.counts = {s: {a: 0 for a in self.action_space} for s in self.q_table}

    def act(self, state, is_training=True):
        if not is_training:
            return self.greedy_action(state)
        n = self.counts[state]
        untried = [a for a in self.action_space if n[a] == 0]
        if untried:
            action = untried[np.random.randint(0, len(untried))]
        else:
            q = self.q_table[state]
            log_total = np.log(sum(n.values()))
            scores = {a: q[a] + self.c * np.sqrt(log_total / n[a]) for a in self.action_space}
            best = max(scores.values())
            candidates = [a for a, v in scores.items() if v == best]
            action = candidates[np.random.randint(0, len(candidates))]
        n[action] += 1
        return action


EXPLORATION = {
    'ε-greedy, decaying (main)': dict(),
    'ε-greedy, constant ε=0.1': dict(epsilon=0.1, epsilon_min=0.1),
    'UCB1, c=10': dict(agent_cls=UCBAgent, c=10.0),
    'No exploration, Q0=0': dict(epsilon=0.0, epsilon_min=0.0),
    'No exploration, Q0=-3000': dict(epsilon=0.0, epsilon_min=0.0, q_init=-3000.0),
}
configs = [dict(cfg, num_episodes=BONUS_EPISODES, seed=s) for cfg in EXPLORATION.values() for s in BONUS_SEEDS]
start_time = time.time()
agents = train_many(configs)
print(f"Trained {len(agents)} agents in {time.time() - start_time:.0f}s")
exploration_runs = {name: agents[i * len(BONUS_SEEDS):(i + 1) * len(BONUS_SEEDS)] for i, name in enumerate(EXPLORATION)}
""")

code(r"""
def summarize(runs, eval_runs=1):
    # Per-strategy metrics, averaged over seeds.
    rows = []
    for name, agents_ in runs.items():
        firsts = [int(np.argmax(a.successes)) if any(a.successes) else np.nan for a in agents_]
        greedy = [greedy_rollout(a) for a in agents_]
        rows.append(dict(
            name=name,
            first=np.nanmean(firsts) if not np.all(np.isnan(firsts)) else np.nan,
            found=sum(not np.isnan(f) for f in firsts),
            mean_train=np.mean([np.mean(a.training_rewards) for a in agents_]),
            train_sr=np.mean([np.mean(a.successes) for a in agents_]),
            last_sr=np.mean([np.mean(a.successes[-1000:]) for a in agents_]),
            greedy_sr=np.mean([g[3] for g in greedy]),
            greedy_ret=np.mean([g[2] for g in greedy]),
        ))
    print(f"{'strategy':28s} {'goal found':>10s} {'1st success':>11s} {'mean train R':>12s} {'train SR':>8s} "
          f"{'last-1k SR':>10s} {'greedy SR':>9s} {'greedy R':>8s}")
    for r in rows:
        print(f"{r['name']:28s} {r['found']:>6d} / {len(BONUS_SEEDS)} {r['first']:>11.0f} {r['mean_train']:>12.0f} "
              f"{r['train_sr']:>8.3f} {r['last_sr']:>10.3f} {r['greedy_sr']:>9.2f} {r['greedy_ret']:>8.0f}")
    return rows


def plot_runs(runs, key, ylabel, title, fname, window=300, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (name, agents_) in enumerate(runs.items()):
        Y = np.array([moving_average(getattr(a, key), window) for a in agents_])
        x = np.arange(Y.shape[1])
        ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=SERIES[k], alpha=0.12, linewidth=0)
        ax.plot(x, np.nanmean(Y, 0), color=SERIES[k], label=name)
    ax.set_xlabel('Episode'); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    save(fig, fname)
    plt.show()


exploration_table = summarize(exploration_runs)
plot_runs(exploration_runs, 'training_rewards', 'Episode reward (moving average, 300)',
          'Exploration strategies: training reward (mean of 3 seeds, band = min/max)', 'bonus_exploration.png',
          ylim=(-2600, 800))
""")

md(r"""
**Analysis** (numbers from the table above; 3 seeds each).
* **The two optimism-driven strategies are clearly best.** UCB1 and "no exploration" with $Q_0=0$ both find the goal after
  ≈4.5k–4.7k episodes and reach a 100% training success rate. They also have by far the highest mean training reward (≈ −300 vs. ≈ −1100 to −1200 for ε-greedy).
* **"No exploration" with $Q_0 = 0$ is not actually exploration-free.** Every real reward is negative, so 0 is an *optimistic* estimate.
  Each time a state-action is tried its value drops below the untried ones, so the greedy agent systematically sweeps through the
  state space (optimism in the face of uncertainty).
* With a **pessimistic** $Q_0 = -3000$, the same greedy agent never finds the goal (0/3). Every tried action looks *better* than the untried
  ones, so the agent keeps repeating what it already knows. This shows that the initial values are what make the greedy agent explore.
* **UCB1** turns uncertainty into an explicit bonus that shrinks with the visit count. Its exploration is *directed* (towards rarely tried actions) rather than
  random. In preliminary runs a large $c$ (100–500, comparable to the reward scale) over-explored and converged later or not at all.
* **ε-greedy explores by undirected dithering, which suits this task poorly.** A successful episode needs about 40 consecutive correct
  actions next to cliffs, and a single random action can push the box over the edge. With decaying ε the goal is first reached only
  after ε has fallen to its floor (≈10.9k episodes, against 7.5k for the decay), so the high-ε phase mostly produces cliff deaths. Constant ε = 0.1 finds the goal
  earlier (≈7.9k) but keeps paying for its random actions: only 57% of its last 1,000 training episodes succeed (95% with decaying ε).
* All strategies except the pessimistic one end with the optimal greedy policy (return 577).
""")

# ---------------------------------------------------------------- bonus B
md(r"""
## 8. Bonus B — sparse reward

The learning signal is now **only** $r = 1000$ when the box reaches the goal and $0$ otherwise (no step cost, no distance shaping,
no cliff penalty, though falling still ends the episode). Before the goal is first reached, every update is $0 \to 0$, so the Q-table
gives no guidance at all: ε-greedy or a greedy agent with $Q_0 = 0$ is a pure random walk, and a random walk practically never
pushes the box 15+ cells along a cliff-lined corridor.

**Idea: deep exploration by optimistic initial values.** Initialise $Q_0 > 0$ and act greedily. With zero rewards the update
$Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha\gamma\max_{a'}Q(s',a')$ shrinks the value of everything already tried, while untried
state-actions keep their optimistic value. Through the bootstrapped $\max$, this optimism is propagated *backwards*: the agent
plans multi-step routes towards the frontier of unexplored states, rather than dithering locally (similar in spirit to R-max).
Falling off the cliff ends the episode and forfeits all future optimism, so the agent also learns to avoid cliffs without
ever being penalised. Once the goal is found, its real +1000 takes over.
""")
code(r"""
SPARSE = {
    'ε-greedy (decaying), Q0=0': dict(),
    'Greedy, Q0=0': dict(epsilon=0.0, epsilon_min=0.0),
    'Greedy, optimistic Q0=100': dict(epsilon=0.0, epsilon_min=0.0, q_init=100.0),
    'Greedy, optimistic Q0=1000': dict(epsilon=0.0, epsilon_min=0.0, q_init=1000.0),
}
configs = [dict(cfg, num_episodes=BONUS_EPISODES, seed=s, sparse_reward=True) for cfg in SPARSE.values() for s in BONUS_SEEDS]
start_time = time.time()
agents = train_many(configs)
print(f"Trained {len(agents)} agents in {time.time() - start_time:.0f}s")
sparse_runs = {name: agents[i * len(BONUS_SEEDS):(i + 1) * len(BONUS_SEEDS)] for i, name in enumerate(SPARSE)}

for a in agents:
    a.training_successes = np.array(a.successes, dtype=float)
sparse_table = summarize(sparse_runs)
plot_runs(sparse_runs, 'training_successes', 'Training success rate (moving average, 300)',
          'Sparse reward: training success rate (mean of 3 seeds, band = min/max)', 'bonus_sparse.png', ylim=(-0.03, 1.03))
""")

code(r"""
# Learned policy of the best sparse-reward agent.
best = sparse_runs['Greedy, optimistic Q0=100'][0]
st, ac, tot, ok = greedy_rollout(best)
print(f"Sparse-reward agent (Q0=100): {len(ac)} steps, success {ok}, return under the ORIGINAL reward {tot}")
print("Actions:", ' '.join(ARROWS[a] for a in ac))
fig, ax = plt.subplots(figsize=(10, 4.6))
draw_world(ax, cell_fontsize=12)
ap = np.array([s[:2] for s in st]); bp = np.array([s[2:] for s in st])
ax.plot(ap[:, 1], ap[:, 0], color=SERIES[0], marker='o', markersize=4, label='agent path', zorder=6)
ax.plot(bp[:, 1], bp[:, 0], color=BOX_COLOR, marker='s', markersize=7, linewidth=3, alpha=.85, label='box path', zorder=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2)
ax.set_title(f'Sparse reward, optimistic Q0=100: greedy trajectory ({len(ac)} steps)')
save(fig, 'sparse_trajectory.png')
plt.show()
""")

md(r"""
**Analysis** (numbers from the table above; 3 seeds each).
* ε-greedy and greedy-with-$Q_0=0$ never see a single success in 15k episodes, so their Q-tables stay at zero (0/3 seeds).
* **Optimistic $Q_0 = 100$ with greedy action selection solves the sparse task for every seed.** The goal is first reached after ≈9.9k
  episodes, the last 1,000 training episodes all succeed, and the greedy policy reaches the goal in 39 steps (the minimum, the same as the dense-reward
  optimum), with a return of ≈572 under the original reward (optimum 577).
* **Too much optimism prevents exploitation.** $Q_0 = 1000$ also finds the goal (≈9.9k episodes) but does not converge within 15k
  episodes (greedy success 0/3). The discounted value of the real solution from the start is only about $1000\,\gamma^{38} \approx 680$, so every
  untried state-action anywhere in the state space still looks better than the known route. The agent therefore keeps exploring until all of that
  optimism has worn off. The optimistic value should be large enough to beat dead ends (worth 0) but below the value of a real solution.
""")

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = copy.deepcopy(template.metadata)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
text = nbformat.writes(nb)
if quick:
    for k, v in {'MAIN_EPISODES = 15_000': 'MAIN_EPISODES = 3_000',
                 'NOISE_SHORT_EPISODES = 15_000': 'NOISE_SHORT_EPISODES = 1_000',
                 'NOISE_LONG_EPISODES = 300_000': 'NOISE_LONG_EPISODES = 2_000',
                 'EVAL_EPISODES = 1_000': 'EVAL_EPISODES = 20',
                 'BONUS_EPISODES = 15_000': 'BONUS_EPISODES = 1_000'}.items():
        assert k in text, k
        text = text.replace(k, v)
open(out_path, 'w').write(text)
print('wrote', out_path, len(cells), 'cells')
