# PPO and discrete SAC on new_env

Activate the same Conda environment used for RP and run from the repository root.
PPO here is the pure PyTorch implementation `ppo_torch_baseline`; Ray/RLlib is
not required. The separate `ppo_baseline` variant still refers to RLlib.

## Commands

PPO:

```bat
python -m scripts.run_experiment --env new_env --rp_variant ppo_torch_baseline --num-episodes 3000 --scenario-schedule-seed 42 --validation-start-episode 2000 --validation-interval 200 --max-decisions-per-ev 80 --max-simulation-steps 6000 --autosave-every 0
```

SAC-Discrete:

```bat
python -m scripts.run_experiment --env new_env --rp_variant sacd_baseline --num-episodes 3000 --epsilon-warmup-episodes 225 --epsilon-decay-end-episode 1200 --scenario-schedule-seed 42 --validation-start-episode 2000 --validation-interval 200 --max-decisions-per-ev 80 --max-simulation-steps 6000 --autosave-every 0
```

Both commands start fresh runs using the existing new_env manifest, EV IDs and
goals. They set 80 decisions per EV, 6000 steps at 0.5 seconds, and scenario
schedule seed 42. The new budget flags default to no override, preserving older
commands. These limits apply to training and its periodic validation. Standalone
validation/testing uses its own config; match those limits when evaluating.
No route, goal, reward, or default config files are changed.

Validation runs at episodes 2000, 2200, 2400, 2600, 2800, 3000 (90 cases each).
Outputs are `outputs/new_env/ppo_torch_baseline` and
`outputs/new_env/sacd_baseline`. `--autosave-every 0` disables periodic full
training-state saves; reward JSON and validation checkpoints are still saved.
Scenario seed 42 matches scenario order, not all model/rollout random seeds.

## Learning and exploration

PPO samples its categorical policy from episode 1, without a DQN replay warmup
or epsilon-greedy decay. It updates after at least 4096 complete transitions,
using minibatch size 512 and 10 epochs. An episode without an update is expected
while the rollout buffer is below that threshold. Validation uses greedy actions.

SAC keeps the existing 225-episode learning warmup in the command above. It uses
uniform replay (PER alpha 0), batch size 256, automatic entropy temperature, and
categorical policy sampling. Optional epsilon exploration and traffic-action
features remain disabled. The shared epsilon schedule does not control SAC
sampling unless `--sacd-use-epsilon-exploration` is explicitly enabled.
Validation uses greedy actions. Keep protocol changes explicit in comparisons.

## Fixes and verification

SAC replay now retains feature axes for action/reward/done across NumPy versions,
including legacy scalar-field replay buffers. This prevents gather errors and
unintended target broadcasting. Plain SAC and PPO skip unused graph queries and
replay graph copies. SAC with optional traffic-action features retains its full
graph path. Goal indices, neighbor masks and reward-related history are retained.

Pure PyTorch PPO now clips the change from the old value prediction and uses
the maximum of clipped/unclipped squared errors, rather than capping squared
error itself. The old cap caused zero critic gradients for large failure costs.
This follows the clipped-value formulation in
[OpenAI Baselines PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py).
A regression test uses reward -20 with only one legal action and verifies that
the critic parameters update independently of policy gradients.

The optional RLlib agent also skips unused graphs, but end-to-end checks in
this release use pure PyTorch PPO, not Ray/RLlib. Short SUMO checks exercise real
transitions, actual learning, checkpoint reload, restart and one validation case;
they do not establish convergence over 3000 episodes. Windows error 4551 remains
a separate application-control issue.

Verified locally on Windows, NumPy 2.0.1, PyTorch 2.3.1, SUMO 1.24.0:
132 focused regression tests passed. CPU smoke runs used the real 379-node,
4-neighbor new_env with 80 decisions / 6000 steps. PPO completed 6 episodes,
one 5511-transition rollout update and 110 optimizer steps; SAC completed one
episode and 64 updates (warmup disabled only for this smoke test). Both reloaded
checkpoints, restarted SUMO, and evaluated one validation case with 14 EVs.
These brief runs recorded 0/14 arrivals; this is a compatibility check of
essentially untrained policies, not a trained-performance evaluation.
