# eDQN runtime optimization (2026-11-13)

`EDQNAgentRefactored` now declares `requires_graph_data = False`. The shared
runner skips graph construction, all-edge traffic feature queries, candidate
graph features, and graph copies in replay for this agent. eDQN uses its existing
current-node/goal/time vector and legal action mask; its Q network never consumes
those graph payloads.

The compact replay bag retains source/goal indices and neighbors for terminal
state construction. Recent-node history still updates once per decision, so
revisit and U-turn rewards keep their original behavior. Network dimensions,
rewards, hyperparameters, decision limits, and simulation limits are unchanged.
Old checkpoints and replay buffers remain compatible. Agents without the new
capability flag keep the existing graph path, including Proposed.

## Verification

On one local Windows / RTX 2060 warmup episode, using the same random seeds,
new_env flow 2650, scenario seed 50032, 80 decisions per EV, and 6000 simulation
steps, measured episode time fell from 139.06 s to 17.49 s (7.95x, 87.4% less).
Timing excludes startup/model creation and includes profiling instrumentation.
This is a single warmup comparison, not a promise about another PC or the later
learning stage. No gradient updates occurred during the timed episode.

Both runs made 1577 decisions and stored 1577 transitions. Route and numeric
transition hashes, all EV costs, and termination reasons matched exactly.
Graph observation hashes intentionally differ because unused graphs are absent.
The untrained random policy's arrival rate is not a trained-model evaluation.

79 focused regression tests passed, covering graph-agent behavior, recent-node
rewards, terminal goal encoding, replay/checkpoint compatibility, routing signals,
and graph/query utilities. The new tests also compare greedy choices and three
real 128-sample learning updates at 379 nodes / 4 neighbors: losses and updated
network parameters match exactly with full versus compact replay payloads.

## Apply on another PC

In the repository root, run `git pull --ff-only`, then restart training with the
same arguments. An already running Python process will keep its loaded code.
This update does not change local decision/simulation limits or autosave settings.
It does not address the separate Windows application-control error (4551).
