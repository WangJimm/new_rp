# eDQN baseline 訓練（2026-11-12）

在新電腦已啟用 OSM_W 的終端、new_rp 根目錄執行。先用 git pull --ff-only 更新，確認 rp/experiment/config.py 的 max_decisions_per_ev=80、max_simulation_steps=6000，與 Proposed 的本輪實驗相同。本次修正不覆蓋這兩個使用者設定。

```powershell
python -m scripts.run_experiment --env new_env --rp_variant edqn_baseline --num-episodes 3000 --epsilon-warmup-episodes 225 --epsilon-decay-end-episode 1200 --scenario-schedule-seed 42 --validation-start-episode 2000 --validation-interval 200 --autosave-every 10
```

此指令從頭建立 eDQN，輸出於 outputs/new_env/edqn_baseline，不載入 Proposed 權重。使用與 Proposed 相同的場景順序、資料分割與共同排程；scenario-schedule-seed 並不控制全部模型初始化與訓練亂數。eDQN 維持非 GNN、uniform replay、Double-Q、competitive Q network 與 Huber loss 的既有設定；不開啟額外 Dijkstra-guided exploration。

啟動後 agent type 應為 EDQNAgentRefactored。第 225 個 episode 起允許 learning；validation 在第 2000、2200、2400、2600、2800、3000 個已完成 episode 進行。完整續訓狀態預設每 10 個 episode 保存一次。

中斷後要接續 eDQN，可在相同指令後加上：

```text
--resume-state outputs/new_env/edqn_baseline/training_state_latest.pt
```

完成後依 eDQN 自己的 validation 選模型（優先到達率、再比較含失敗成本），不要強制使用 Proposed 的第 2000 個 episode。選定模型後再使用 scripts.evaluate --env new_env --rp_variant edqn_baseline --checkpoint <所選模型> --output <測試結果.json> 進行保留 testing 評估。

## NumPy replay 維度相容修正

UniformReplayMemoryNumpy 的 state/action 子陣列維度改用明確 tuple，reward/done 明確為 (1,)；sample 同時將舊 pickle buffer 的 scalar 欄位恢復為 [batch, 1]。保留原始 replay、抽樣方式、reward、TD target 與演算法超參數，不需清空舊資料。

在本機 NumPy 2.0.1 / PyTorch 2.3.1，以受影響版本的 scalar dtype 重現了 gather 的 Index tensor must have the same number of dimensions as input tensor，並驗證修正後的新舊 replay 都能完成 new_env 379 節點／4 鄰居、128 筆 batch 的 forward/backward/optimizer 更新，prediction 和 TD target 均為 [128, 1]。這是本機相容測試，不代表已在另一台 Blackwell 電腦執行完整訓練。

26 項相關測試通過，涵蓋 eDQN 新／舊 replay、實際更新、checkpoint 選擇與既有 RL 相容性檢查。
