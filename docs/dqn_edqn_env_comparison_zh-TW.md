# DQN 新舊環境支援與 eDQN 比較（2026-11-14）

目前的 `dqn_baseline` 可用 `--env env` 執行舊 Brooklyn 資料，也可用 `--env new_env` 執行 Berlin 資料。兩者沿用各自的 manifest、70／15／15 資料切分、EV／goal 配對與 SUMO 網路。每個環境須分別從頭訓練，兩張路網的 one-hot 索引與輸入維度不同，不能直接互載模型權重。

## 本次修改

1. 在 `DQN/dqn_re.py` 宣告 `requires_graph_data = False`。DQN 仍讀取自己的局部號誌觀測，但略過它未使用的全路網 GNN 特徵、候選圖特徵與 replay 圖物件複製。數值狀態、合法動作、reward、PER 與 Q 更新公式不變。
2. 確認既有 DQN factory 會從實際路網取得節點數與最大鄰居數；共用 PER 已有 NumPy scalar／length-one replay 的相容修正，這次實際測試 DQN 使用兩種 replay 格式更新，不再額外修改共用 buffer。
3. 修正 DQN 的預設評估模型路徑：新舊環境均優先使用自己的 `dqn_best_100pct_arrival_lowest_cost.pt`，若不存在則使用 `dqn_best_arrival.pt`。舊環境不再指向 `dqngnn_final.pt`。
4. 啟動訊息的 `use_gnn` 依所選 agent 類型顯示；DQN 與 eDQN 顯示 False。原本 config 的通用 True 旗標不等於這些 agent 實際使用 GNN。

這次沒有將 PPO 新版的道路幾何、訪問歷史、死路篩選或 shared candidate scorer 加入 DQN；此版本保持原有 DQN baseline 定義。

## 目前程式的 DQN 與 eDQN 差異

以下依本 repository 的實作比較，並非泛指文獻中所有以 DQN／eDQN 命名的方法。這裡的 DQN 已包含 Double DQN、Dueling 與 PER，論文宜稱為「Dueling Double DQN + PER」，避免讓讀者誤以為是最原始的 vanilla DQN。eDQN 的名稱也不代表它在每個設計面向都比這個 DQN 更複雜。

| 項目 | dqn_baseline | edqn_baseline |
|---|---|---|
| 狀態 | 目前節點、目標節點、每個候選動作的兩個號誌時間 | 目前節點、目標節點、模擬時間 |
| 號誌內容 | 距下一次綠燈時間、剩餘綠燈時間，各除以 60 | 不直接輸入號誌時間 |
| 全域時間 | 不直接輸入 | 當前秒數除以 3600 |
| 舊環境輸入 | 2×115 + 2×4 = 238 | 2×115 + 1 = 231 |
| 新環境輸入 | 2×379 + 2×4 = 766 | 2×379 + 1 = 759 |
| 隱藏層 | 100 → 1024 → 512，ReLU | 256 → 128 → 64 → 16，ReLU |
| 輸出結構 | Dueling：V + A − mean(A) | Competitive／Dueling：V + A − mean(A) |
| 舊環境單個 Q 網路參數 | 654,689 | 101,669 |
| 新環境單個 Q 網路參數 | 707,489 | 236,837 |
| Target 計算 | Double DQN | Double DQN |
| Replay 抽樣 | PER，alpha=0.6；beta 從 0.4 隨抽樣次數增至 1 | Uniform，不放回均勻抽樣 |
| Replay 容量 | 10,000 transitions | 20,000 transitions |
| Loss | importance weight 加權的 MSE | Huber，delta=1，均勻權重 |
| Optimizer | Adam，lr=0.0003、weight decay=0.00001 | Adam，lr=0.00001、無 weight decay |
| Batch size | 64 | 128 |
| Gamma | 0.99 | 0.95 |
| Target network 更新 | 每次 learn 做 soft update，tau=0.003 | 每 200 次 learn 做 hard copy |
| Gradient clipping | norm 上限 10 | norm 上限 10 |
| GNN 前向計算 | 無 | 無 |
| 合法動作 | 對局部候選動作做 masking | 相同 |
| 選擇動作 | 訓練 epsilon-greedy、validation／testing greedy | 相同 |
| 共用訓練 runner | warmup、epsilon 排程與每回合最多 64 次更新 | 相同 |

兩者都是 online Q 網路選下一個動作、target Q 網路估計其值的 Double target，不能把「有沒有 Double DQN」當成目前兩種方法的差異。eDQN 的 target hard update 間隔是 **200 次學習更新**，與共用 runner 每 **200 episode** 重新啟動 SUMO 是兩個不同機制。

## 差異可能造成的影響

- **不是只有演算法不同，觀測也不同。** DQN 直接看到局部紅綠燈等待資訊；eDQN 只有全域時間。若 DQN 較好，不能全部歸因於 PER 或 MSE。這是兩套完整實作的比較，不是控制其他條件後單獨比較 replay 的消融實驗。
- **DQN 的網路更大、learning rate 為 30 倍、gamma 更高。** 它可能更快調整，也可能因 MSE 對大 TD error 更敏感而不穩定。PER 增加高 TD error 經驗被抽到的機會，但不是成功路徑才有高 priority。不能事先保證會優於 eDQN。
- **eDQN 的 Huber 與較低 learning rate 較保守。** 這不代表訓練沒正常執行；也不能拿兩種 loss 的絕對大小直接比較效果。
- **兩者仍有共同的狀態限制。** 沒有明確輸入進入道路、候選節點身分或完整訪問歷史；而同一 junction 從不同方向進入，候選動作可能不同。換用 PER 或較大網路本身不能保證解決繞圈。
- **正式比較要對齊 protocol。** 使用相同背景／EV／goal、scenario schedule、每輛 EV 80 次決策、每場景 6000 個 0.5 秒步長，以及同一 checkpoint 選擇規則；另外揭露上述觀測與超參數差異。若研究要隔離演算法因素，可再另做共同觀測與共同動作限制的消融版本，不能悄悄更換 baseline 的狀態。

## 這批舊環境 eDQN 的實際表現

分析來源是 `outputs/old_env_701515/edqn_baseline`。training_rewards.json 有 2210 筆完成的 episode，沒有整回合執行錯誤；三種需求各出現 748／721／741 次。scenario schedule seed 為 42。前 224 回合尚未 learning，EV 加權抵達率 29.39%；最後 200 回合為 41.19%，因此確實有學到部分行為。

| Validation checkpoint | 抵達 | 抵達率 | 決策上限失敗 | 模擬時間上限失敗 | 全部 EV 抵達的場景 | Better ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2000 | 475／1230 | 38.62% | 737 | 18 | 0／90 | 0／90 |
| 2200 | 486／1230 | 39.51% | 744 | 0 | 0／90 | 0／90 |

2200 相較 2000 增加 11 輛抵達；所有 744 輛失敗 EV 的路徑都已選到 80 段道路。整批路徑中，716 條符合「尾端連續出現三個相同區塊、每個區塊 2～20 段道路」的簡單繞圈檢查。這不是完整環路檢測，但支持目前主要問題為策略重複繞圈；單靠這些輸出仍不能完全排除個別路由問題。沒有整回合 SUMO 崩潰紀錄。

最後三個完整 200-episode 區間的訓練抵達率為 41.65%、38.69%、40.78%，已呈現一段平台。可以先停下來試 DQN；不過只有兩次 validation，尚不能嚴格判定完全收斂。若 DQN 跑到 3000 而 eDQN 只用 2200，應揭露訓練預算不同；直接公平比較時，先使用兩者共同的 2000／2200 checkpoint，或補齊 eDQN 至相同預算。

這批輸出未找到 run_configuration.json，僅能確認 scenario schedule seed；不能據此確認當初模型初始化的 training seed。若正式結果要求從頭可重現，後續各方法都應明確指定 training seed，保存輸出設定，並以多個 training seed 檢查變異。

既有 Dijkstra 結果有 EV_5 teleport，而 RL 與 Dijkstra 對 EV teleport 的失敗計分方式不同。本次沒有改變該評估定義；這不影響上述 744 個決策上限失敗的判讀，但正式旅行時間比較仍須統一或揭露此差異。

## 訓練指令

在已啟用 conda 環境的 RP_refactor／new_rp 根目錄，使用 PowerShell。這些指令從頭訓練，不使用 eDQN 或其他路網的權重。

舊環境：

```powershell
$env:PYTHONHASHSEED = "42"
python -m scripts.run_experiment --env env --rp_variant dqn_baseline --num-episodes 3000 --epsilon-warmup-episodes 225 --epsilon-decay-end-episode 1200 --scenario-schedule-seed 42 --training-seed 42 --deterministic --max-decisions-per-ev 80 --max-simulation-steps 6000 --validation-start-episode 2000 --validation-interval 200 --autosave-every 0 --output-dir outputs/old_env_701515/dqn_baseline_seed42
```

新環境：

```powershell
$env:PYTHONHASHSEED = "42"
python -m scripts.run_experiment --env new_env --rp_variant dqn_baseline --num-episodes 3000 --epsilon-warmup-episodes 225 --epsilon-decay-end-episode 1200 --scenario-schedule-seed 42 --training-seed 42 --deterministic --max-decisions-per-ev 80 --max-simulation-steps 6000 --validation-start-episode 2000 --validation-interval 200 --autosave-every 0 --output-dir outputs/new_env/dqn_baseline_seed42
```

若先對齊這次 eDQN 的最後 validation，可在命令後加 `--max-episodes 2200`。總排程仍是 3000，保留 warmup／epsilon 的邊界，但此輪只跑到 2200；autosave 關閉時，中斷後不能把一般模型 checkpoint 當成完整 RNG／replay 接續狀態。

validation 仍是 2000 起每 200 次，3000 回合共有六次。決策上限明確指定 80，避免本機與可攜版過去不同的預設值影響實驗。只改 `--env` 還要確認該設備有對應路網、背景車流、EV 與 goal 檔；GitHub 程式更新不會自動補齊未上傳的舊環境資料。

## 獨立評估

自訂 output-dir 時，要明確提供 checkpoint。訓練完成後先依 validation 選模型，再用 testing 評估。例如舊環境：

```powershell
$env:PYTHONHASHSEED = "42"
python -m scripts.validation --env env --rp_variant dqn_baseline --checkpoint outputs/old_env_701515/dqn_baseline_seed42/dqn_best_arrival.pt --training-seed 42 --deterministic --max-decisions-per-ev 80 --max-simulation-steps 6000 --output outputs/old_env_701515/dqn_baseline_seed42/validation_recheck.json
```

testing 使用 `python -m scripts.evaluate`，其餘模型與預算參數一致。若欲固定比較 2200 回合，可改讀 validation/episode_002200.pt。新環境改用 new_env 及其對應 checkpoint 路徑。

## 本次驗證

新舊環境均已實際執行：1 個訓練 episode（短測強制略過正常 warmup）、64 次學習更新、模型存讀逐張量一致，以及 1 個 validation 場景。舊路網 115 節點、輸入 238 維；新路網 379 節點、輸入 766 維。replay 均未保存無用圖物件；loss 為有限值，無整回合執行錯誤。這只是相容性檢查，不能把 1 回合模型的 validation 當作 DQN 性能結果；本次沒有啟動完整訓練，也沒有使用 testing 資料調整模型。

259 項相關回歸測試通過。排除的三個既有問題模組與先前相同：兩個 legacy alias 匯入問題及一個 Dijkstra 假路網測試；前一輪已在修改前的程式重現。新增測試涵蓋兩張路網的維度、fresh／legacy scalar replay、實際梯度更新、TLS 觀測保留、graph／non-graph payload 的數值更新與 PER priority 完全一致、模型存讀與各環境的 DQN 評估路徑。
