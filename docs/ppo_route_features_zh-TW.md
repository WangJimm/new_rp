# PPO 道路狀態改良與可重現訓練（2026-11-14）

針對舊環境 PPO 在 3000 episode validation 僅 728／1230 輛抵達（59.19%）、354 輛進入非目標死路的問題，新增獨立模式 `ppo_route_features`。原 `ppo_torch_baseline` 保留。這次從隨機初始化重新訓練，不載入舊權重，也不使用 testing 資料調整模型。

## 狀態與模型

原 PPO 的狀態只有目前節點 one-hot、目標節點 one-hot 與模擬時間，舊環境共 231 維。新模式保留這些資訊，另加入進入道路的 one-hot 與四個全域特徵：剩餘決策預算比例、目前至目標的直線距離、曾訪問節點比例、目前節點訪問次數。

每個候選動作另有 12 個特徵：道路長度、SUMO 回報的即時道路旅行時間、候選節點至目標直線距離、相對目前位置的直線距離改善、目標方向的 dx／dy、轉向 cos／sin、下一段道路可繼續行駛的出口數、是否為目標、累計訪問次數、是否出現在最近 16 個決策節點。

距離使用路網包圍盒對角線正規化；長度以公里、旅行時間以 120 秒正規化並截限。訪問次數截限在 5 次。這些是道路幾何、當下交通觀測與該 EV 的歷史，沒有加入最短路徑距離、Dijkstra 示範、未來旅行時間或測試結果。

輸入維度：舊環境 540、新環境 1595。候選道路共用一套評分網路，因此第 1 個動作並不固定代表某種方向；重新排列候選動作只會重新排列其分數。critic 使用有效候選的平均嵌入，忽略補齊用的無效動作。

## 合法動作與終止

只對新模式，排除「選入該道路後，沒有任何 passenger 可用的下游出口，而且其終點不是目標」的動作。目標本身即使位於死路也允許進入；平行道路保留有可行出口者，再依原本的局部旅行時間規則選道路。這只是下一段道路的檢查，不是完整路徑規劃，仍可能繞圈或進入更深層的困境。

另外修正新模式的上限邊界：若最後一次允許的決策已選入通往目標的道路，允許它繼續行駛至抵達，不再提前判決策上限失敗；不增加新的決策機會，模擬時間上限仍有效。

reward、成功獎勵、失敗懲罰、重訪懲罰、資料切分與交通物理設定維持既有配置。這是明確的新觀測／動作版本，論文比較時須揭露；不能把改善全部解釋為 PPO 演算法本身的差異。

## PPO 訓練設定

| 設定 | 原 PPO | 新模式 |
|---|---:|---:|
| Learning rate | 0.0001 | 0.0003 |
| 完整 EV 軌跡累積門檻 | 4096 transitions | 1024 transitions |
| Minibatch | 512 | 256 |
| 每次更新最多 epochs | 10 | 10 |
| PPO clip | 0.2 | 0.2 |
| Entropy coefficient | 0.01 | 0.01 |
| Gamma／GAE lambda | 0.99／0.95 | 0.99／0.95 |
| KL 控制 | 調整 KL penalty coefficient | KL 超過 1.5 × 0.02 後停止當次更新後續 epoch |

新模式沿用每輛 EV 獨立完整軌跡計算 GAE。PPO 不使用 DQN 的 epsilon warmup；蒐集到足夠完整軌跡即可更新。較小的 rollout 門檻讓早期資料更快進入學習。KL 停止是限制更新幅度，不能保證策略不會退步。

KL 停止的做法可參考 [Spinning Up 的 PPO 說明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

## 可重現性

`--scenario-schedule-seed 42` 固定每一個 episode 抽到的背景車流／EV 資料；新增 `--training-seed 42` 固定 Python、NumPy、PyTorch CPU／CUDA 的初始化與抽樣。SUMO 仍優先採用 manifest 的背景 seed，讓方法間使用同一場景的模擬亂數；只有沒有場景 seed 的舊格式才退回 training seed。

`--deterministic` 啟用 PyTorch 確定性演算法、停用 cuDNN benchmark 與 TF32，並配置 cuBLAS。`PYTHONHASHSEED` 必須在啟動 Python 前設定，不能在已啟動的直譯器內補設定而視為生效。

每次訓練輸出 `run_configuration.json`：完整參數、實際模型維度、超參數、軟體版本、裝置、執行緒數、資料與程式的 SHA-256。若另有啟用完整 autosave，會保存／還原 Python、NumPy、Torch 與 CUDA 的 RNG；validation 暫時使用 RNG 後會還原訓練的 RNG。舊 autosave 缺少 RNG 時無法補回過去的隨機狀態。

可重現的範圍是匹配軟體、硬體與執行設定下的重跑；不能保證不同 GPU、PyTorch 版本或 CPU／GPU 之間逐 bit 相同，見 [PyTorch 2.3 reproducibility](https://docs.pytorch.org/docs/2.3/notes/randomness.html)。正式論文建議至少以 3 個 training seed 報告平均與變異；所有方法使用相同資料切分、scenario schedule、預算及模型選擇規則。重複同一 seed 是工程驗證，不能取代多個 training seed 的結果。

## 執行命令

在 RP_refactor 根目錄、已啟用原有 conda 環境的 PowerShell 中：

```powershell
$env:PYTHONHASHSEED = "42"
python -m scripts.run_experiment --env env --rp_variant ppo_route_features --num-episodes 3000 --training-seed 42 --deterministic --scenario-schedule-seed 42 --max-decisions-per-ev 80 --max-simulation-steps 6000 --validation-start-episode 2000 --validation-interval 200 --autosave-every 0 --output-dir outputs/old_env_701515/ppo_route_features_seed42
```

這個命令從頭訓練，不加 checkpoint／resume 參數。validation 依照先前偏好，在 2000、2200、2400、2600、2800、3000，共 6 次。autosave 關閉，仍會有 validation 的模型存檔；中途關閉時不保證能接續同一訓練。

指定 `--env new_env` 即可使用 Berlin 資料；請同時改用獨立輸出目錄，例如 `outputs/new_env/ppo_route_features_seed42`。不同 training seed 使用不同輸出目錄，scenario-schedule-seed 仍可固定 42，讓方法使用相同回合資料。

訓練完成後，依 validation 選模型，再單次執行 testing；不要直接假定最後 3000 的模型最好。獨立 validation／testing 也須指定一致的決策與模擬上限，因為剩餘決策預算是新模型的輸入之一，例如：

```powershell
$env:PYTHONHASHSEED = "42"
python -m scripts.validation --env env --rp_variant ppo_route_features --checkpoint outputs/old_env_701515/ppo_route_features_seed42/ppo_route_best_arrival.pt --training-seed 42 --deterministic --max-decisions-per-ev 80 --max-simulation-steps 6000 --output outputs/old_env_701515/ppo_route_features_seed42/validation_recheck.json
```

testing 改為 `python -m scripts.evaluate`，並使用相同模型及上限。

## 驗證範圍與限制

本次僅執行短程測試，沒有啟動完整 3000 episode 訓練。固定挑選 validation 每種需求 × 每種 EV 滲透率的第一筆，共 6 場景、82 輛 EV，作為接線與初期效果檢查，不等同完整 90 場景、1230 輛 EV 的正式 validation。

相同 seed 42 的兩次獨立從頭訓練，各 20 episode、10 次 rollout 更新、580 次 minibatch 梯度更新。完整 training_rewards.json、validation JSON 與所有模型張量逐一完全相同，兩次資料雜湊與程式雜湊也相同。測試使用本機 CPU、Torch 單執行緒，沒有驗證跨 GPU 或跨軟體版本的逐 bit 一致性。

| 同一批 6 個 validation 場景 | 未訓練的新模式 | 訓練 20 episode 後（兩次結果相同） |
|---|---:|---:|
| 抵達 EV | 7／82 | 81／82 |
| 抵達率 | 8.54% | 98.78% |
| 決策上限失敗 | 38 | 1 |
| 模擬時間上限失敗 | 21 | 0 |
| EV teleport 失敗 | 16 | 0 |
| 非目標死路失敗 | 0 | 0 |

未訓練的新版也已套用死路篩選，因此其死路失敗為 0 不代表策略學會避開死路。20 episode 後的抵達改善，則是與同一套新觀測／動作設定的未訓練模型比較所得。

20 episode 後，這 6 場景的原始總 EV 旅行時間 483.5083 分鐘，Dijkstra 為 150.1 分鐘，場景 better ratio 為 0／6。表示初步主要改善抵達，尚未完成旅行時間最佳化。不能以這 82 輛 EV 的 98.78% 直接宣稱優於原版完整 1230 輛 EV 的 59.19%；須完成相同訓練預算及完整 validation 後再比較。

模型張量雜湊：`01550bb3337e172b0aa23257819e59eee351fd2c4feb292ae94a06cb8798de44`。原始比對結果保存在本次工作輸出的 repeatability.json。

另外經正式 CLI 在 new_env 執行 1 個 episode 接線測試，成功完成並有 70 次 minibatch 梯度更新、無整回合執行錯誤；不是 Berlin 的性能評估。本次沒有改動原有 SUMO 安裝位置偵測的本機／可攜版差異；指令明確指定決策上限 80，以免兩份程式先前不同的預設值影響結果。

既有完整測試探索有兩個 legacy alias 匯入錯誤（`count_evs_in_rou`）及一個 Dijkstra 假路網測試失敗，已在修改前的程式副本重現。排除這三個既有問題模組後，254 項測試通過，包含新增道路特徵、候選排序等變性、實際梯度更新、模型狀態還原、RNG 保存還原、seed 設定檢查與 validation／testing 預算參數。

先前發現 RL 與 Dijkstra 對 EV teleport 的失敗計分方式不同，此次沒有改變這個評估定義。正式比較旅行時間時仍需揭露或統一處理；不能因為少數 EV teleport 就刪除整個測試場景來提高結果。
