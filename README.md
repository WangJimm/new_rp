# new_rp：可搬移的 SUMO 路徑規劃訓練套件

包含 RP 訓練程式與 Berlin `new_env` 完整資料，供另一台 Windows 電腦重新開始訓練。原始程式目錄與正在運行的訓練不受本套件影響。

## 已包含

- `rp/`、`DQN/`、`scripts/`、相容入口及原始程式測試。
- `new_env/env_static_tls.net.xml` 路網與 `osm.sumocfg`。
- training：210 份背景、420 份 EV、420 份 goal。
- validation：45 份背景、90 份 EV、90 份 goal。
- testing：45 份背景、90 份 EV、90 份 goal。
- 2650／2800／2950 各需求內均為 70／15／15；p01／p02 與同背景共用 split。
- `scenario_manifest.csv` 保留所有需求、比例、seed 的既有分配，不重新切分。
- 目標 junction：1419061109、446290091、5984003368。

不包含已訓練模型、replay memory、日誌、輸出、Python 虛擬環境與原 Brooklyn `env` 資料。若要接續目前的訓練，另行安全搬移相同方法的 `training_state_latest.pt`，並使用 `--resume-state`；不需把模型公開上傳。

## Windows 安裝

安裝 64-bit Python 3.12（或 3.11），並安裝 Git。在 PowerShell 執行：

```powershell
git clone https://github.com/WangJimm/new_rp.git
Set-Location new_rp
```

先驗證檔案（不需要第三方套件）：

```powershell
python tools/verify_package.py
```

顯卡不確定時先安裝 CPU 版本，不會自動開始訓練：

```powershell
python tools/setup_environment.py --torch-build cpu
```

安裝器會在 repository 內建立 `.venv`，不需要變更 PowerShell execution policy，也不需要啟用 venv。

檢查安裝、資料、GPU 狀態（不啟動 SUMO）：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.check_installation --env new_env
```

準備好後開始訓練：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.run_experiment --env new_env --rp_variant proposed
```

預設 1500 episodes；可加 `--num-episodes 1500`、`--use-gui` 或選擇 `--rp_variant edqn_baseline`。使用 `--env new_env`，因為本套件未攜帶原 Brooklyn `env` 資料。

輸出在 `outputs/new_env/<rp_variant>/`；SUMO logs 在 `logs/new_env/`，均已排除 Git 追蹤。程式會自行偵測此設備上的 SUMO，不需沿用原電腦的 D: 路徑。

## NVIDIA GPU（確認型號後再選）

檢查系統是否能找到顯卡：

```powershell
nvidia-smi
```

若顯卡與驅動支援此 PyTorch 版本，可選擇官方 CUDA 12.1 build：

```powershell
python tools/setup_environment.py --torch-build cu121
& '.\.venv\Scripts\python.exe' -m scripts.check_installation --env new_env --require-cuda
```

也提供 `--torch-build cu118`。新的 NVIDIA 顯卡可能需要更新的 PyTorch/CUDA，應依確切型號再調整；CPU 安裝成功不代表 GPU 已啟用。安裝器固定 PyTorch 2.3.1 以貼近來源程式，未在另一台設備驗證。

## 模型驗證與最終測試

proposed 的完整到達率最低成本 checkpoint，需 validation 全部 EV 到達才會產生：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.validation --env new_env --rp_variant proposed --checkpoint '.\outputs\new_env\proposed\gine_action_best_100pct_arrival_lowest_cost.pt' --output '.\outputs\new_env\proposed\validation_result.json'
& '.\.venv\Scripts\python.exe' -m scripts.evaluate --env new_env --rp_variant proposed --checkpoint '.\outputs\new_env\proposed\gine_action_best_100pct_arrival_lowest_cost.pt' --output '.\outputs\new_env\proposed\testing_result.json'
```

只查看場景分配：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.verify_training_validation_split --env new_env
```

## 版本與可重現性

來源設備：Python 3.12.3、PyTorch 2.3.1、PyG 2.6.1、SUMO 1.24.0。安裝器採 NumPy 1.26.4，避免來源設備 NumPy 2.x 與舊 PyTorch 的 NumPy bridge 相容性問題。主要套件固定版本；完整 transitive dependencies 由 pip 解析，本套件不是逐套件鎖定的容器。

步長維持 0.5 秒、上限 6000 steps（3000 秒）、每輛 EV 最多 40 次決策。未改變演算法、reward、資料切分或路線內容；改動僅限跨設備路徑尋找與包裝。

其他方法如 `ppo_torch_baseline` 不需要 Ray；`ppo_baseline` 為 RLlib 版本，需要另外安裝相容的 Ray/RLlib，未納入此預設環境。

已通過純 Agg PNG 繪圖（包含工作執行緒及垃圾回收）與既有工具相容性測試，並完成語法與資料完整性檢查。未執行訓練、SUMO、套件安裝或完整測試套件。依系統與硬體差異，不保證不同設備達成逐位元相同的訓練結果。

Linux 可使用同一份程式與安裝器，執行路徑改為 `.venv/bin/python`；主要交付流程以 Windows 為主。

資料來源與授權資訊見 [DATA_NOTICE.md](DATA_NOTICE.md)。安裝指令參考 [PyTorch 官方舊版本](https://pytorch.org/get-started/previous-versions/) 與 [SUMO 官方 Python 安裝說明](https://sumo.dlr.de/docs/Downloads.php#python-packages--virtual-environments)。

## 繪圖崩潰修正與續訓

訓練與 validation 只輸出 PNG，啟動時固定 Agg；simple_plot 直接使用 FigureCanvasAgg，不建立 Tk/Qt 視窗。此設定不影響 SUMO-GUI。

若舊版在 validation 開始時出現 Tkinter `main thread is not in main loop` / `Tcl_AsyncDelete`，更新程式後啟動新 Python process，並使用完整狀態續訓：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.run_experiment --env new_env --rp_variant proposed --resume-state '.\outputs\new_env\proposed\training_state_latest.pt'
```

完整續訓保留模型、optimizer、replay、場景排程和已記錄的指標。只載入 tmp_file.pt 不等同完整續訓。
若想先單獨檢查中斷時的模型，可執行：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.validation --env new_env --rp_variant proposed --checkpoint '.\outputs\new_env\proposed\tmp_file.pt' --output '.\outputs\new_env\proposed\recovery_validation_result.json'
```

validation 不會因改用 Agg 而更動場景、goal、到達率門檻或評估公式。此修正已做繪圖回歸測試，完整 90 場景動態 validation 仍需由實際訓練／驗證執行確認。

## Validation 合法轉向修正（2026-11-06）

若 validation 出現 `EV_2` 的 `1147628417 -> 1451112465` 無連接錯誤，原因是舊 Dijkstra 只使用路口鄰接，忽略 edge 之間的合法轉向；這組起點與目標其實可達，需要繞行。現在改用 incoming edge 為搜尋狀態，逐段檢查 SUMO connection 與車種權限，也允許從另一條道路再次抵達同一路口。Dijkstra baseline 仍使用 TraCI 即時 edge travel time 作為權重，並依實際車輛 class 搜尋；共享的 Dijkstra 導引與距離查詢也使用相同合法轉向規則。

不需重訓即可用現有模型，直接跑 90 個 validation 場景（含 Dijkstra 比較）：

```powershell
& '.\.venv\Scripts\python.exe' -m scripts.validation --env new_env --rp_variant proposed --checkpoint '.\outputs\new_env\proposed\tmp_file.pt' --output '.\outputs\new_env\proposed\turn_fixed_validation_result.json'
```

請啟動新的 Python process 才會載入修正。`tmp_file.pt` 是每 100 個訓練 episodes 保存的模型，可能早於最新 autosave；`training_state_latest.pt` 是完整續訓狀態，不可直接當成這個 `--checkpoint` 使用。此指令只做推論與比較，不執行訓練，也不更新最佳模型檔。

已通過 22 項相關回歸測試（路徑搜尋、轉向限制、車種權限、baseline 與相容介面），另以獨立反向 BFS 檢查 600 場景的 12,700 筆 goal 指派全部可達，並驗證 1,876 組不同起點／目標的修正版路徑均符合合法 connection。完整 90 場景動態 validation 尚未執行；圖形連通不保證車輛不壅塞或不 teleport。資料、模型權重、需求、seed 和 split 未變更。

## 訓練期間的 validation 模型與結果保存（2026-11-07）

每次排程中的 validation 成功完成後，會先保存當時模型，再以相同檔名保存完整結果。未通過 100% 到達門檻也會保留；validation 執行失敗則不建立成功結果。

以 `proposed` 為例，預設輸出為：

```text
outputs/new_env/proposed/
  validation/
    episode_001200.pt
    episode_001200.json
    episode_001350.pt
    episode_001350.json
    episode_001500.pt
    episode_001500.json
  gine_action_best_arrival.pt
  gine_action_best_arrival.json
  gine_action_best_100pct_arrival_lowest_cost.pt
  gine_action_best_100pct_arrival_better_ratio.pt
```

`episode_001500` 表示第 1500 個已完成訓練 episode。模型內部 `episode` 欄位仍為從 0 起算的 1499；JSON 另列 `episode_number: 1500`，並記錄對應 checkpoint 路徑。完整 JSON 保留每場景成本、逐車成本、失敗原因、Dijkstra 比較、路徑與 teleport 統計等原始 validation 輸出。JSON 中非有限數值以 `null` 表示。

新增的 `gine_action_best_arrival` 先比較整批 validation 的 EV 到達率，到達率相同時比較總懲罰成本，較低者勝出；兩者完全相同時保留先前模型。這個模型可以在未達 100% 時保存，旁邊的 JSON 是該模型對應的完整評估結果。其他方法使用其 agent type 作為前綴。

原本兩個 `best_100pct_arrival` 模型仍依原規則，只有全部 EV 到達時才挑選／更新。`training_rewards.json` 與完整 autosave 會保存新增排名狀態及 validation 模型／結果索引。更新前的結果不會自動回填。

此功能自動啟用，原訓練指令即可使用。3000 episodes 且維持 225／1200 探索邊界的範例：

```bat
python -m scripts.run_experiment --env new_env --rp_variant proposed --num-episodes 3000 --epsilon-warmup-episodes 225 --epsilon-decay-end-episode 1200 --scenario-schedule-seed 42
```

validation 頻率、場景資料與評分公式沿用原設定。每次保存的是模型 checkpoint，不含完整 replay memory；完整續訓仍使用 `training_state_latest.pt`。以來源 proposed 模型約 4.7 MB 計算，3000 episodes 的 13 次 validation 歷史模型約增加 61 MB，另加 JSON 與最佳模型檔；實際大小依模型與結果而異。

已以 17 項測試驗證模型可載入、完整結果配對、排名優先順序、平手處理、100% 原規則、寫入異常及排名狀態保存。測試未執行訓練或啟動 SUMO。
