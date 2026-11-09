## 路況查詢與靜態號誌連線快取（2026-11-09）

`proposed` 的 `static_occupancy` 現在只查詢全路網 occupancy，其他模式依需要收集動態欄位；八欄 edge 特徵格式不變。每個 episode 在載入場景後重新建立靜態 movement→TLS/link index 與 outgoing lane 快取，即時相位、等待時間與車流數仍即時讀取。

在同一台 RTX 2060 電腦，以相同兩個車流、SUMO seed、探索 seed、80 次決策上限與 6000 模擬步數測試，warmup episode 平均從 176.97 秒降到 63.95 秒（減少 63.86%）。每個 episode 的 TraCI 指令平均從 446 萬次降到 158 萬次。這兩個案例逐次 observation、選路與 replay transition 數值的雜湊相同，逐車旅行時間、失敗原因、決策數與模擬步數也相同。54 項相關回歸檢查通過。

量測涵蓋 epsilon=1、尚未 learning 的 warmup 階段，不含首次初始化、存檔或繪圖；其他電腦及開始梯度更新後的實際加速比例需另行量測。本次不修改 `config.py`、模型、資料或訓練排程。

在新電腦的專案根目錄執行 `git pull --ff-only` 更新。若本機只修改了 `config.py` 的決策上限，以及 `DQN/dqngnn_action_re.py` 的模型載入參數，這次更新沒有觸及這兩個檔案。Python 需要重新啟動才會載入修正，可使用既有 `--resume-state` 接續訓練。
