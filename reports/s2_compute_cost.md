# S2 计算成本（实测）

| 度量 | DSeq-JEPA | MAE |
|---|---:|---:|
| 训练+10次周期val墙钟(s) | 12155.94 | 16232.47 |
| 平均optimizer update(s) | 0.8683 | 1.2500 |
| 训练allocator峰值(GiB) | 3.5138 | 3.5214 |
| 训练reserved峰值(GiB) | 3.7656 | 3.7930 |
| 最终best全量评价(s) | 2233.40 | 406.73 |


MAE训练及最终评价合计 4.6220任务运行小时。
同一物理GPU1 A6000/BF16/activation checkpoint；CPU threads4/workers8。
MAE额外的neck参数更新检查保留了neck.stages.0.0.weight抽样clone（2359296个FP32值，9MiB）；allocator比较包含这项审计内存，模型本身的参数数量和形状完全一致。
MAE前4000步平均0.8895s/update，后6000步平均1.4902s/update；另一个GPU1进程在第4000步val期间启动，具体实时进程证据见s2_runtime_events.md。时间分段是环境观察，不能作为初始化导致速度差异的因果证据。
MAE设备遥测峰值显存47520.9MiB、温度89.0°C；10秒采样区间16680.85s，板卡功率积分1.17744kWh。设备采样涵盖训练、最终评价及分析启动，含其它进程时不能全部归因于本任务；不等于整机能耗/电费。任务allocator峰值更适合比较模型显存。
S1设备遥测存在共享任务及缺失起始170秒，不能把设备能耗差直接解释成模型差异；本轮时间同样受到设备调度和I/O影响。Smoke20步单列，不算正式10000步。
来源：两条真实train_log/status/evaluation_provenance、gpu_telemetry.csv；S1固定成本见s1_gpu_compute.md。
