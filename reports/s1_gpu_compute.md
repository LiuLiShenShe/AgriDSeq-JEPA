# S1 GPU成本（实测）

状态 TRAINING_IN_PROGRESS，正式optimizer updates 4420/10000，microbatch2×accum8=有效16。
物理GPU1 RTX A6000，UUID GPU-6836f890-98b2-5891-07ef-675c25199c65；BF16+作者activation checkpoint。
训练和周期val累计墙钟 5081.37s (1.4115 单GPU任务运行小时，非独占GPU小时)；平均每optimizer update 0.8710s。
训练allocator峰值 3.5138GiB，reserved峰值 3.7656GiB；NVML监控峰值 10456.9MiB。
10s采样遥测区间 4910.25s，GPU板卡功率梯形积分 0.32648kWh。初始GPU1空闲，但训练期间另有用户任务启动，设备显存/功率含其他进程，不能全部归因于本实验，也不含整机CPU/磁盘能耗或代表电费账单。PyTorch allocator峰值属于本任务。遥测开始于正式训练约170秒后；不外推缺失区间功耗。实测设备温度峰值89.0°C。遥测原始CSV在Git忽略的run中。
烟雾工程检查另行记录reports/smoke_verification.json和smoke_resume_verification.json，不计正式10kupdates。最终best重新评价计入持续遥测区间，训练status耗时终点为训练完成。

来源：实际train_log.csv、status.json、gpu_telemetry.csv；没有估算未执行的更新。
