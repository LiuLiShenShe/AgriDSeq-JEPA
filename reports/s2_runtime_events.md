# S2 实际运行事件

2026-11-10 09:37 UTC（训练第4000步评价完成后）：NVML显示物理GPU1上本训练PID863354约4224MiB，另一个PID1304045约10336MiB。整卡显存14581MiB、利用率100%、温度88°C；本任务allocator峰值3.5214GiB。其它进程不由本任务管理。后续成本报告区分allocator和设备遥测；共享GPU可影响训练和评价墙钟，不能将整卡功率/显存全部归因于MAE。

来源：实际nvidia-smi进程与设备查询、run/gpu_telemetry.csv、逐步train_log.csv。

2026-11-10T10:45:28.493654+00:00（约第6163步）：再次查询进程如下；最近update_seconds回到约0.86s。

```text
pid, gpu_uuid, used_gpu_memory [MiB]
4150817, GPU-db3ec610-630b-fa4f-66f2-b69f269ca856, 2702 MiB
3372944, GPU-db3ec610-630b-fa4f-66f2-b69f269ca856, 360 MiB
863354, GPU-6836f890-98b2-5891-07ef-675c25199c65, 4224 MiB
1684866, GPU-6836f890-98b2-5891-07ef-675c25199c65, 1156 MiB
```
