# 正式运行期资源记录

正式训练始终为同一条轨迹、同一 Python 训练进程 PID3322358；10000 optimizer updates 与80000 microbatches已由完整checkpoint独立核验，未追加第二条训练。

2026-11-10 06:45:28 UTC前后核查发现，启动控制会话和旧只读进度监控会话返回143；Python训练子进程仍存活、PPid变为1，GPU遥测进程PID3340613继续运行。本项目没有主动终止训练或其他科研进程。另建只读监控观察原进程，并未重启训练。

10k周期验证曾暂时停在400/772张预测。实际读取进程状态为D、wchan为folio_wait_bit_common；数据磁盘/data使用/dev/sda1 ext4，jbd2/sda1-8及另一Python进程也出现磁盘等待。连续读取/proc/diskstats确认设备计数仍推进。内核stack/syscall/dmesg读取因权限不足不可用，因此不推断硬件故障或数据损坏。随后预测实际推进到500、600、700/772。该I/O等待计入真实训练及周期验证墙钟耗时。

运行期间GPU1有其他科研任务。本任务不终止其他进程；NVML设备总显存、功率及积分能耗不能全部归因于本任务，PyTorch allocator峰值独立记录。

来源：本机/proc进程和磁盘状态、nvidia-smi、run/console.log、run/gpu_telemetry.csv，以及s1_checkpoint_state_10000.json。

最终原训练子进程自然退出，status.json记录TRAINING_COMPLETE、10000updates、训练及周期验证12155.9408s。随后独立恢复best(step10000)，全部772张预测重新生成，官方mIoU90.02%，与周期结果及confusion交叉核对一致。该独立评价实际耗时2233.4019s，期间同样有间歇性I/O等待。CPU误差分析完成后，仅停止本项目专属GPU遥测进程PID3340613，未终止其他科研任务。
