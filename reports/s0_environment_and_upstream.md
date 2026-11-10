# S0-A：环境与上游

工程 `/fj/AgriDSeq-JEPA`，独立于历史只读工程 `/fj/jepa-二维图像任务`。
新远端 https://github.com/LiuLiShenShe/AgriDSeq-JEPA （私有），账号 LiuLiShenShe。
分支 `agri-phenobench-transfer`。上游 https://github.com/SkyShunsuke/DSeq-JEPA 完整 SHA `123082374389f78e4516157187681f9d5edb4310`；fetch remote 为 upstream，push URL 为 DISABLED。保留所有原始作者代码及 MIT LICENSE，Copyright (c) 2025 Swapman。原版 requirements.txt 未改动。

初始磁盘 `/fj` 607GB 可用；原始数据位于独立 `/data` 分区，约1.1TB可用。
设备 NVIDIA RTX A6000：物理GPU1，49140MiB，UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`；初始14MiB、0%利用率，无计算进程。GPU0 UUID `GPU-db3ec610-630b-fa4f-66f2-b69f269ca856` 当时已有PID4150817和3372944，本任务未终止或使用这些进程。
启动固定 `CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0` 即物理GPU1。驱动590.48.01。

独立 `uv venv --python 3.11 .venv`，Python3.11.15，torch2.5.1+cu124，torchvision0.20.1+cu124，CUDA12.4；BF16设备能力实测为True。官方全部requirements安装到新环境；补充numpy1.26.4、pytest9.2.0、官方devkit/torchmetrics0.10.3。为兼容devkit的pkg_resources，固定setuptools80.9.0。全部解析依赖见 `configs/agriculture/requirements-lock.txt`；旧环境没有被修改。

复用：build_pretrained_encoder、load_jepa_target_encoder_weights、ViTDenseBackbone、ViTMultiScaleNeck、ViTUPerNet、主/辅CE、get_finetune_optimizer、PolyLRSchedule、slide_inference、ADE20K图像增强helper。
新增：PhenoBench adapter/factory注册；农业配置、训练/评价/审计脚本、测试、轻量报告。最小核心修正：官方公开裸state_dict兼容分支；slide_inference显式BF16。
没有复制旧项目的模型、UPerNet、预训练训练器或复杂框架。数据/权重/预测/run全部Git忽略，现有大型原始数据直接读取。

用户引用的 `/home/test/.codex/RTK.md` 实际不存在，未声称读取。Bocha实测HTTP403余额/配额不足；一手证据来自实时克隆的上游和用户指定官方页面。Context7已实际先library后docs核查PyTorch2.5.1、GitHub CLI及PhenoBench文档。

来源：[官方代码](https://github.com/SkyShunsuke/DSeq-JEPA/tree/123082374389f78e4516157187681f9d5edb4310)、[PyTorch CUDA环境文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/cuda_environment_variables.rst)、本机nvidia-smi/df/uv真实执行记录。
