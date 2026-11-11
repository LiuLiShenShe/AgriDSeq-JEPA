# S2 官方 MAE checkpoint 来源

实时核对 MAE 官方远端 HEAD：`efb2a8062c206524e35e47d04501ed4f544c0ae8`。
[官方固定 SHA README](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/README.md) 的 pre-trained ViT-Base 下载链接指向本次文件。
[官方原始自监督预训练权重](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)，没有使用 ImageNet 分类微调权重。
开始时检查目标pretrained目录，并检索 `/fj` 可读目录，未发现该文件；使用 curl 从官方地址下载，原文件保留于 `runs/pretrained/mae_pretrain_vit_base.pth`。

实际大小 **343,249,461 bytes**；完整 SHA-256 **`aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`**。
顶层只有 `model`，内部150个FP32 encoder张量：patch embedding2、CLS1、pos1、12 blocks×12、final norm2。
此发布文件已经去除 decoder；没有 mask_token、decoder、分类 head 或 optimizer。原始150项形状与dtype清单保存 `runs/s2_source/checkpoint_structure.json`；映射完整清单见 `reports/s2_mae_weight_mapping.json`。

[官方模型定义](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py)：ViT-B/16，224预训练输入，768维，12层，12 attention heads，MLP ratio4，QKV bias，LayerNorm eps1e-6；训练模型另含512维8层 decoder。本次不实例化该网络，仅复用发布 encoder 初始化当前 S1 分割实现。

博查搜索实际调用返回HTTP403 `You do not have enough money or package quota`，搜索不可用。上述信息来自实时读取的官方固定SHA源码、官方文件及本地反序列化；没有用训练记忆代替来源。PyTorch文档由Context7获取：[v2.5.1 state_dict文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/modules.rst)。
