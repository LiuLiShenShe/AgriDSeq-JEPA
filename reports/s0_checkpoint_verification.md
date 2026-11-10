# S0-B：真实作者预训练权重

状态：实际下载、CPU与新环境GPU严格加载、512×512四层特征/UPerNet前向通过。
文件 `runs/pretrained/dseqjepa_vit_base_latest.pth`，343239900 bytes。
完整SHA256 `3e3ca279c9a4929bfb66c10c3a21f92a702899bb556cda9e528ee57e72a2cc5e`。
下载来源：https://drive.google.com/uc?export=download&id=1CjueAvk52g5YIf30yI7ZFDIbsfqYq5tt ，来自作者README链接的公开Drive目录，经实际确认下载表单取得。原文件保持原样。

公开文件是150个Tensor的裸state_dict，不含encoder/target_encoder包装。本轮为作者load_jepa_target_encoder_weights新增明确的裸tensor字典分支，仍strict=True；build_pretrained_encoder直接调用该函数。CPU严格架构检查All keys matched successfully，GPU同一官方构建路径无missing/unexpected key错误。

官方 `configs/dseqjepa/vitb16_in1k.yaml` 指定vit_base、patch16、224预训练网格、use_class_token=True；`src/frameworks/dseqjepa/pretrain.py:179`明确use_masked_vit=True。真实文件含cls_token[1,1,768]、pos_embed[1,196,768]、patch_embed.proj.weight[768,3,16,16]、12 blocks，均匹配。详细证据 `reports/checkpoint_source.json`。

512输入经位置编码插值产生32×32 patch网格。GPU实测0-index层3/5/7/11各[1,768,32,32]；官方neck输出[1,768,128,128]/[1,768,64,64]/[1,768,32,32]/[1,768,16,16]；decoder logits[1,3,512,512]。autocast指定BF16，最后插值与加位置编码可提升输出为FP32，这不等同于FP16或伪BF16。

原作者dense backbone保留固定sin-cos位置编码，并丢弃/冻结无空间位置的class token；除此之外所有活跃backbone参数与neck、decode_head、auxiliary_head参与优化，freeze_backbone=False。这两项冻结是官方实现既有行为，不是Adapter或冻结backbone训练。

权重不含epoch/config/optimizer元数据，因此不能声称该文件的精确预训练epoch或数据子集经过文件验证；官方yaml中的600epochs/10%子集是公开配置，非权重自身证明。没有使用旧I-JEPA、MAE、DINO或随机权重代替官方模型。

来源：[作者代码/权重入口](https://github.com/SkyShunsuke/DSeq-JEPA/tree/123082374389f78e4516157187681f9d5edb4310)、[原论文](https://arxiv.org/abs/2511.17354)、[实际下载文件](https://drive.google.com/uc?export=download&id=1CjueAvk52g5YIf30yI7ZFDIbsfqYq5tt)。
