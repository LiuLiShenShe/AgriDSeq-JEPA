# S2 同一分割架构与初始化审计

S1冻结SHA `20ffaadcfa0f1172303c2ccbc09a456458f80e3d`；真实S1训练启动SHA `399f9a84fe09471fe3d990f68734ec096fe540ca`。
通过 `git show` 读取启动时训练脚本并执行其原始build_model；model/dataset/optimizer依赖在启动SHA和冻结SHA间未变。
S2沿用同一 `init_target_encoder → pretrained load → init_segmentation_model` 顺序、seed20261110和CUDA_VISIBLE_DEVICES=1。
新增分支仅在 `src/eval/common.py` 的加载器选择；默认DSeq加载语句保留。MAE不实例化官方MAE网络，不改变模型初始化所消耗的随机数，不加CLS前向。

| 部分 | 总参数 | 可训练参数 |
|---|---:|---:|
| backbone | 85,797,888 | 85,646,592 |
| neck | 7,081,728 | 7,081,728 |
| decode_head | 32,257,539 | 32,257,539 |
| auxiliary_head | 1,770,755 | 1,770,755 |
| 合计 | 126,907,910 | 126,756,614 |

两个模型repr、参数名/形状/可训练状态完全相同。唯一冻结参数为pos_embed和无空间作用的cls_token，符合S1。全12层、patch/norm实际加载MAE；150/150严格加载，无缺失、无随机兜底。MAE patch位置 `[1,197,768][:,1:,:] → [1,196,768]`；保留S1 bicubic插值到32×32。CLS参数来自MAE但冻结且不传入dense前向。
attention是同一MaskedAttention（scaled_dot_product_attention）；12 heads、QKV合并顺序相同、LayerNorm eps1e-6、MLP ratio4/GELU；drop_path0.1、activation checkpoint、window_size0、out_layers[3,5,7,11]相同。
真实val RGB裁剪前向得到四层 `[1,768,32,32]`，logits `[1,3,512,512]`。20步smoke另验证batch2和真实forward/backward/optimizer更新。

Neck8、decode_head74、auxiliary_head8项初始化state（含BN buffers）全部torch.equal，逐张量SHA见 `s2_initialization_evidence.json`。模型创建后的CPU与CUDA RNG摘要相同；正式启动另外保存同一组初始张量SHA并核对，避免只以seed推断张量一致。
证据边界：S1没有独立存档step-zero张量，因此可证明S1固定启动代码/config/seed重建张量与S2相等，不能声称直接读取并验证历史S1原始step-zero存档。

运行审计：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python scripts/agriculture/verify_mae_control.py`。
针对性测试：6项通过，覆盖严格全加载、CLS位置切除、decoder白名单排除、RNG不变及缺失/未知/错误形状/非有限值拒绝。
来源：[官方MAE模型定义](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py)，本地固定代码及实测审计JSON。
