# 最终 best-val 农业失败分析

对象为step10000的best checkpoint，重新导出的772张1024×1024预测；官方mIoU90.02%、Weed IoU74.62%。先完成全量数值统计，再直接查看实际RGB、GT、预测及红色错误叠图。没有使用生成图或外部识图API。

全val杂草Precision88.4015%、Recall82.7191%。TP3,537,657、FP464,149、FN739,054；其中547,960个漏分像素被判为soil，191,094个被判为crop，分别约占FN的74.14%和25.86%。

GT weed四连接组件面积≤100px、且至少50%像素被识别为weed的组件召回为44.8276%（494/1102）。这是像素连通区域统计，不能等同于真实植株实例数或物理面积。它说明整体Weed IoU之外，微小区域仍明显困难。

| 真实val成员 | Weed GT像素 | TP / FN | Weed IoU | 直接观察与整数标签证据 |
|---|---:|---:|---:|---|
| 05-15_00202_P0030953.png | 125 | 1 / 124 | 0.52% | 土块纹理、强烈植株投影的图像中，上边缘小weed区域几乎漏分；123px被判soil。大作物轮廓总体识别较好。不能仅凭此样例认定阴影导致错误。 |
| 05-15_00253_P0030792.png | 348 | 5 / 343 | 0.67% | 稀疏小苗与土壤背景中存在crop/weed互相混淆：227个weed像素判crop，291个crop像素判weed；小目标类别判断比整体前景分割更困难。 |
| 05-26_00242_P0034016.png | 22316 | 12606 / 9710 | 54.31% | 左侧相邻、交叠叶片区域的GT weed大片被并入crop；7837个weed像素判crop，1873个判soil。图中另一些孤立杂草能够正确识别。 |
| 06-05_00145_P0037991.png | 38309 | 29808 / 8501 | 74.38% | 较密植株区域仍有weed/crop混淆，6280个weed像素判crop；叶缘及细长结构可见局部漏分和过分，红色误差沿部分边界分布。 |

这些是按val误差排序选取的诊断成员，不代表独立test，也没有按遮挡、阴影或边界难度建立人工分层标注；因此不报告这些条件下的独立精度或因果结论。阴影、复杂土块背景和相邻叶片是实际可见的现场条件。

全部成员统计见`s1_per_image_errors.csv`，汇总见`s1_error_summary.json`。真实四栏图仅保留在Git忽略目录`runs/phenobench_dseq_vitb16_s20261110/analysis/*_case.png`，四栏为RGB、GT、Prediction、Errors；显示调色板brown=soil、green=crop、magenta=weed，错误为red。显示图与提交官方评价的单通道ID PNG不同用途。

相同772val、原1024坐标下，对官方公开预测重新运行同一devkit：本方法mIoU90.02/Weed74.62%，ERFNet87.77/68.65%，DeepLabV3+87.47/68.00%。相对ERFNet分别提高2.25和5.97个百分点；相对DeepLabV3+分别提高2.55和6.62个百分点。公开baseline的backbone、训练crop、损失权重和优化预算不同，本方法还经过val选择；这证明当前农业迁移结果具有竞争力，差值不能单独归因于JEPA初始化。

下一阶段优先MAE ViT-B/16公平对照：本轮已证明真实迁移闭环并得到较高val结果，应先固定backbone容量、UPerNet/aux、增强、crop、BN统计、LR、有效batch、updates和val选择协议，仅改变初始化，判断DSeq-JEPA表征贡献。随后再用CropAndWeed检验外部泛化。本轮未启动这两项实验。

来源：本项目实际最终预测、官方devkit结果、全val整数标签统计、直接查看真实对照图；baseline下载、SHA及协议见[published_baselines_val.md](published_baselines_val.md)。
