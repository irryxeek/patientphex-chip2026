# PatientPheX 第二轮优化及线上评测

日期：2026-10-20。已完成 CPU 优化、完整 A/B 预测、模型重载复核和天池赛后评测。
当前选用 `runs/optimization_v1/full/`；后续患者状态补充实验未通过开发集验收，未替换模型。

## 天池实际结果

2026-10-20 20:06:47（北京时间）从登录后的
[我的成绩](https://tianchi.aliyun.com/competition/entrance/532512/score)核实：
阶段为 **赛后评测，seasonId=1862**，总分 **0.6117086747298865**，当时排名 **26**。
该排名是查询时的赛后榜快照，不是正式赛期名次或获奖结果。

| 官方指标 | 原始值 | 百分制 |
| --- | ---: | ---: |
| Task 1 mention F1 | 0.6722107742125404 | 67.2211 |
| Task 1 document F1 | 0.7385998715478485 | 73.8600 |
| Task 2 micro F1 | 0.5557564798071127 | 55.5756 |
| Task 2 macro F1 | 0.4802675733520447 | 48.0268 |
| 四项均值 | 0.6117086747298865 | **61.1709** |

提交文件为 `submissions/20261020_optimization_v1_b.jsonl`，797,011 字节，SHA256：
`b4ced01f30e193c6dd08336bdfcfc7b81ee7336bf95c23581e1ca92542d92ad9`。
覆盖 B 集全部 100 篇文献、244 位指定患者，包含 6,147 个实体、270 个否定实体和 2,393 条患者表型关联。

页面延迟回显期间重复提交了一次，同一个文件产生两条记录：

| 提交时间（北京时间） | 文件 ID | 成绩记录 ID | 状态 |
| --- | ---: | ---: | --- |
| 2026-10-20 19:51:04 | 2793245 | 3051373 | 测评完成 |
| 2026-10-20 19:55:51 | 2793249 | 3051377 | 测评完成 |

两条记录的四项 F1 及总分相同；不将其作为两次独立实验。查询时剩余提交次数为 3。
排查确认上传 HTTP 200 后，后台评测记录存在回显延迟；最终以服务端文件/成绩记录为成功依据。
已停止重复上传。浏览器已恢复正常页面，未导出登录凭证。

脱敏回执见 `submissions/20261020_receipt.local.json`，
文件清单见 `submissions/20261020_optimization_v1_b.manifest.local.json`，
页面证据见 `submissions/20261020_score_screenshot.png`。
上述本地目录被 Git 忽略，不含签名下载地址、Cookie 或上传凭证。

## 方法及选择过程

继续沿用原基线的 64 篇拟合 / 16 篇开发划分；开发集包含 54 位患者。
原始数据、评分程序、金标准异常处理及外层划分均未改动。
优化协议和补充协议分别见 [OPTIMIZATION_PROTOCOL.md](OPTIMIZATION_PROTOCOL.md)
及 [PATIENT_REFINEMENT_PROTOCOL.md](PATIENT_REFINEMENT_PROTOCOL.md)。

实体提取加入训练别名精度过滤、保守单复数匹配和文献内部明确括号定义的缩写。
拒绝全局单词长度小于 4 的训练别名；短缩写只在有文内定义时恢复，并区分大小写，
防止将 `He/he` 等代词识别成表型。所有匹配保留原文 Unicode 字符偏移。

在 64 篇拟合文献内按 PMID/相同全文分组做四折验证，seed=1729。
五种预定实体配置依据内层合并 mention/document F1 均值选择。
选中 `filter050_morph_abbr`：训练别名精度阈值 0.50，开启词形和文内缩写。

患者关联使用 HistGradientBoostingClassifier，输入为候选表型与指定患者的数值上下文特征：
同句/同段、显式指称、患者及竞争患者距离、章节、跨段最近患者、提及频次等。
不把文献 ID、患者 ID 或 HPO ID 本身作为记忆特征，不读取推理输入的金标准字段。
固定 150 次迭代、学习率 0.05、最多 15 叶、叶节点最少 20 个样本、L2=1、seed=42。
在内层验证比较原规则与阈值 0.20/0.30/0.40/0.50/0.60，选中分类器及阈值 **0.20**。
配置和选择记录在读取本轮外层标签前保存；开发集在前轮已被查看，仍属于开发比较。

全部计算在 CPU 完成；优化搜索与全量导出共约 74.26 秒。
随后用全部 80 篇官方训练文献拟合，生成 A/B 预测。
没有启动 GPU、收费模型 API 或在线实验跟踪，没有使用额外人工标注数据。

## 本地开发集比较

以下分数使用本地参考评分器，均为百分制。它们与上方官方 B 集成绩来自不同集合，
不能用两者差值推断性能提升或评分器一致性。

| 方法 | mention F1 | document F1 | patient micro F1 | patient macro F1 | 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原训练别名基线 | 65.06 | 69.59 | 41.08 | 38.93 | 53.66 |
| 改进实体 + 原关联规则 | 68.95 | 72.84 | 43.45 | 41.05 | 56.57 |
| 改进实体 + 关系分类器（选用） | 68.95 | 72.84 | 51.92 | 37.07 | **57.69** |
| 患者状态补充实验（未采用） | 68.95 | 72.84 | 49.78 | 35.45 | 56.76 |

选用方法相对原基线总分增加 4.03 点。10,000 次配对文献 bootstrap、seed=42，
总分差的 95% 区间为 [0.49, 8.27]；patient micro 差为 +10.84 [4.03, 17.57]，
**patient macro 差为 −1.86 [−14.77, 10.37]**。
开发集仅 16 篇且经过多轮观察；区间不覆盖方法选择、评分定义及标注系统误差。
总分改善不能掩盖患者宏平均退步。

为改善宏平均，补充比较了患者健康/受累状态特征，以及每位患者等总训练权重。
内层选中的状态特征方案在外层反而降至 56.76，相对选用方案差为 −0.94 点，
配对区间约 [−1.70, −0.25]，因此未采用。完整 43.70 秒实验及全部候选保留在
`runs/patient_refinement_v1/`，没有选择性删除失败结果。

## 仍待解决的错误

选用模型在开发集的实体 TP/FP/FN 为 857/241/531，患者关联为 237/208/231。
实体漏检中，363 个没有重叠候选，122 个边界错误，25 个被判否定，21 个同边界 ID 错误。
患者关联漏检中，181 个概念未抽取，50 个已有概念未分配；误关联中，
85 个属于其他指定患者，123 个不是该文献的金标准患者关联。

这些统计支持继续改进上下文实体召回及跨患者证据判断。只增加关联召回会带来较多误关联，
需要同时观察患者宏平均。Qwen3-8B 或更小公开模型可作为下一轮候选，
本任务 GPU 使用范围尚待用户指定；未沿用其他实验的 GPU 授权。

## 复现与工程核验

当前权重、词典和推理配置位于 `runs/optimization_v1/full/`，
其对应推理源码位于 `runs/optimization_v1/source/`。
根目录 `enhanced.py` 另包含未被采用的补充特征，精确复现应使用冻结源码。
`predict_bundle.py` 在新进程中加载冻结代码，并检查源码、模型、固定 HPO 及依赖版本。
版本为 Python 3.10.12、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2、
threadpoolctl 3.7.0；可使用本地状态文件记录的现有 inference 环境。

从 `LongVT/` 执行，数据根目录替换为本地官方数据解压位置，输出必须是新文件：

```bash
PATIENTPHEX_DATA=/path/to/PatientPheX/official_v1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python reproduction/patientphex_532512/predict_bundle.py \
  --model-dir reproduction/patientphex_532512/runs/optimization_v1/full \
  --input "$PATIENTPHEX_DATA/PatientPheX-V1-B/PatientPheX-V1-B.jsonl" \
  --hpo "$PATIENTPHEX_DATA/PatientPheX-V1-A/hp.obo" \
  --output reproduction/patientphex_532512/artifacts/optimization_v1_b.replay.jsonl \
  --evidence reproduction/patientphex_532512/artifacts/optimization_v1_b.replay.evidence.jsonl
```

如需重做当时的优化实验，运行已冻结的旧版脚本，使用新目录：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python \
  reproduction/patientphex_532512/runs/optimization_v1/source/run_optimization.py \
  --baseline-run reproduction/patientphex_532512/runs/baselines_v1 \
  --output-dir reproduction/patientphex_532512/runs/optimization_v1_retrain
```

真实 A/B 在独立进程重载后的预测和证据均逐字节一致，记录见
`runs/optimization_v1/verification.json`；A 预测 SHA256 为
`050936cb49a830b92c5da048c3fddec3aa01969de2ff66e9a53e822020e71a25`。
原始 train/A/B/HPO 哈希均未改变。

交付时的当前脚本还增加了自动导出 bundle 清单，避免未来训练后需要手工封装。
58 项 CPU 测试通过，其中包含缩写作用域、Unicode 偏移、标签污染拒绝、
分组划分、完整合成优化流程、独立进程重载及冻结源码被改动时拒绝推理。
本轮改动文件通过仓库 pre-commit 的 Ruff 和 Ruff format hooks。
未运行整个 LongVT 仓库测试或 GPU 测试；交付检查记录见 `DELIVERY_CHECKS.local.json`。

已经取得官方线上成绩，但尚未取得官方评分源码。本地评分器仍标记为
`local_reference_only_not_official_scorer`，不能声称其全部边界定义已与官方一致。
