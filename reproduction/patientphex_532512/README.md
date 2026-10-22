# CHIP2026 评测一 / PatientPheX / 天池 532512

> 本页保留实验工程说明。首次使用请阅读根目录 [README](../../README.md) 和
> [REPRODUCING](../../REPRODUCING.md)。文中 `runs/`、`submissions/`、`*.local.json`
> 指服务器保留的实验产物，不随源码上传。

当前状态：**神经改进、A/B 独立复算和一次天池赛后评测均已完成。**
2026-10-22 16:41（北京时间）核实的官方赛后总分为 **0.6567392（65.67）**，
比前版 0.6258642 提高 **3.0875 个百分点**，四项 F1 均提高。
同一 64/16 篇开发划分总分从 59.91 提高至 **67.21**，患者 Macro 从 38.55 提高至 **54.10**。
当前选用 `runs/neural_frontier_v1/full/`：BioLinkBERT 跨度识别、SapBERT HPO 检索及患者证据概率融合。
12 个模型共完成 17,700 个更新步骤；A/B 预测和证据的新进程复算逐字节一致。
当前汇总排名仍对应旧分数，未报告新版排名。
任务方案见 [TASK_PLAN.md](TASK_PLAN.md)，最新结果、回执与复现命令见
[NEURAL_FRONTIER_REPORT.md](NEURAL_FRONTIER_REPORT.md)；前轮 CPU 结果见
[RECALL_OPTIMIZATION_REPORT.md](RECALL_OPTIMIZATION_REPORT.md)，未选中方案见
[SET_OPTIMIZATION_REPORT.md](SET_OPTIMIZATION_REPORT.md)；前轮历史结果见
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) 和 [BASELINE_REPORT.md](BASELINE_REPORT.md)。

2026-10-22 前沿调研见 [FRONTIER_RESEARCH_20261022.md](FRONTIER_RESEARCH_20261022.md)：
包括组织方新发布的前三名方案、近期 HPO 抽取方法及下一轮实验优先级。
后续神经方案已完成四折选择和外层验证，本地总分提高至 **67.21**，患者 Macro 提高至 **54.10**。
获用户授权后使用 GPU 6/7，已完成全量训练、A/B 生成与官方赛后评测。
方案见 [FRONTIER_EXECUTION_PROTOCOL.md](FRONTIER_EXECUTION_PROTOCOL.md)，
实际执行状态见 [NEURAL_FRONTIER_REPORT.md](NEURAL_FRONTIER_REPORT.md)。

任务以 [CHIP2026 官网评测一](http://cips-chip.org.cn/2026/eval1) 为准，对应
[天池 532512](https://tianchi.aliyun.com/competition/entrance/532512)；
[组织方仓库](https://github.com/DUTIR-BioNLP/PatientPheX) 提供配套说明。
2026-10-20 核对时，官网公布的正式 B 榜已于 10 月 1 日截止。
**天池实时接口另有“赛后评测”（seasonId 1862）：2026-10-03 至 2029-10-03。**
这不等于恢复正式赛期或获奖资格；本次已核实登录后的下载、提交和成绩查询权限。
最初的匿名下载接口要求登录；本次已通过用户登录后的 Chrome 下载 A、B 两个原始数据包，未导出登录凭证。

## 已下载的正式数据

| 集合 | 文件 | 文献数 | 患者数 |
| --- | --- | ---: | ---: |
| 训练集 | `PatientPheX-V1-A/PatientPheX-train.jsonl` | 80 | 209 |
| A 榜 | `PatientPheX-V1-A/PatientPheX-A.jsonl` | 20 | 53 |
| B 榜 | `PatientPheX-V1-B/PatientPheX-V1-B.jsonl` | 100 | 244 |

原始 ZIP 已保留并通过 CRC 检查，大小与官网下载列表一致；本地清单记录各文件 SHA256。
三份数据的 PMC ID 无交集，所有输入 passage 与患者提及的位置检查通过。
随包 HPO 与此前固定的 v2026-06-23 本体 SHA256 完全一致。
本地具体路径记录在 `STATUS.local.json`；下载清单和源文件审计记录见 `DATA_AUDIT.local.json`。

训练集包含重复患者关联、分号形式关联、辅助 `D` 标记及部分超出当前严格 HPO 校验范围的 ID。
已区分金标准读取与提交校验，并增加回归测试。
训练别名仅使用有效 ID、明确的本体替代关系及 `-1`；含未解决 ID 的实体不用于别名学习。
评估保留异常金标准 ID 和未知辅助标记，并报告复合关联口径的敏感性结果。
原始标注未被改写或删除；完整审计在 `runs/baselines_v1/gold.audit.json`。

## 已实现的行为

- 子任务一：逐个原始 BioC passage 提取表型，输出原文字符 offset/length、HPO ID 和否定标记。
- 子任务二：使用输入中给定的患者提及及局部上下文关联表型，输出每个指定患者的表型集合和独立证据记录。
- 固定 HPO v2026-06-23，仅使用 `HP:0000118` 分支的在用术语、名称与 EXACT 同义词。
- 可从官方训练集学习实体表述及复合 ID；不会使用推理输入的 entities/association 字段。
- 验证集按 PMID 与相同输入全文分组，只从官方训练集切分。推理默认拒绝与拟合数据重复的 PMC、PMID 或全文。
- 本地评分报告 mention F1、document F1、patient micro/macro F1 和四者均值。
- 根据 passage 元数据排除明确标记的参考文献、致谢、经费与作者贡献；其余非正文类别待官方评分器核验。
- 提交检查覆盖全文档/全患者完整性、重复记录、字符偏移、原文一致性、HPO 版本和概念分支。
- 输出文件采用独占创建，已存在的预测、模型和报告不会被自动覆盖。

当前方法在别名、词形、文内缩写、跨度合并及梯度提升分类器之上，增加神经跨度、语义检索和证据关系融合。
它不是组织方 PhenoTagger + Qwen3-8B 基线的复现；边界、概念消歧、隐含表型、表格及跨段共指仍有限制。
当前模型已完成开发验证和线上 B 集赛后评测。

## 运行

从本 GitHub 仓库根目录执行。下方原版 `patientphex.py` 只依赖 Python 3.10+ 标准库。
当前神经模型需要 PyTorch、Transformers、scikit-learn 等固定依赖，请按
[最新报告](NEURAL_FRONTIER_REPORT.md#交付与复现) 使用冻结的 `predict_neural_bundle.py` 与 CUDA 环境。
旧 CPU 模型仍可用 `predict_bundle.py` 复现；两者均不需要安装 verl。
正式数据已就绪；下方基础命令中的文件名是占位约定，实际以下载文件为准。
首轮完整比较和 A/B 生成的一条命令见 [执行报告](BASELINE_REPORT.md#复现)。
HPO 本体来自[官方固定版本发布](https://github.com/obophenotype/human-phenotype-ontology/releases/tag/v2026-06-23)。
本机会话下载位置和 SHA256 记录在 `sources/manifest.local.json`，没有使用额外人工标注表型数据。

```bash
PATIENTPHEX_DATA=/path/to/official/competition/data
PATIENTPHEX_HPO=/path/to/hp.v2026-06-23.obo

python reproduction/patientphex_532512/patientphex.py audit \
  --input "$PATIENTPHEX_DATA/train.jsonl"

# 仅从官方训练集划分 20% 的文献组作本地验证。
python reproduction/patientphex_532512/patientphex.py holdout \
  --train "$PATIENTPHEX_DATA/train.jsonl" --hpo "$PATIENTPHEX_HPO" \
  --output-dir reproduction/patientphex_532512/runs/holdout_v1

# 根据本地验证结果选定方法后，再用全部官方训练集构建模型。
python reproduction/patientphex_532512/patientphex.py build \
  --hpo "$PATIENTPHEX_HPO" --train "$PATIENTPHEX_DATA/train.jsonl" \
  --output reproduction/patientphex_532512/artifacts/train.model.json

python reproduction/patientphex_532512/patientphex.py predict \
  --model reproduction/patientphex_532512/artifacts/train.model.json \
  --input "$PATIENTPHEX_DATA/test_b.jsonl" \
  --output reproduction/patientphex_532512/artifacts/test_b.predictions.jsonl \
  --evidence reproduction/patientphex_532512/artifacts/test_b.evidence.jsonl

python reproduction/patientphex_532512/patientphex.py validate \
  --input "$PATIENTPHEX_DATA/test_b.jsonl" --hpo "$PATIENTPHEX_HPO" \
  --predictions reproduction/patientphex_532512/artifacts/test_b.predictions.jsonl
```

`holdout` 保留划分、模型、预测、证据和指标；再次运行请使用新目录。
`run_baselines.py` 比较两种方法、冻结代码与配置，并生成全量拟合后的 A/B 文件。
`analyze_baselines.py` 对同一批验证文献做配对 bootstrap，保留患者宏平均的患者权重。
`--allow-training-overlap` 仅用于明确的训练样本诊断，不能据此报告验证成绩。
本工具不会调用天池提交接口、收费 API 或线上实验跟踪。

## 评分边界

目前只有公开规则，没有取得官方评分代码，因此本地结果标记为 `local_reference_only_not_official_scorer`。
复合 ID 拆成独立单元；金标准否定实体不进入正例，预测为普通表型时会产生 FP。
金标准 `-1` 的提及按边界匹配；患者关联中的未映射表型使用原文字符串。
重复金标准项按集合计数，旧 ID/异常 ID 保留原值；只有 `NO` 作为否定标记。
患者关联主口径拆分 HPO 复合值，另存 `metrics.literal_associations.json` 检查原始字符串口径。

天池实时赛题明确不计算参考文献及其他非正文预测。
目前依据明确的 passage 元数据做保守过滤，保留 title/abstract；具体类别范围尚未取得官方代码确认。
公开说明没有完全规定无 ID 实体的文档级集合、全局空分母等边界。
当前金标准文档级集合保留除 `-1` 外的原始 ID，包括异常 ID；预测仍须满足有效 HPO 约束。
全局空分母取 0；单个患者两集合均空时 P/R/F1 取 1。
这些约定写入每份本地指标报告，拿到官方评分程序后需做一致性核验。
不能将本地参考分数直接称为天池成绩。

官网与组织方 README 的个别统计存在差异，例如训练集唯一 HPO ID 数及复合表型数量；
实际审计以下载的正式数据为准，不用说明页中的统计值修改数据。

## 下一步

当前神经模型已提交一次并获得更高的官方赛后成绩，2026-10-22 查询时剩余提交次数为 4。
后续重点是精确边界、HPO 消歧及已有候选却未正确分配的关系；保留本轮召回和精度收益。
新增超参数继续在 64 篇拟合部分内做分组验证；已查看的开发集和线上反馈不能作为未触碰测试集。
本任务已获物理 GPU 6/7 授权，后续启动前复核当前占用；公开模型仍须遵守不超过 10B 的限制。
外部资源须公开可获取并记录出处；禁止外部人工标注表型数据，也不能将 A 榜答案用于 B 榜训练。

当前已完成新版训练、GPU 推理和一次赛后提交；没有发送材料邮件。
合成 smoke 样例和测试仅用于软件检查，不能代替正式数据实验。

## 检查

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest reproduction/patientphex_532512/tests -q
ruff check reproduction/patientphex_532512/*.py reproduction/patientphex_532512/tests/
ruff format --check reproduction/patientphex_532512/*.py reproduction/patientphex_532512/tests/
```

2026-10-22 共 92 项 CPU 测试通过。当前状态保存在 `STATUS.local.json`，
选用模型与推理入口记录在 `CURRENT_MODEL.local.json`；旧版交付检查见
`DELIVERY_CHECKS.local.json`，新神经方案完整检查见 `runs/neural_frontier_v1/verification.json`。
