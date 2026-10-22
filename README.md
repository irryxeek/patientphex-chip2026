# PatientPheX · CHIP2026 评测一

面向生物医学全文的表型概念识别与指定患者表型关联系统，对应
[CHIP2026 评测一](http://cips-chip.org.cn/2026/eval1) /
[天池 532512](https://tianchi.aliyun.com/competition/entrance/532512)。

采用 **BioLinkBERT 表型跨度识别 + SapBERT HPO 检索 + 患者证据关系融合**。
本项目独立运行，不需要安装 LongVT 或 verl。

## 已验证结果

2026-10-22 官方 **B 集赛后评测**总分为 **0.6567391667419546（65.67）**，
比上一版提高 **3.09 个百分点**。这属于赛后评测，不代表正式赛期排名或获奖资格。

| 官方指标（%） | 上一版 | 本版 |
| --- | ---: | ---: |
| Mention F1 | 68.18 | **69.36** |
| Document F1 | 75.66 | **78.50** |
| Patient Micro F1 | 56.91 | **61.23** |
| Patient Macro F1 | 49.59 | **53.61** |
| 四项平均 | 62.59 | **65.67** |

精确数值见 [官方指标](results/official_metrics.json)。本轮只提交一个新文件；
新版排名未核实。公开报告不包含登录会话或页面截图。

本地固定 16 篇开发文献、54 名患者的参考总分为 **59.91 → 67.21**。
开发集曾被多轮实验查看，本地分数和置信区间不能代替独立测试结果。
详细消融、错误分析及限制见 [实验报告](reproduction/patientphex_532512/NEURAL_FRONTIER_REPORT.md)。

## 快速开始

已验证环境为 Linux、Python 3.10.12；神经实验使用 RTX 4090。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
pre-commit run --all-files
python reproduction/patientphex_532512/patientphex.py --help
```

原始训练标注、A/B 数据、HPO 文件、模型权重和预测文本不包含在仓库中。
请通过比赛官方入口取得数据，并阅读 [完整复现步骤](REPRODUCING.md)。
本仓库提供固定预训练模型 revision 和文件哈希；不自动登录或提交天池。

## 目录

| 路径 | 内容 |
| --- | --- |
| `reproduction/patientphex_532512/` | 工作源码、CPU 回归测试、实验协议与历次报告 |
| `reproducibility/` | 基线、优化、召回与最终神经实验实际采用的源码快照 |
| `resources/` | 公开预训练模型的固定 revision 与下载文件 SHA256 |
| `results/` | 脱敏后的官方指标、开发集指标、配置和验证摘要 |
| `SOURCE_MANIFEST.json` | 导出来源及导出前后文件哈希 |

复现实验时使用 `reproducibility/` 中的对应版本：工作目录包含曾尝试但未采用的后续修改，
尤其是 `enhanced.py`，不能直接用它替换取得所报告成绩的版本。
导出副本将服务器专用路径改为可配置路径；模型算法和原服务器实验产物保留。

原实验完成 12 个神经模型、17,700 个优化步骤，92 项 CPU 测试通过。
四折预测以及最终 A/B 预测和证据文件均通过独立进程复算；
逐字节一致性只在记录的软件环境与 RTX 4090 上验证。

方法来源见 [前沿调研](reproduction/patientphex_532512/FRONTIER_RESEARCH_20261022.md)，
资源归属与使用说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
