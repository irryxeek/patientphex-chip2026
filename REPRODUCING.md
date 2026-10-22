# 复现说明

从本仓库根目录运行以下命令。运行目录独占创建，已有完整产物可按对应脚本的规则重用，
不完整运行不会自动恢复优化器状态。请为新实验使用独立目录。

## 1. 环境与数据

按根目录 README 安装固定依赖。GPU 训练前确认 PyTorch 能访问你获授权使用的设备；
在同一软件环境中完成拟合、保存和重载，模型包会检查依赖版本与文件哈希。

通过比赛官方入口自行取得训练、A、B 数据及 HPO v2026-06-23。原实验使用 80 篇训练文献，
其中 64 篇用于内层训练和选择，16 篇用于开发验证；A/B 不用于模型选择或训练。
源文件名和原实验 SHA256 见 [experiment_config.json](results/experiment_config.json)。

```bash
export PATIENTPHEX_TRAIN=/path/to/PatientPheX-train.jsonl
export PATIENTPHEX_A=/path/to/PatientPheX-A.jsonl
export PATIENTPHEX_B=/path/to/PatientPheX-V1-B.jsonl
export PATIENTPHEX_HPO=/path/to/hp.obo
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=disabled
```

项目不附带官方数据、人工表型标签、训练得到的权重或预测文本。
测试数据为人工构造的软件测试夹具。

## 2. 下载固定预训练模型

此步骤需要网络，会下载约两个 BERT 编码器的权重。资源受各自原始条款约束。
下方代码使用清单中的固定 revision，逐文件核验 SHA256；默认下载到被 Git 忽略的 `models/`。

```bash
python - <<'PY'
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

for name in ("biolinkbert_base", "sapbert"):
    manifest_path = Path("resources") / f"{name}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    directory = Path("models") / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, info in manifest["files"].items():
        path = Path(hf_hub_download(
            repo_id=manifest["repo"], filename=filename,
            revision=manifest["revision"], local_dir=directory,
        ))
        with path.open("rb") as handle:
            actual = hashlib.sha256()
            for block in iter(lambda: handle.read(1048576), b""):
                actual.update(block)
        if actual.hexdigest() != info["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}/{filename}")
    shutil.copyfile(manifest_path, Path("models") / manifest_path.name)
PY
```

## 3. 按历史采用顺序构建前置模型

下列命令使用每个实际运行阶段的源码版本，不使用工作源码中的未采用尝试。
命令会重新拟合模型并生成本地结果，不发送线上提交。

```bash
python reproducibility/baselines_v1/source/run_baselines.py \
  --train "$PATIENTPHEX_TRAIN" --hpo "$PATIENTPHEX_HPO" \
  --a "$PATIENTPHEX_A" --b "$PATIENTPHEX_B" \
  --output-dir runs/baselines_v1

python reproducibility/optimization_v1/source/run_optimization.py \
  --baseline-run runs/baselines_v1 --output-dir runs/optimization_v1

python reproducibility/recall_optimization_v1/source/run_recall_optimization.py \
  --previous-run runs/optimization_v1 --output-dir runs/recall_optimization_v1
```

## 4. 神经训练与导出

将 `CUDA_VISIBLE_DEVICES` 改成获授权且显存足够的 GPU 编号或 UUID。
原实验的两卡加速只并行独立折；下方按相同种子与设置顺序执行各折。

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python reproducibility/neural_frontier_v1/source/run_neural_frontier.py \
  --previous-run runs/recall_optimization_v1 \
  --output-dir runs/neural_frontier_v1 \
  --model-root models --artifact-root artifacts/neural_frontier_v1 \
  --device cuda:0 --stage all
```

脚本先在四个内折选择实体配置及关系融合，再比较开发集。
只有总分严格提高且患者 Macro 不下降，才继续全量训练并导出 A/B。
原实验选中实体 `replace / similarity=0.90 / margin=0.03`，
神经关系权重 0.25、关联阈值 0.35；不应依据 B 集反馈再次调参。
各阶段日志、模型元数据和生成的 `summary.json` 是本次实际运行结果，不能以预存报告代替。

## 5. 独立重载推理

全量模型成功生成后，在相同依赖环境中使用新的输出路径：

```bash
python runs/neural_frontier_v1/source/predict_neural_bundle.py \
  --model-dir runs/neural_frontier_v1/full \
  --input "$PATIENTPHEX_B" --hpo "$PATIENTPHEX_HPO" \
  --output runs/neural_frontier_v1/b.replay.predictions.jsonl \
  --evidence runs/neural_frontier_v1/b.replay.evidence.jsonl \
  --device cuda:0
```

模型包包含绝对运行路径及文件哈希，生成后请保留其目录与依赖关系。
仓库提供源码与配置导出，不把原服务器模型包伪装成可任意搬动的二进制发行包。
原产物的重载结果见 [verification.json](results/verification.json)。

## 导出差异与统计限制

`SOURCE_MANIFEST.json` 记录了实际运行源码的原哈希和本仓库副本哈希。
神经入口的 `--model-root` 默认值改为相对目录 `models`；命令示例移除了服务器路径与设备 UUID。
算法源码其余部分保持原样，工作源码和历史快照分开放置。

本地参考评分器的边界细节尚未与组织方源码逐项核对。
开发集曾被多轮查看；四名空表型患者来自同一篇文献，分组收益不能直接外推。
官方总分为一次已完成的赛后 B 集评测；原始回执和预测保留在服务器，不随本仓库发布。
