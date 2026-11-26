# MCL-ASF 公开代码说明

本仓库提供论文 MCL-ASF 的主要模型、损失函数、光谱扰动增强、训练和推理接口。**不提供私有高光谱数据集和预训练权重。**

核心实现基于原有论文对齐版本整理，空间分支采用 1 × 1 自适应平均池化，默认模型包含 544,773 个可训练参数（约 0.545 M）。此处调整了旧压缩包的空间池化和全连接层，旧包权重不兼容；尚未核实该结构与论文实测训练权重完全一致。

- `mcl_asf/model.py`：MSCR 光谱分支、空间分支、样本门控融合、ASF 波段注意力、五分类头。
- `mcl_asf/losses.py`：跨条件 MCL、特征一致性、ASF 权重 KL 一致性及总损失。
- `mcl_asf/augmentations.py`：高斯噪声、光谱轴平移、波段 Dropout、独立双视图。
- `mcl_asf/data.py`：通用 CSV 清单形式的数据接口，支持私有 `.npy` patch。
- `scripts/train.py`：根据配置文件训练，不读取户外测试数据。
- `scripts/infer_patches.py`：预测 Patch 的 Softmax 概率。
- `scripts/infer_cube.py`：可选整图推理，输出未经分位数裁剪的四类概率图。
- `scripts/smoke_test.py`、`tests/test_core.py`：无数据自检。

## 快速运行

在仓库根目录执行：

```bash
python -m pip install -e .
python -m scripts.smoke_test
python -m unittest discover -s tests -v
```

训练时自行建立 CSV 文件，参考 `examples/train_manifest.example.csv`。其 `patch_path` 可以是绝对路径，也可以相对于 CSV 所在目录。标签 0～4 对应钢珠、钨珠、钨块、石子、背景；三种采集条件明确区分室内单目标45 klux、室内单目标65 klux、室内三混65 klux。根据采集协议调整 `configs/train.example.json`。训练命令：

```bash
python -m scripts.train --manifest /path/to/train.csv --config configs/train.example.json --preflight-only
python -m scripts.train --manifest /path/to/train.csv --config configs/train.example.json --out outputs/train --seeds 0-9
```

**注意**：配置采用修订稿参数：噪声标准差 0.01、最大光谱偏移 1 个波段、波段丢弃概率 0.05；公开代码不等于完整论文数据复现。早期 `mcl_asf_sc_v2_best.pth` 与本次论文对齐版的模型层级不同，无法直接载入。

仓库没有添加开源许可证，需要仓库所有者先确认许可方式。源数据、预训练权重、作者邮箱和私人目录均不在公开包中。

完整说明见英文 `README.md` 和 `docs/RELEASE_NOTES.md`。
