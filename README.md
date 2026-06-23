# Hand Keypoint Short-Term Motion Prediction with LSTM

> 基于 LSTM 的手部关键点序列短时运动预测——《人工智能》课程论文配套代码仓库

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

本仓库为《人工智能》课程论文《基于 LSTM 的手部关键点序列短时运动预测——一种基于自建视频数据集的算法复现与分析》的配套代码与实验数据。

研究任务：给定过去 5 帧手部 21 关键点坐标，使用 LSTM 预测下 1 帧的关键点位置。

### 核心发现

- **聚合层面**：LSTM 整体 MPJPE（27.48 像素）高于朴素基线（15.86 像素）
- **分层后**：LSTM 在高运动帧上反超朴素基线约 7%（39.02 vs 41.96 像素）
- **逐关节**：手腕等锚点关节误差仅 2.4 像素，指尖等末端关节误差约 30 像素

## 项目结构

```
.
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── LICENSE                            # MIT 许可证
├── .gitignore                         # Git 忽略规则
│
├── data/                              # 自建数据
│   ├── videos/                        # 5 段原始 mp4 视频（git lfs 管理）
│   │   ├── seg1_open_close.mp4
│   │   ├── seg2_wave.mp4
│   │   ├── seg3_finger_bend.mp4
│   │   ├── seg4_point.mp4
│   │   └── seg5_switch.mp4
│   └── keypoints/                     # 提取后的关键点 csv
│       └── all_keypoints.csv
│
├── src/                               # 源代码
│   ├── extract_keypoints.py           # 用 YOLO11n-pose 提取关键点
│   ├── dataset.py                     # PyTorch Dataset 类
│   ├── model.py                       # LSTM 模型定义
│   ├── train.py                       # 训练脚本
│   ├── evaluate.py                    # 评估脚本（MPJPE / PCK）
│   ├── stratified_analysis.py         # 分层误差分析
│   └── visualize.py                   # 结果可视化
│
├── configs/                           # 配置文件
│   ├── within_segment.yaml            # 段内划分配置
│   └── cross_segment.yaml             # 跨段划分配置
│
├── results/                           # 实验结果
│   ├── logs/                          # 训练日志 csv
│   ├── predictions/                   # 测试集预测结果
│   └── figures/                       # 可视化图片
│
└── paper/                             # 论文文档
    └── course_paper.pdf
```

## 环境配置

### 硬件要求

- GPU：NVIDIA GTX 1660 及以上（推荐 RTX 4060，本项目实测显存占用 < 4 GB）
- 训练时间：单次完整训练约 30 分钟

### 软件依赖

- Python 3.10
- CUDA 11.8 或 12.x
- PyTorch 2.0+

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/hand-keypoint-lstm.git
cd hand-keypoint-lstm

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate          # Linux/Mac
# 或 Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载 YOLO11n-pose 预训练权重（首次运行 extract_keypoints.py 会自动下载）
```

## 数据准备

本仓库已提供提取好的关键点 csv 文件（`data/keypoints/all_keypoints.csv`），可直接用于第二阶段 LSTM 训练。

若需从原始视频重新提取关键点：

```bash
python src/extract_keypoints.py \
    --video_dir data/videos/ \
    --output data/keypoints/all_keypoints.csv \
    --model yolo11n-pose.pt
```

## 运行步骤

### 1. 训练 LSTM（段内划分）

```bash
python src/train.py --config configs/within_segment.yaml
```

训练完成后输出：
- `results/logs/lstm_within_baseline_log.csv`：训练日志
- `results/predictions/lstm_within_baseline_test_predictions.csv`：测试集预测
- 控制台打印最终 MPJPE 和 PCK

### 2. 训练 LSTM（跨段划分）

```bash
python src/train.py --config configs/cross_segment.yaml
```

### 3. 分层误差分析

```bash
python src/stratified_analysis.py \
    --predictions results/predictions/lstm_within_baseline_test_predictions.csv \
    --output results/figures/stratified_mpjpe.png
```

### 4. 可视化结果

```bash
python src/visualize.py --all
```

生成所有论文图表至 `results/figures/`。

## 实验结果

### 段内划分

| 方法 | MPJPE (px) | PCK@0.05 |
|---|---:|---:|
| 朴素基线 | **15.86** | **0.9115** |
| LSTM baseline | 27.48 | 0.8180 |
| LSTM + velocity | 27.73 | 0.8194 |

### 跨段划分

| 方法 | MPJPE (px) | PCK@0.05 |
|---|---:|---:|
| 朴素基线 | **18.75** | **0.8760** |
| LSTM baseline | 64.91 | 0.4828 |
| LSTM + velocity | 68.22 | 0.4704 |

### 分层误差分析

| 帧档位 | 朴素基线 MPJPE | LSTM MPJPE | 优势方 |
|---|---:|---:|:---:|
| 低运动 (<4.5 px) | 2.34 | 16.21 | 朴素 |
| 中运动 (4.5–14.3 px) | 8.93 | 26.88 | 朴素 |
| **高运动 (≥14.3 px)** | 41.96 | **39.02** | **LSTM (+7%)** |

## 关键代码模块说明

### model.py

```python
class HandLSTM(nn.Module):
    """两层 LSTM + 全连接输出"""
    def __init__(self, input_dim=42, hidden_dim=128, num_layers=2, output_dim=42):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, 5, 42)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

### dataset.py

时序窗口构造：每段视频独立切分滑动窗口，避免段间错配。归一化采用"手腕原点 + 边界框对角线尺度"方案。

### stratified_analysis.py

按"该帧真实运动幅度"分位数自动确定阈值（33% / 66%），划分低中高三档，分别统计 MPJPE。

## 课程论文信息

- 课程：《人工智能》
- 选题方向：（一）复现任意人工智能算法
- 复现算法：LSTM（Hochreiter & Schmidhuber 1997）
- 自建数据集：5 段共 2199 帧手部动作视频
- 论文 PDF：见 `paper/course_paper.pdf`

## 引用与致谢

本项目参考了以下工作：

1. Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Computation*, 1997, 9(8): 1735-1780.
2. Martinez J, Black M J, Romero J. On human motion prediction using recurrent neural networks. *CVPR*, 2017.
3. Ultralytics YOLO11 documentation: https://docs.ultralytics.com/

论文写作过程中使用了 Claude（Anthropic）辅助进行结构梳理、文字润色与公式排版建议，最终实验与结果分析由本人完成。

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 联系方式

如有问题，请通过 GitHub Issues 提出。
