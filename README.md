# Deep Learning HW1


## 环境依赖

- Python `3.10+`
- 依赖见 requirements.txt

## 代码结构

```text
.
├── EuroSAT_RGB/
├── experiments/
├── search_experiments/
├── src/eurosat_landcover_classifier/
│   ├── analysis/        # 权重可视化、错例分析、类别对错分率分析
│   ├── cli/             # train / search / evaluate / analyze 入口
│   ├── data/            # 数据加载、分层划分、标准化
│   ├── evaluation/      # 测试评估、混淆矩阵、训练曲线
│   ├── models/          # 自动微分张量与 MLP
│   ├── search/          # 随机搜索与网格搜索
│   ├── training/        # 训练循环、SGD、学习率衰减
│   └── utils/           # checkpoint、JSON、随机种子等工具
├── README.md
├── report.md
└── requirements.txt
```

## 项目要求对应实现

- 数据加载与预处理：`src/eurosat_landcover_classifier/data/`
- 模型定义与反向传播：`src/eurosat_landcover_classifier/models/`
- 训练循环：`src/eurosat_landcover_classifier/training/`
- 测试评估：`src/eurosat_landcover_classifier/evaluation/`
- 超参数查找：`src/eurosat_landcover_classifier/search/`
- 结果分析：`src/eurosat_landcover_classifier/analysis/`

当前模型支持：

- 自定义 `hidden_dim`
- `relu / sigmoid / tanh` 激活切换
- 交叉熵损失
- SGD
- Step learning rate decay
- L2 正则化
- gradient clipping
- 按验证集准确率保存 `best_model.npz`

## 数据集约定

默认数据目录为 `EuroSAT_RGB/`，目录结构应为按类别分文件夹存放图像。代码默认使用分层划分：

- 训练集：`70%`
- 验证集：`15%`
- 测试集：`15%`

训练集统计均值和标准差，验证集和测试集复用该统计量。

## 正式训练

当前两阶段搜索得到的最佳参数为：

- `learning rate = 0.0125`
- `hidden_dim = 512`
- `weight_decay = 5e-4`
- `activation = relu`

正式训练命令：

```bash


PYTHONPATH=src python -m eurosat_landcover_classifier.cli.train \
  --data-dir EuroSAT_RGB \
  --output-dir experiments \
  --run-name final_best_model \
  --epochs 40 \
  --batch-size 64 \
  --hidden-dim 512 \
  --activation relu \
  --lr 0.0125 \
  --weight-decay 5e-4 \
  --lr-decay 0.5 \
  --decay-every 5 \
  --max-grad-norm 5.0
```

训练产物位于 `experiments/final_best_model/`，其中：

- `best_model.npz`：验证集最优权重
- `last_model.npz`：最后一轮权重
- `history.json`：训练过程日志
- `curves.png`：训练/验证曲线
- `summary.json`：训练配置和最终结果摘要

## 超参数搜索

使用两阶段搜索。

第一阶段：随机搜索

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.search \
  --data-dir EuroSAT_RGB \
  --output-dir search_experiments/random_stage1_rerun \
  --mode random \
  --num-trials 12 \
  --epochs 15 \
  --batch-size 64 \
  --learning-rates 0.01,0.005,0.002,0.001 \
  --hidden-dims 128,256,512 \
  --weight-decays 0.0,1e-4,5e-4,1e-3 \
  --activations relu,tanh \
  --lr-decay 0.5 \
  --decay-every 5 \
  --max-grad-norm 5.0 \
```

第二阶段：网格搜索

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.search \
  --data-dir EuroSAT_RGB \
  --output-dir search_experiments/grid_stage2_rerun \
  --mode grid \
  --epochs 20 \
  --batch-size 64 \
  --learning-rates 0.0125,0.01,0.0075 \
  --hidden-dims 256,384,512 \
  --weight-decays 0.0,1e-4,5e-4 \
  --activations relu \
  --lr-decay 0.5 \
  --decay-every 5 \
  --max-grad-norm 5.0 \
```

搜索结果会写入各自目录下的 `search_results.json`。

## 测试评估

使用验证集最优权重在独立测试集上评估：

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.evaluate \
  --checkpoint experiments/final_best_model/best_model.npz \
  --data-dir EuroSAT_RGB \
  --output-dir experiments/final_best_model/evaluation
```

该命令会：

- 在终端打印测试集准确率
- 打印混淆矩阵
- 保存 `experiments/final_best_model/evaluation/confusion_matrix.png`

## 权重可视化与错例分析

当前分析脚本会把结果分到两个子目录：

- `analysis/weights/`：第一层权重可视化
- `analysis/errors/`：错例图、混淆矩阵、类别对错分率摘要

推荐分析命令：

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.analyze \
  --checkpoint experiments/final_best_model/best_model.npz \
  --data-dir EuroSAT_RGB \
  --output-dir experiments/final_best_model/analysis \
  --top-units-per-class 16 \
  --top-error-pairs 5 \
  --max-errors-per-pair 12
```

该命令会生成：

- `analysis/weights/first_layer_weights.png`
- `analysis/weights/<class_name>_first_layer_weights.png`：默认对全部 10 个类别分别生成
- `analysis/errors/confusion_matrix.png`
- `analysis/errors/misclassified_examples.png`
- `analysis/errors/pairwise_misclassification_summary.json`
- 若干类别对错例图，例如 `analysis/errors/highway_vs_river_misclassified_examples.png`

如果只想限制到部分权重类别，例如 `River` 和 `Forest`：

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.analyze \
  --checkpoint experiments/final_best_model/best_model.npz \
  --data-dir EuroSAT_RGB \
  --output-dir experiments/final_best_model/analysis \
  --weight-classes River,Forest \
  --top-units-per-class 16 \
  --top-error-pairs 5 \
  --max-errors-per-pair 12
```

如果只想分析指定类别对，例如 `Highway` 与 `River`：

```bash
PYTHONPATH=src python -m eurosat_landcover_classifier.cli.analyze \
  --checkpoint experiments/final_best_model/best_model.npz \
  --data-dir EuroSAT_RGB \
  --output-dir experiments/final_best_model/analysis \
  --weight-classes River,Forest \
  --error-pairs "Highway,River" \
  --top-units-per-class 16 \
  --max-errors-per-pair 12
```
