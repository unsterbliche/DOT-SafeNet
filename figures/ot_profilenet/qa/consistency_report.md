# OT-ProfileNet图表数据一致性检查

## 数据来源

- 论文正文和补充材料提取文本：`original_data/manuscript_extracted.txt`、`original_data/supplementary_materials_extracted.txt`。
- A100旧PreMOTA回归预测：28个pK/pAC50观测值与预测值数组，共45,527条记录。
- A100模型比较结果：`affinity_model_compare.xlsx`和`model_compare_target.xlsx`。
- 预训练记录：`pretraining_loss.csv`，包含第1至100轮训练和验证损失。
- 公开代码：预训练和微调的训练脚本、模型定义、数据处理notebook。

图中散点未进行像素反推，未生成模拟观测值。

## Figure 2与Supplementary Figures 2–4

Figure 2a–b的数据来自Supplementary Table 9。七类靶点数量之和为194。Figure 2c、Supplementary Figure 4分别使用A100数组中的pAC50和pK。Figure 2d使用A100比较工作簿。Figure 2e–f使用7个未纳入训练的hERG化合物预测表。

Supplementary Figure 3a使用100轮预训练损失；b–d使用A100分类比较工作簿。Supplementary Figure 2a可由7,258条蛋白序列直接计算。7,063条序列长度不超过2,000个氨基酸，占97.31%。Supplementary Figure 2c–d的阳性、阴性和总配对数满足：1,105,014 + 997,753 = 2,102,767。

Supplementary Figure 2b已由V100归档的原始预训练配对表重建。原始表含2,102,767条配对；按UniProt靶点计数后得到7,258个靶点，与Supplementary Figure 2a、2c和2d的统计一致。

## Supplementary Table 1

补充材料中的35行、140个评价数值与A100比较工作簿完全一致。

对A100逐条预测数组重新计算每个终点的RMSE、PCC、Spearman和R²，并在pK与pAC50之间取均值。27个fine-tuned指标与表中值的绝对差不超过0.002。NR的RMSE存在一处差异：

| 数据来源 | NR RMSE |
|---|---:|
| 补充材料 | 0.6340 |
| A100比较工作簿 | 0.6340 |
| A100逐条数组重算 | 0.6237 |

现有数值重绘使用0.6237。若Figure 2d继续使用逐条数组重算值，Supplementary Table 1中的NR RMSE应改为0.624，以保持图表一致。若保留原发表格0.634，图注和数据文件需注明该值来自原比较工作簿。

## Supplementary Table 12

| 参数 | 补充材料 | 代码或训练记录 | 结果 |
|---|---:|---:|---|
| epochs | 70 | 代码为100；loss.csv包含1–100轮 | 不一致 |
| learning rate | 0.0001 | 0.0001 | 一致 |
| dropout | 0.2 | 0.2 | 一致 |
| molecule embedding dimension | 256 | 256 | 一致 |
| protein embedding dimension | 256 | LSTM hidden size为512 | 不一致 |
| hidden dimension | 512 | 512 | 一致 |
| attention heads | 2 | 2 | 一致 |

预训练loss.csv的最低验证损失出现在第77轮。正文“converged after 70 epochs”不能由当前训练记录复现。训练脚本以验证集AUROC选择最佳checkpoint，checkpoint保存的具体轮次尚未进入本地归档。

蛋白输入来自1,280维ESM2残基表示，LSTM hidden size为512。代码中未发现256维蛋白表示。Supplementary Table 12中的“protein embedding dimension”若表示LSTM输出，应改为512；若表示ESM2输入，应改为1,280，并修改参数名称。

## Supplementary Table 13

epochs、learning rate、early stop、LSTM层数、分子表示维数、hidden dimension和attention heads均与微调代码一致。protein embedding dimension仍存在256与512的差异。七个靶点家族的微调loss.csv和checkpoint轮次未进入本地归档，实际停止轮次尚未核对。

## 当前完成情况

- Figure 2：六个panel的数据来源已整理。
- Supplementary Figure 2：a–d已整理；b由V100原始预训练配对表重建。
- Supplementary Figures 3–4：数据和原始notebook已整理。
- Supplementary Tables 1、12、13：已生成逐项审计CSV。
- 数值、校验和、PNG解码和PDF文件检查均通过。

