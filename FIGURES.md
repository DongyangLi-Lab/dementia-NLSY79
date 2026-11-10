# 从 result 一键重建图表

```powershell
python -m pip install -r requirements-figures.txt
python run_figures.py --result packages/result.zip --output outputs/result_figures
```

也可将 `--result` 指向解压的 `result/` 或它的父目录。所有数据入口都在这个目录；不读取旧 `results/tables`、原始 HRS 文件、`model.zip`、训练环境或 UNC 目录。不要使用旧 `make_figures.py` 重建本次图稿。

先检查输入：`python run_figures.py --result packages/result.zip --check-only --output outputs/figure_input_check`。
只画某张：加 `--only fig4`。可选阶段为 `tables,fig1,fig2,fig3,fig4,fig5,supplement`。
默认完整运行；`--dpi 600` 导出高分辨率版本；`--overwrite` 将既有输出改名为带时间戳的 previous 目录，然后重建。默认 PNG 为300 dpi，PDF/SVG保留矢量内容。

## 自动运行范围

- 五张当前主图：观察路径、访谈覆盖联合分布、年龄/BMI与选定v3箱线小提琴布局、八模型重要性、概率可靠性。
- 三张辅助图：四窗口死亡排除流程、八模型SHAP/置换排名、TabICL正式500人×60项贡献分布。
- 全部23张保存的编号表：精确CSV、HTML与分页PDF。宽表分栏、重复标识列；不重新取整源表结果。
- 已有runtime估算表及英文状态S1按字节保留，runtime作者代码不恢复。
- 中英主图图注、逐图来源数据、输入/输出哈希、队列核验、绘图对齐和PDF几何检查、浏览入口 `index.html`。

Fig3G保留选定的v3**布局**，用新队列重绘数据。Fig4采用已完成验收的全部八模型，不再沿用旧版七模型排除。Fig5全部面板使用本轮修订结果；仅C/D保留图形CI，默认2,000次共享家庭聚类bootstrap，蓝/朱红透明度0.24。Bootstrap只重采样保存的预测，不训练模型。

## 明确不冒充重绘的历史材料

Figure P、Figure M、core combined_story 依赖旧队列投影或旧模型响应探针，当前result里只有其成品而非可用于本次修订的全部计算来源；不将旧图复制进新版输出。原论文 S2–S9 与后来选定主图的编号体系也不同，未映射的图不会静默编造或改号。这些条目写入 `run_report.json`，辅助图使用描述性名称。

`complete` 表示上面明确列出的入口范围成功，不表示全文/补充材料/审稿回复已经改完。PDF自动碰撞检测如发现FAIL，入口返回非零并保留结果供修正；WARN仍须人工检查。

## 结果包契约

入口要求 `tables/audited_sources/`、`cohorts/`、`predictions/`、`importance/`、`figure_inputs/`。旧版result缺少 `figure_inputs` 时明确报错。打包脚本 `tools/package_result_model.py result` 自动从已保存的选择文件导出32个轻量文件；不读取pickle。

绘图代码在 `plotting/`，由此前获选Python图稿做结构适配，旧源码及成品保留为历史。复制的通用审计工具来自项目使用的nature-figure技能。科学模型输出不被修改。
