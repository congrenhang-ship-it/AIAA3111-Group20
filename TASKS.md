# Group 20 完成记录与剩余事项

更新：2026-11-19。技术实现、正式测试、代码整理、报告和海报结果整合已完成。

## 已完成

- [x] E01–E16 全部实现、开发验证、G2 冻结及 ULB/Handbook/PaySim 正式测试。原始方法、参数、切分、规则、审核预算及模型保留。
- [x] 主线与四个成员分支的交付已整合，成员分支保留。
- [x] 176 组预算、分组加总、固定阈值、15 个标签条件、150 次置换、9 个历史缺失条件及 17 张测试图核对完成。
- [x] 代码中的中文注释清理、必要说明及输出改成英语；所有命令参数、默认值和功能保留。
- [x] 重复验证/导出逻辑共用辅助函数。核心文件留在 code/；模块验证、规则复核及分类检查移入 code/validation/；下载、冻结及复现工具移入 code/workflows/。
- [x] 共享分类选择与规则诊断函数放在核心模块，算法不依赖验证脚本。AST 核对确认数值逻辑保留。
- [x] 精简及移动后 98 项公共检查＋11 项分类检查通过；完整 ULB 统一验证通过。三数据集 8 份正式汇总数值一致，48 份预测/案例 CSV、17 张测试图与原版本逐字节相同，原模型文件不变。
- [x] GitHub Linux 独立重建通过。补充随机森林 AP 和八项金额组分配的原有平台差异逐项保留，见 reproduction.json；其他核心对照通过。
- [x] 报告已替换旧验证阶段描述，整合正式结果、5 张报告尺寸矢量图、10 张表、结论和局限、引用及实际 AIGC 披露。正文 10 页。
- [x] report/Report.md 为可编辑正文；report/build_report.py 从正式 JSON 绘图并生成 Report.pdf，不触发训练或评分。正文页数和 15 张数值表已自动核对。
- [x] README、数据获取说明、CI 命令、复现证据及技术包同步新结构。

- [x] 对照课程要求与 L3–L7 原课件复核报告，补强指标、聚类和规则分析。独立重应用两套冻结 R001 规则，覆盖数与正式测试一致。

- [x] 五张报告图已重绘为矢量 PDF，核对了绘图数值及不确定性范围。海报使用三张报告图，已通过图源、排版及单页 PDF 检查。HTML 版可切换审核预算和特征组合、放大图表，交互数值已与正式测试汇总逐项核对。

- [x] 补充审核工作量表、审核反馈回放、有序与无序历史对照、GMM 漏检诊断及客户历史金额偏离对照。正式测试数值保留，新增 Handbook 结果单独列为探索性分析。正文集中讲审核决策，较细诊断汇总于附录 C。

## 剩余事项

- [ ] 填写 report/Report.md 中四位成员姓名和学号；组员核对技术内容并确认个人 AIGC 使用及外部协作者披露，重新构建 PDF。
- [x] 按课程通知调整为 80 cm 宽 × 100 cm 高的单页竖版，并检查 PDF 的实际尺寸与版面。
- [ ] 报告确认后，将 Report.pdf、代码、README 和数据获取说明整理为 Group_20_Project.zip；Poster.pdf 单独提交。
- [ ] 全组实际演练、Canvas 提交及个人互评。技术验收和已有讲解材料不代表这些人工事项已完成。

## 证据与入口

| 内容 | 位置 |
|---|---|
| 安装、数据、运行和报告构建 | README.md；data/README.md |
| 正式报告与可编辑正文 | report/Report.pdf；report/Report.md |
| 模型/协议冻结 | results/tables/group_freeze.json |
| 正式测试总览 | results/tables/project_test.json |
| 三数据集正式结果 | results/tables/ulb_test.json、handbook_test.json、paysim_test.json |
| 标签、重要性、金额、成本和区间 | results/tables/b_extensions_test.json、c_extensions_test.json |
| 规则及机制 | results/tables/handbook_d_test.json；ULB 在 ulb_test.json 的 E04 中 |
| 代码精简、目录移动及独立重建 | results/tables/reproduction.json |
| 完整模型与正式预测 | 本地 data/final_preparation_20261117/、data/final_test_20261117_r2/ |

历史源码、交接记录和原结果保留在 Git 历史。首次正式运行的冷启动汇总错误用同一批冻结模型修复重跑，记录仍在；后续代码整理没有更换实验方案。
