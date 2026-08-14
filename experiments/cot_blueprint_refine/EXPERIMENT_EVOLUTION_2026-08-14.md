# Goedel-Architect COT / Blueprint 主线：代码演进与实验登记

更新时间：2026-08-14

代码基线：`goedel-architect@623399d`

范围：从完整 COT 建图与 COT refine，经过 Claim/Scope、Step-grounded、Phase-1B 局部修复，到当前 Whole-COT 全图生成与语义审计。StepFun/Qwen 节点证明作为下游辅助评估单列。早期 VerisoftBench 通用 prover 改动、纯观测工具和仅 launcher/config 的准备项不作为独立“质量实验”。

## 口径先行

- `root_proved`：Phase 2 根节点被 prover 证明；不代表 Blueprint 忠实。
- `phase1_accepted`：当时的结构/静态门/Lean 条件通过；各时期门禁不同，不能横向等同。
- `strictAccepted`：完整机械检查与当时的严格语义审计通过；人工深审已确认仍有假阳性。
- `acceptedWithWarnings`：严格审计通过，但保留 warning。
- 本文只把相同输入、相同 seed/source、相同阶段、相同判定口径的结果视为受控比较。不同 Phase-1A seed 的数字只作现象登记。
- `smoke`、部分运行、空结果文件、已验证但未启动的 launcher 均不计为完整实验结果。

## 主表：按时期和问题分组

| exp | parent exp | 问题 | 观察到的现象 | 失败根本原因 | 证据 | 修改方案 / 已做改动 | 针对的 subset | 目标 metric | 结果 | 结论 | 下一步 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0：完整 COT 建图 + RobustPA + COT refine（2026-07-13～08-07） | 原始 Goedel-Architect Blueprint/Prover | 完整 COT 建图缺少 source grounding | Blueprint 难以 ground 到原 COT 的具体步骤；省略步骤或把复杂关系压成数值/短节点；refine 容易保留原答案 | 生成器只看到整段 COT，没有稳定 source-span 契约；`root_proved` 奖励“容易证明”而非“忠实翻译”；局部节点证明反馈不能证明整图对象/关系正确 | [E01](#e01) | 增加完整对话、trace、Parquet、COT-only 对照；随后转向 provenance/static gate | 646 个 eligible 样例；另有 Wrong76 语义矩阵 | `root_proved`；refine correctness；paired Blueprint vs COT-only | 两个 646-run 为 235/646、236/646 root proved；受控 refine 中 Blueprint 595/646，COT-only 619/646，Blueprint 少 24 个正确结果 | 证明吞吐可运行，但 Blueprint 没展示 refinement 增益；早期根证明率不能当语义质量 | 保留 COT-only 对照；把 source coverage、对象/关系、root closure 变成独立指标 |
| P1：Claim/Scope + 静态语义门（2026-08-07～08-08） | P0 | Claim 粒度与静态语义退化 | 需要把完整 COT 拆成可追踪断言；同时阻止 `True`、裸 `Prop`、反身式、答案硬编码 | v2 相比 v1 减少伪 Claim，但仍产生大量极细 Claim；模型为满足“一 Claim 一落点”制造 vacuous node 和 root reachability 冲突 | [E02](#e02) | Claim/Scope v2；identifier-aware reachability；静态 anti-degeneration；整图语义审计 | 同一 Wrong76 | Claim/Scope 风险；Phase-1 accepted；静态问题数 | v1→v2：Claims 2202→1902，边界风险 195→60，accepted 3→8；46 个静态失败中重算 568 个问题 | 拆分质量改善，但 Claim 粒度仍是主约束；8/76 也不是忠实率，`counting/731` 已是漏检 | 不再把每个细 Claim 强制变成 theorem；保留 source span 作为索引而非生成硬契约 |
| P2：Step-grounded 模糊步骤拆分 + 静态门（2026-08-08） | P1 | Step 粒度能否缓解 Claim 过细 | Claim 太细；希望以接近人类推理步的 span ground Blueprint | vacuity 显著下降，但强制每个 Step 成为 root-reachable proposition 仍不合理；叙述、纠错、分隔符会被伪造为数学节点 | [E03](#e03) | lossless Step splitter；`COT_STEP:Sxxx`；一个 Step 可映射多 node；静态门 + 一次 repair + 全量 Step audit | 46 个 Wrong76 交集样例 | Step coverage；vacuity；Phase-1 accepted | 443 Steps，46/46 字符连续覆盖；四类 vacuity 250→32（-87.2%）；仅 3/46 accepted；出现 8 个仅为 `$$` 的伪 Step | Step 比 Claim 更适合作 source anchor，但 narrative order 不是 Lean DAG；不能要求每个 Step 独立成命题 | 使用完整 COT + 稀疏、多对多 Step/source-span anchors；允许说明性/错误路线作为 metadata 或 side branch |
| P3：Phase-1A/1B CRUD 局部修复 + 严格 Decompiler/Comparator（2026-08-09～08-10） | P2 | 局部 tool repair 能否提高 fidelity | 初稿局部有错，希望 tool call 只改 bad node；结构通过是否等于语义通过未知 | CRUD/standalone 检查提高结构通过率，但 local edit 缺少“obligation → affected nodes → success predicate”的闭环；严格审计揭示大量结构假阳性 | [E04](#e04) | Phase-1A 生成；Phase-1B node CRUD；standalone compile；Formal Decompiler + Strict Comparator；结构化 obligation ledger | Wrong76；严格复核 44；seed repair 22 | 结构 accepted；`strictAccepted`；负对照拒绝率 | 结构口径 41/76→54/76；严格口径随后仅 6/76；44 条离线复核仅 2/44 strict，14/14 高置信负例均拒绝 | 结构改进是真实工程收益，但不是忠实性收益；局部修复只能稳定修 binder、简单空壳和依赖遗漏 | 优先解决共享对象→关系→下游 type→root 的全局链，而不是继续增加局部重试 |
| P4：多节点 Planner / RepairSpec / Stable / Closure / Search（2026-08-10～08-11） | P3 | 多节点编排能否恢复 hard cases | 单节点 edit 无法完成跨节点修复；希望规划 subgraph、稳定提交、按需 Mathlib search | 编排层增加 token/rollback/retry；模型仍不能可靠重建概率、几何的对象与关系；原子批次容易因一个 no-op/坏节点整体失败 | [E05](#e05) | Plan/Subgraph、RepairSpec、no-op 过滤、directEdit、Stable/Closure、Search、`leanErrorsOnly` | 固定 10 条：3 controls + 7 hard cases；部分 arm 使用不同 Phase-1A seed | fixed-10 `strictAccepted`；hard recovery；tokens；commit/retry | 同 seed：Plan/Subgraph 3/10，RepairSpec 1/10，PlanDirect/DirectEdit 均 2/10，Stable/Closure 仍 2/10；fresh seed 的 16-turn DirectEdit 为 3/10；Lean-only search fresh seed 0/10 | 简单 `directEdit` 是更干净的局部 baseline；更多 controller 没增加 hard-case recovery。不同 seed 的 3/10、0/10 不能作策略因果结论 | 停止扩展 orchestration；改做全文件重生成或显式 typed Semantic DAG；Search 只用于外部 Lean 标识符类错误 |
| P5：Phase-1D 全文件再生成过渡方案（2026-08-11） | P4 | 全图协同修改能否替代局部 patch | 对象绑定需要整张图协同修改，局部修复不够 | 已实现 greedy/thinking/two-stage/full-regeneration 路由与 shared Phase-1A，但完整 D 实验没有持久化结果 | [E06](#e06) | 新增 `phase1d.py`、共享 Phase-1A、全文件 regenerate smoke；随后由新 `blueprint_generation.py` 主线替代 | A/B/C 记录 73 个 seed-eligible rows；D 有 1～2 条 smoke，计划 Wrong76 | 完整 76 条 terminal 结果 | A/B/C 目录各 73 行；D 的 Wrong76 `results.jsonl` 为 0 行。完整 full-regeneration 分支没有 population 结果 | 这是过渡实现；A/B/C 与 D 不能合成一个“v13 质量结果” | 若复用思想，应基于冻结的同一 Phase-1A seed 重跑；当前以 P6 新生成器为准 |
| P6：Whole-COT 全文件生成 + 统一机械检查 + 严格语义审计（2026-08-12） | P5 | Whole-COT 是否优于 Step/local repair | 直接从完整 COT 重建整图，避免 Step 硬契约和局部 patch deadlock | Whole-COT 略高于 Step，但 strict audit 仍被答案别名、对象替换、缺量词/最优化关系骗过 | [E07](#e07) | 将主流程收敛到 `blueprint_generation.py`；每轮整文件生成；whole-file Lean→canonical rebuild→Phase2 contract→standalone→semantic audit；Step 只保留为对照/索引 | matched Wrong76 Phase-1-only | `strictAccepted` / warning / semantic / structural；人工 FP | Whole：29 strict + 2 warning；Step：27 strict；人工深审 Whole 8/31 FP、Step 8/27 FP | Whole-COT 是更简洁的生成主线，但 `strictAccepted` 仍不能作为训练真值；Step 的价值应保留为稀疏锚点 | 增加对象/关系/概率/量词/最优化的确定性或人工校准；按冻结候选比较 audit，不要边审边重生 |
| P7：Whole-COT 语义审计协议消融（2026-08-12～08-13） | P6 | 审计成本、协议与判定稳定性 | Separate 两请求成本高；Joint 是否可合并；temperature 是否影响审计 | Separate 两组都 42/76，但只有 28 个共同接受；Joint 大量 schema error；审计本身不稳定且有假阳性 | [E08](#e08) | Separate / Joint；thinking/sampling plumbing；joint enum alias 与截断处理修复；减法 prompt | 同一 Wrong76，但每 arm 会重新生成候选 | 接受集合重合；audit error；人工 FP；耗时/token | Separate t0.6/t0.0 均 42/76，交集 28；人工 FP 8/42、9/42；Joint 63/65 个 `semanticAuditError`，0 accepted 无质量含义 | 不能从 Joint 的 0 accepted 说质量更严格；也不能把 t0 差异归因给 Judge，因为生成 sampling 同时变化 | 用同一批 frozen `formal_blueprint` 做 audit-only replay；先把 schema valid rate 做到 100%，再比较 FP/FN |
| P8：统一 Formal Blueprint + Compact/Direct + 匿名命名 + unreachable shadow（2026-08-14） | P7 | unreachable、命名泄漏与 Decompiler 成本 | unreachable 规则混合了定义引用、必要 proof spine 和合法 side branch；想检验匿名命名及去 Decompiler 的成本/质量 | 语义命名在两种审计下都比匿名命名多 7～8 个 strict；Direct 快约 34 分钟；但 accepted 集合波动大，且两种审计都已有明确假阳性。当前不是“unreachable 下降”，而是“不再硬拒绝” | [E09](#e09) | 候选先 canonicalize 为统一 `formal_blueprint`，后续 contract/canonical Lean/standalone/audit 都用它；static gate 进入 shadow；新增 semantic/anonymous naming、compact separate、direct comparator；记录 graph shadow | 四个完整 Wrong76 Phase-1-only arm | strict/semantic/structural；wall time；accepted overlap；shadow unreachable；人工 FP | Compact anonymous 32、named 39；Direct anonymous 30、named 38。Compact 103～105 min，Direct 69～71 min。Named/anonymous strict 交集仅 20、19。accepted 中仍有 24/32、28/39、24/30、35/38 带 shadow-unreachable | “非匿名更好”和“Direct 更快”有重复观察；质量因果尚未成立。更关键的是 FormalView 仍只用显式 `sorry_using`，没有复用已有 `effective_blueprint_dependencies`，会制造假 unreachable | 先统一 FormalView dependency 口径；对 frozen candidates 做 Compact vs Direct audit-only；人工深审四臂 accepted；把 proof spine、必要支持、合法 side branch 分开标注 |
| A1：下游节点证明与精确真假标签（2026-08-13～08-14，辅助线） | P6 的 45 个 accepted Blueprint | 生成 acceptance 与可证性/真假混淆 | `strictAccepted` 不说明节点可证，更不说明命题真；旧 pointwise negation 不能给参数化 theorem 做可靠 `disproved` | Prover 能力、依赖阻塞和命题真假混在一起；负向证明必须闭合 theorem 参数/假设后再取反 | [E10](#e10) | StepFun REPL prover；Goedel self-correct prover；`closed_theorem_exact_v1`；手工可回放 Lean 标签；matched 45 IDs | Whole-COT strict accepted 45 | root solved；node proved/disproved/blocked；label coverage | Qwen Phase2 8/45 roots；StepFun 2/45 roots、120 solved nodes、6 exact negations；Codex 标签 671 active nodes 中完成 663，另 8 pending | 这些是 prover/label 指标，不是生成 fidelity 指标；节点 `proved` 也不能反向证明 Blueprint 忠实 | 完成 8 个 pending；按同一 45 ID/同一 negation semantics 比较 prover；将真值标签用于错误分析而非替代语义忠实审计 |

## 证据登记

<a id="e01"></a>
### E01 — P0：完整 COT 建图与 COT refine

代码时期：`c036c6c`、`029dda5`、`1c514f6`。主要代码入口为当时的 `experiments/cot_blueprint_refine/run_experiment.py`、`run_cot_refinement.py`、`evaluate.py`，以及旧 `src/pipeline.py` / `src/orchestrator.py` / `src/prover.py`。

宏观数据：

- `qwen3_8b_397b_refine_ablation`：235/646 `root_proved`，总时长 15,828.075 秒（4h23m）。
- `qwen3_8b_blueprint_refine_40`：236/646 `root_proved`，总时长 4,147.758 秒（1h09m）。两者 Phase 1 都约 66 分钟；慢 run 的 Phase 2 Lean peak inflight 为 128，快 run 为 39。因此差异主要是证明反馈调度/长尾，不是 Blueprint 生成 token。
- 当前保存的 646×2 paired refinement 指标位于：
  `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_refine_ablation/evaluation/metrics.json`。
  Blueprint arm 595/646 correct、24 wrong→correct、34 个 answer changed；COT-only arm 619/646 correct、48 wrong→correct、59 个 answer changed。两臂均无 correct→wrong，paired outcome 为 594 both correct、25 COT-only only、1 Blueprint only、26 both wrong。
- 结论只针对这组 paired refinement：把旧 Blueprint context 加入 prompt 没有产生增益，反而少 24 个正确结果。它不等于证明所有 Blueprint 都有害。

样例/审计证据：

- 早期完整 COT 没有 source-span contract；node title、proof prose 或最后答案相同，不能说明 node formal type 覆盖了原 COT 的对象与关系。
- 完整对话 sidecar、分析 Parquet 和 pipeline snapshot 已落在上述实验目录；历史 sidecar 与实时完整记录要区分。

<a id="e02"></a>
### E02 — P1：Claim/Scope 与静态门

代码时期：Claim/Scope 主线在 `ab8b880` 前的历史提交；当前代码已删除该持久层，需用 git 历史查看。

宏观数据：

- 结论文件：`/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_claim_scope_v2_phase1_only/analysis/conclusions_zh.md`。
- v1→v2：Claims 2202→1902，Scopes 76→107，确定性 Claim 边界风险 195→60，完整 Phase 1 accepted 3→8，本地静态门拒绝 60→46。
- 对 46 个静态失败终稿从 Lean 重新计算得到 568 issues：`VACUOUS_PROP_DEFINITION` 190、`CLAIM_NOT_ROOT_REACHABLE` 163、`VACUOUS_PROP_STEP` 76、`VACUOUS_TRUE_STEP` 51、`VACUOUS_TRUE_SHELL_STEP` 29、`REFLEXIVE_STEP` 13 等。46/46 都至少有一个足够维持拒绝的真实问题。

样例证据：

- `MATH-500/test/counting_and_probability/731.json` 被当时流程接受，但 root 是 let 包裹的同义反身等式；静态反身规则和 397B audit 都漏检，说明 8/76 不能作 fidelity positives。
- `geometry/434`：几何关系大面积退化为 `Prop := True`，root 为无约束存在。
- `aime_2025/10`：约 46 个数学定义/条件被 `Prop := True` 替代。

<a id="e03"></a>
### E03 — P2：Step-grounded

代码时期：`ab8b880`。主要历史入口为 `formal_step_splitter.py`、`formal_steps.py`、`run_cot_split.py`、旧 `semantic_fidelity.py`。

宏观数据：

- 完整报告：`/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong46_step_v1_phase1_only/analysis/final_report.md`。
- 46/46 完成，443 Steps，平均 9.63 Step/样例；46/46 对原 COT 100% 连续覆盖。
- 四类 vacuity 从旧 46 条的 250 次降为 32 次（-87.2%），但终态只有 3 accepted；34 semantic gate、7 Lean check、2 semantic audit。
- 397B audit 给 18/46 PASS；加本地结构/静态门只剩 6/46；再加 Lean 只剩 3/46。
- paired 人工 review（46 条）：412/413 Step 有显式绑定；排除 forced shell 后，Step clearly better 20、Whole-COT better 10、equivalent 16。这个比较说明 Step span 有信息价值，不说明 one-Step→one-theorem 契约正确。

样例证据：

- 八条样例各出现一个仅为 `$$` 的 Step，生成器为了覆盖这些 span 制造 `True`/恒等/final-box node。
- `geometry/880`：`unfolding_valid : ... → True`，房间维度/坐标也未完整进入 root closure；397B 却给 PASS。
- `counting_and_probability/765`：先定义 `physics_students := 115`，再证明等于 115；这是答案硬编码而非计数推导。

<a id="e04"></a>
### E04 — P3：Phase-1A/1B 局部修复与严格审计

代码时期：`ab449b2`、`514b062`、`b7e6c47`。主要历史代码为旧 `src/blueprint.py`、`src/semantic_audit.py`、`run_robustpa_refine.py`。

宏观数据由原始 `results.jsonl` 重算：

| run | rows | 口径 | 结果 |
| --- | ---: | --- | --- |
| `wrong76_step_v3_phase1_ab` | 76 | 结构/旧门 | 41 accepted，35 error |
| `wrong76_step_v4_phase1_ab` | 76 | 结构/旧门 | 54 accepted，22 error |
| `wrong76_step_v5_phase1_ab_semantic_judge` | 76 | strict | 6 strict、44 semantic、25 structural、1 infra |
| `wrong76_step_v6_semantic_audit44` | 44 | frozen offline strict re-audit | 2 strict、42 semantic；14/14 高置信负例拒绝 |
| v6 seed repair | 22 | strict repair | 5 strict、11 semantic、6 structural；负例修复 2/14 |

样例证据：

- 成功边界：`counting_and_probability/765` 把 vacuous `known_quantities` 改为实际等式；`prealgebra/378` 把 root 从 `area_shaded := 6; area_shaded = 6` 改成对 `MeasureTheory.volume shaded_region` 的绑定；`intermediate_algebra/662` 只需补 root dependency。
- 失败边界：`precalculus/1056` 的 COT 是 enclosed interior `≤ 36`，Blueprint 长期只 formalize sphere boundary `= 36`；16 轮搜索/编辑仍未修复。
- `cmimc_2025/23` 的交点/鞋带公式节点即使新增，也没有进入 `pentagon_area` root 的正式对象链。

<a id="e05"></a>
### E05 — P4：局部修复编排时期

代码时期：`818ad5e`、`bb26cca`、`04a0aae`。历史核心代码为 `src/phase1b.py`。

受控 fixed-10 汇总：

| 策略时期 | strict | hard recovery | 关键成本/现象 |
| --- | ---: | ---: | --- |
| Plan/Subgraph | 3/10 | 0/7 | 8/10 有 multi-node commit；5 case 重复同一 node set ≥5 轮 |
| RepairSpec | 1/10 | 0/7 | 2/3 controls 回归；只解决 no-op/atomic mechanics |
| PlanDirect vs DirectEdit（同 seed） | 2/10 vs 2/10 | 0/7 vs 0/7 | token 约 1.35M vs 1.35M；DirectEdit 更少 structural failure |
| Stable/Closure | 2/10 | 0/7 | DirectEdit token 1.35M→2.16M，acceptance 不变 |
| Search open，fresh Phase-1A | DirectEdit-8 2/10；DirectEdit-16 3/10 | 后者 1/7 | 唯一 hard recovery `cmimc/23` 没使用 Search |
| `leanErrorsOnly`，另一个 fresh Phase-1A | 0/10 | 0/7 | search 66→8（-87.9%）；240 eligibility checks 仅 9 hit |

完整报告索引：

- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v7_phase1b_plan_subgraph_report/report.md`
- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v8_phase1b_repair_spec_report/report.md`
- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v9_comparison_report/report.md`
- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v10_comparison_report/report.md`
- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v11_search_comparison_report/report.md`
- `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong10_step_v12_search_policy_comparison_report/report.md`

样例证据：

- `counting/731`：DirectEdit 能把严格缺陷降到 1～2 个，但 rhombus 构型、随机点/测度和 probability root 仍未形成同一个 formal data flow。
- `geometry/434`：大量局部等式和 comment 不能替代平行线、等腰三角形、共享角对象。
- `hmmt/18`：DirectEdit 可持续提交，Planner/Controller 经常 0 commit；说明 orchestration 自身会阻断渐进改进。

<a id="e06"></a>
### E06 — P5：Phase-1D 过渡期

代码时期：`69bf168`。该提交新增 `src/phase1d.py`、shared Phase-1A 和 A/B/C/D 配置。

运行状态：

- A/B/C Wrong76 目录各记录 73 个 seed-eligible rows；它们不是 full-regeneration D 的 76 条 population 结果。
- `qwen3_8b_397b_wrong76_step_v13_d_full_regeneration/robustpa/blueprint/results.jsonl` 为 0 行。
- 1～2 条 smoke 只验证 reasoning parser、persistence 和 shared seed 路径。

因此这一时期的可复用结论是“整图修改方向合理、代码路径存在”，不是“某个 v13 策略质量更好”。

<a id="e07"></a>
### E07 — P6：Whole-COT 全文件生成

代码时期：`642e2bc`、`e4d7981`、`7907b28`。当前主入口为 `src/blueprint_generation.py`。

matched Wrong76 Phase-1-only：

| grounding | strict | warning | semantic reject | structural reject |
| --- | ---: | ---: | ---: | ---: |
| Whole-COT | 29 | 2 | 21 | 24 |
| Step-grounded | 27 | 0 | 22 | 27 |

人工深审：Whole accepted 31 中 8 个高置信 FP；Step accepted 27 中 8 个高置信 FP。

经典 FP：

- Whole `aime_2025/10`：只 formalize `final_answer = 69`，没有 formalize grid/counting object。文件：
  `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_whole_cot_blueprint_generation/robustpa/blueprint/blueprints/qwen3_8b_math_verify/aime_2025/robustpa_aime_2025_10/generation_round_3.lean`。
- Step `MATH-500/test/counting_and_probability/731.json`：出现 rectangle/bisector/rhombus 名词，但 root 没有随机点、region measure 或 event binding。
- Whole/Step 都说明：自然语言名字、proof description 和正确答案 literal 不能替代 formal object/relationship closure。

<a id="e08"></a>
### E08 — P7：语义审计协议消融

原始汇总：`/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_semantic_audit_ablation_suite_runtime_attempt_20260812T174904Z_952639/semantic_ablation_summary.json`。

| arm | strict | semantic | structural | audit error | 人工 FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| separate t0.6 | 42 | 17 | 16 | 1 | 8/42 |
| separate t0.0 | 42 | 13 | 20 | 1 | 9/42 |
| joint t0.6 | 0 | 0 | 11 | 65 | 无法评价 |
| joint t0.0 | 0 | 0 | 13 | 63 | 无法评价 |

Separate 两组 strict 交集为 28，各自独有 14。Joint 主要错误为不支持的 `semantic_effect` 枚举（如 `theoremStatement`、`propertyAssertion`），所以 0 strict 是协议失效，不是严格质量结论。

代表性人工 FP：

- `geometry/465`：裸 `theta = 150` 替代具体 `∠ADC` 和圆心构造。
- `geometry/880`：把一次 unfolded distance 命名为 shortest path，没有合法 surface path 集合和最小性。
- `aime_2025/13`：交概率被硬编码为 `1/5`、`1/3`、`1/15`，没有 sample space/event。
- `hmmt_feb_2025/20`：期望合并时间直接定义为 `n_arcs`。
- `counting_and_probability/430`：用 `Set color_set` 表示转移后可能有重复颜色的袋子。

<a id="e09"></a>
### E09 — P8：当前 8.14 四臂消融

代码：`623399d`。

当前已经实现：

1. `_validate_round` 在原始 whole-file Lean 通过后，先 `canonicalize_blueprint`；之后 Phase2 contract、canonical Lean、standalone、static shadow 和 semantic audit 都读取同一个 `formal_blueprint`。
2. `semantic_audit_mode=compact_separate` 保留 blind Formal Decompiler，但压缩 payload/schema；`direct` 在一次请求中直接读取 sanitized FormalView，不运行 Decompiler。
3. `generation_node_naming=anonymous` 强制 definitions 为 `d1...`、proof nodes 为 `n1...`、root 为 `n_final`；Judge prompt 明确把名字视为 opaque handle。
4. unreachable 不再由 static gate 硬拒绝，改为 `phase1GraphShadow` 和结果中的 `graphShadow`；Comparator 只在缺失 material path 时报告 dependency issue。

四臂宏观结果：

| arm | strict | semantic | structural | audit error | elapsed | strict 中带 shadow-unreachable | strict unreachable node 总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact + anonymous | 32 | 24 | 20 | 0 | 6,228.062s / 1h43m | 24/32 | 227 |
| Compact + semantic names | 39 | 19 | 17 | 1 | 6,321.739s / 1h45m | 28/39 | 151 |
| Direct + anonymous | 30 | 18 | 28 | 0 | 4,290.406s / 1h11m | 24/30 | 235 |
| Direct + semantic names | 38 | 12 | 26 | 0 | 4,149.332s / 1h09m | 35/38 | 288 |

集合稳定性：

- Compact named/anonymous strict 交集 20；named-only 19，anonymous-only 12。
- Direct named/anonymous strict 交集 19；named-only 19，anonymous-only 11。
- Compact/Direct named strict 交集 28；Compact-only 11，Direct-only 10。
- 因为每个 arm 会根据各自 audit feedback 重新生成整份 Blueprint，acceptance 差异同时包含“候选变化”和“审计变化”；不是 frozen-candidate audit A/B。

样例证据与当前缺口：

- Compact named 接受 `geometry/880`。最终 root 只断言存在 `path_length = 2√29` 且等于某次 unfold 后的 Euclidean distance；没有对所有合法 surface paths 的最小性。文件：
  `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_compact_separate_named_t00/robustpa/blueprint/blueprints/qwen3_8b_math_verify/MATH_500/robustpa_MATH_500_test_geometry_880_json/round_00_phase1.lean`。
- Direct named 接受 `counting_and_probability/731`。它定义了 `favorable_region`，但又把 `favorable_region_area` 直接定义成对角线公式，没有任何 measure/area theorem 把该 Set 与面积连接；root 只是公式代数化。文件：
  `/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_direct_comparator_named_t00/robustpa/blueprint/blueprints/qwen3_8b_math_verify/MATH_500/robustpa_MATH_500_test_counting_and_probability_731_json/round_00_phase1.lean`。
- Compact anonymous 的 `prealgebra/1003` 被判 dependency issue：`d13` 被 root 的 type 和 `n8` 引用，却未出现在 audit `root_closure`。当前 `src/semantic_fidelity.py::effective_blueprint_dependencies` 会从 formal declaration 推断 identifier edge；但 `src/semantic_audit.py::build_formal_view` 仍只复制显式 `node.dependencies`。这与“统一 dependency 口径”的目标不一致，也是当前大量 shadow-unreachable 的一个确定根因。
- 因此当前结果证明的是：named 两次 aggregate 都更高、Direct 两次都更快；尚未证明 anonymous 本身降低 fidelity，也未证明 Direct 与 Compact 质量等价。

建议的下一组受控实验：

1. 让 `build_formal_view` 与 static fidelity 共用同一个 effective dependency builder，并用 `prealgebra/1003` 做回归测试。
2. 冻结四臂第一次 mechanical-pass 的 `formal_blueprint`，对同一 SHA 分别 replay Compact/Direct；主要指标是 schema valid、FP/FN、token、latency，而不是重新生成后的 acceptance。
3. 对四臂 strict 集合做统一人工 rubric：root object、definitions、object/relation closure、quantifiers、probability/measure、optimization/completeness；至少覆盖所有只在单臂接受的 ID。
4. unreachable 分成 `material_proof_spine`、`required_support`、`justified_side_branch`、`omitted_from_root`；只对最后一类产生 hard defect。

<a id="e10"></a>
### E10 — A1：下游证明与节点标签

代码时期：`e9f650b`、`8a55322`、`623399d`。入口：`experiments/stepfun_blueprint_prover/` 与 `experiments/blueprint_node_labels/`。

宏观结果：

- 输入是 Whole-COT 生成中 45 个 strict accepted Blueprint；另外 31 个 Wrong76 为 Phase1 ineligible。
- matched Qwen 397B Phase2：45 eligible 中 8 roots solved，37 exhausted；按全 Wrong76 分母为 8/76，但 prover 能力比较应使用 8/45。
- StepFun 7B closed-negation run：2/45 roots solved；239 个 positive attempted 中 120 solved；6 个 `closed_theorem_exact_v1` formally negated。`blocked_by_dependency` 不代表独立尝试失败。
- Codex node truth labeling：45 Blueprints、671 active nodes；304 `definition_valid`、222 `proved`、63 `disproved`、74 `blocked_by_dependency`，另 8 pending manual。已完成 663/671，不应写成全量完成。
- Goedel-v2-8B 的 0/45 run 出现 152 `extract_failed`、42 `output_truncated`，且 Lean HTTP 为 0；这是输出提取失败的能力/协议结果，不是 Lean 证明失败率。

精确真假口径：parameter/hypothesis 先闭合，再对整个 theorem 取反；只有可回放 Lean proof 才能标 `disproved`。旧 pointwise negation 结果已归档，不能混入当前统计。

## 可复算命令

以下命令只展示核心分组；运行前确认路径仍对应上述 frozen artifacts。

```bash
# 任意 Phase-1 run 的终态分组
jq -r '.status' robustpa/blueprint/results.jsonl | sort | uniq -c

# P3 结构口径 41/76、54/76 与 strict 6/76
for n in \
  qwen3_8b_397b_wrong76_step_v3_phase1_ab \
  qwen3_8b_397b_wrong76_step_v4_phase1_ab \
  qwen3_8b_397b_wrong76_step_v5_phase1_ab_semantic_judge
do
  f="/ssd/czx/czx_work/cot_blueprint_refine/$n/robustpa/blueprint/results.jsonl"
  echo "$n"
  jq -r '.status' "$f" | sort | uniq -c
done

# P8 四臂 status 与 wall time
for n in \
  compact_separate_genanon_t00 compact_separate_named_t00 \
  direct_comparator_genanon_t00 direct_comparator_named_t00
do
  d="/ssd/czx/czx_work/cot_blueprint_refine/qwen3_8b_397b_wrong76_$n"
  echo "$n"
  jq -r '.status' "$d/robustpa/blueprint/results.jsonl" | sort | uniq -c
  jq '{total_elapsed_s,total_elapsed_time}' "$d/robustpa/blueprint/runtime_history.json"
done

# P8 accepted 中 shadow-unreachable 的样例数与节点数
jq -s '{
  strict: map(select(.status == "strictAccepted")) | length,
  strict_with_unreachable: map(select(
    .status == "strictAccepted" and
    (.generation_validation.graphShadow.unreachableNodeNames | length) > 0
  )) | length,
  strict_unreachable_nodes: map(select(.status == "strictAccepted") |
    (.generation_validation.graphShadow.unreachableNodeNames | length)) | add
}' robustpa/blueprint/results.jsonl
```

## 总结判断

主瓶颈已经发生三次迁移：

1. 从“Blueprint/Prover 是否能跑”迁移到“每个 source span 是否有正式落点”；
2. 再从“结构/Lean 是否通过”迁移到“对象、关系、量词和 root closure 是否忠实”；
3. 当前迁移到“如何在不过度惩罚 side branch 的前提下，稳定审计 material dependency，并降低审计成本”。

现阶段最可靠的工程结论是：Whole-COT 全文件生成应保留为主线，Step/source-span 应保留为稀疏证据锚点，`formal_blueprint` 应成为所有后续检查和持久化的唯一对象。最不可靠的结论仍是把 `strictAccepted` 直接当作 fidelity truth；P6、P7、P8 都已有可复现的深审假阳性。
