"""Run once after discovery and before processing held-out trip outcomes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'refine-logs'

def run():
    assert (ROOT/'results/discovery_summary.json').exists()
    assert not (ROOT/'data/processed/daily_2024.parquet').exists(), 'Validation already processed'
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite a frozen protocol'
    OUT.mkdir(exist_ok=True)
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M%S')
    d=json.loads((ROOT/'results/discovery_summary.json').read_text())
    cohort=json.loads((ROOT/'results/discovery_station_cohort.json').read_text())
    spec=dict(frozen_at=now.isoformat(),discovery_year=2023,validation_year=2024,
              hypothesis='Rain is negatively associated with the casual/member trip ratio after adjustment; fixed-station replication checks composition.',
              primary_treatment='PRCP >= 1 mm',primary_control='PRCP < 1 mm',primary_outcome='log(casual/member)',
              model='OLS; cubic temperature spline df5 bounds[-40,50]; AWND; log1p(SNOW); log1p(SNWD); categorical month and weekday; US federal holiday',
              primary_inference='HAC covariance with 7 lags; two-sided 95% CI',success='beta < 0 and CI upper bound < 0',
              station_cohort_selection='2023 only: total>=1000; active days>=180; each member type>=100',station_cohort=cohort,
              matched_experiment='2024 rain>=1 vs PRCP=0; same month and weekday; abs temp diff<=3C; optimal one-to-one no replacement; month-block bootstrap 10000',
              bootstrap_seed=20261112,
              sensitivity=['zero-rain controls','Midway rain/temp/wind and O Hare snow','duration (0,7 days]','Apr-Oct','rain threshold 5mm','equal-weight fixed station log ratios with 0.5 smoothing','fixed cohort aggregate counts'],
              prediction='OLS 2023 -> 2024; log ratio; calendar, weather without rain, weather with rain; MAE/RMSE; no test-year tuning',
              note='Local freeze after exploratory analysis; not externally registered; no causal identification claim')
    plan=f'''# 实验计划

日期：{now.isoformat()}。使用 ARIS experiment-plan 技能整理。研究目标是验证观测关联，不声称新的因果识别方法。

## 问题与主张映射

| 主张 | 最低证据 | 实验 |
| --- | --- | --- |
| C1：雨天临时/会员比例的负关联可跨年复现 | 2024年预设主模型beta<0且95%CI上界<0 | B1 |
| C1的支持性线索：关联并非完全来自站点构成变化 | 固定2023站点队列下方向仍为负 | B2 |
| 需排查的替代解释：季节、温度、星期和清洗阈值驱动原始差异 | 调整模型、匹配与敏感性方向比较 | B1/B3 |

## 探索背景

2023年日度回归估计雨天比例变化{d['effects'][0]['percent']:.2f}%。该结果仅用于形成假设，不用于最终确认。选定{len(cohort)}个固定站点ID。2024年文件允许先下载，但此时未处理其行程结果。

## 实验块

### B1：独立年份主实验（MUST）

- 数据：2023年探索；2024年全部366天验证；因变量log(casual/member)。
- 模型：OLS，雨天≥1mm；温度三次样条df5、边界[-40,50]，风速、log1p降雪/积雪、月份/星期固定效应、美国联邦节假日。
- 推断：HAC滞后7天，双侧95%CI。成功：beta<0且CI上界<0。
- 失败解释：跨年稳定性未获支持，报告阴性结果，不重选阈值。
- 产物：validation_effects.csv第一行、07_validation_forest.png。

### B2：固定站点复核（MUST）

- 队列只用2023年选定：≥1000行程、≥180活跃天、每类≥100行程。
- 2024年固定站点×日网格；日内等权平均log((casual+0.5)/(member+0.5))，同B1控制变量。
- 平衡面板的共同日期解释变量下，点估计等价站点固定效应；按天HAC，不将站点当独立天气实验。
- 同时检验固定队列汇总比例，减少平滑依赖。
- 失败解释：站点构成或低流量/零值可能影响全市关联。

### B3：匹配日额外实验与敏感性（MUST）

- 2024年雨≥1mm对0mm；同月份和星期、温差≤3°C，最优一对一不放回。
- 指标：配对log比例差，转相对百分比；按月份块bootstrap10000次，种子20261112；同时报告温度、风雪、节假日平衡。
- 敏感性：完全无雨对照；Midway雨温风+O’Hare雪；0秒至7天时长；仅4—10月；5mm阈值。
- 成功线索：方向一致；不要求每项检验都显著，也不据此重定义主检验。
- 产物：matched_pairs.csv、validation_effects.csv、08_matching_prediction.png。

### B4：跨年预测与失败诊断（辅助）

- 三个简单OLS基线：日历、天气不含降雨、天气含降雨。2023年训练、2024年测试，MAE/RMSE；不调参。
- 真实天气作为已知输入，不声称可提前预测；从测试残差挑出待核查日期，不给出未经证实的事件解释。
- 预测不改善也如实报告；显著系数不保证预测改善。

## 执行顺序与决策点

| 阶段 | 工作 | 决策 | 成本 |
| --- | --- | --- | --- |
| M0 | 下载、质量检查、2023探索 | 汇总守恒与天气完整才继续 | CPU/网络 |
| M1 | 冻结本方案及代码哈希 | 不读取2024行程结果选模型 | 秒级 |
| M2 | 2024清洗与B1/B2 | 阴性也保留结果 | CPU分钟级 |
| M3 | B3/B4与证据核对 | 不根据验证年改主检验 | CPU分钟级 |
| M4 | 中文报告、图表和交付 | 数字均可追到CSV | CPU秒级 |

## 预算、风险与范围

GPU 0小时；使用完整24个月行程，原始压缩数据约数百MB；磁盘保留压缩原件与聚合表，不展开全量CSV。最大瓶颈为官方下载速度。随机性仅在bootstrap，固定一个种子；确定性回归不需要三次随机种子重复。没有新模型贡献，不引入LLM预测器或复杂消融。局地天气、时段、车辆供给和真实出行目的均不可识别；只得出初步关联证据。

检查项：主检验/阈值/拆分已固定；独立验证；替代解释检查；模型保持简单；辅助结果单列；阴性结果允许；不主张因果或论文首创。
'''
    tracker='''# 实验执行跟踪

| Run ID | 阶段 | 目的 | 数据拆分/系统 | 指标 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | 质量检查与探索 | 2023清洗与日/站点统计 | 守恒/缺失/探索系数 | MUST | DONE |
| R002 | M1 | 固定验证方案 | protocol.json及代码哈希 | 冻结记录 | MUST | DONE |
| R003 | M2 | 主检验 | 2024全市日度 | beta/95%CI | MUST | TODO |
| R004 | M2 | 站点构成复核 | 固定2023队列 | beta/95%CI | MUST | TODO |
| R005 | M3 | 匹配与敏感性 | 2024配对与替代口径 | 配对效应/CI | MUST | TODO |
| R006 | M3 | 预测与异常诊断 | 2023训练→2024测试 | MAE/RMSE/残差 | AUX | TODO |
'''
    for name,content in [('EXPERIMENT_PLAN.md',plan),('EXPERIMENT_TRACKER.md',tracker),('protocol.json',json.dumps(spec,indent=2,ensure_ascii=False))]:
        p=Path(name);(OUT/f'{p.stem}_{stamp}{p.suffix}').write_text(content);(OUT/name).write_text(content)
    files=['refine-logs/protocol.json','scripts/analyze.py','scripts/prepare.py','results/discovery_station_cohort.json']
    receipt=dict(frozen_at=now.isoformat(),sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'protocol_freeze_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':run()
