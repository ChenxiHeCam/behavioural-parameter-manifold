"""Paired independent-truth summaries from the locked formal report."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,sys
import numpy as np

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'results/prospective_20261004/formal_results_20261004'
d=json.loads((root/'formal_report.json').read_text());p=d['protocol'];cases=[c for c in d['cases'] if c['status']=='complete'];arms=p['arms']
if not cases:raise RuntimeError('No completed cases; report failure denominators instead of an endpoint estimate')
pred=np.array([[next(a['heldout_RMSE_mV'] for a in c['arms'] if a['arm']==arm) for arm in arms] for c in cases])
param=np.array([[next(a['log_parameter_RMSE'] for a in c['arms'] if a['arm']==arm) for arm in arms] for c in cases])
indices=np.random.default_rng(921711).integers(0,len(cases),(10000,len(cases)))
summaries={};contrasts={}
for j,arm in enumerate(arms):
    boot_pred=pred[indices,j].mean(axis=1);boot_param=param[indices,j].mean(axis=1)
    summaries[arm]={'mean_heldout_RMSE_mV':float(pred[:,j].mean()),'heldout_mean_bootstrap95':np.quantile(boot_pred,[.025,.975]).tolist(),'median_heldout_RMSE_mV':float(np.median(pred[:,j])),'mean_log_parameter_RMSE':float(param[:,j].mean()),'parameter_mean_bootstrap95':np.quantile(boot_param,[.025,.975]).tolist(),'arm_noise_match_failures':sum(not next(a['noise_matched'] for a in c['arms'] if a['arm']==arm) for c in cases)}
for j,arm in enumerate(arms[1:],1):
    delta=pred[:,0]-pred[:,j];dp=param[:,0]-param[:,j]
    contrasts[arm]={'mean_paired_prediction_difference_mV':float(delta.mean()),'paired_prediction_difference_bootstrap95':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist(),'mean_paired_log_parameter_difference':float(dp.mean()),'paired_parameter_difference_bootstrap95':np.quantile(dp[indices].mean(axis=1),[.025,.975]).tolist(),'prediction_wins':int(np.sum(delta<0)),'prediction_ties':int(np.sum(delta==0)),'parameter_wins':int(np.sum(dp<0)),'n':len(cases)}
result={'status':'complete','n_planned':len(d['cases']),'n_complete':len(cases),'baseline_or_ensemble_failures':[{'case':c['case'],'status':c['status']} for c in d['cases'] if c['status']!='complete'],'comparison_scope':'All complete cases, with all arm fitting failures retained. Conditional if any baseline/ensemble failure occurred.','uncertainty':'Independent-truth paired bootstrap, 10000 samples, seed 921711. Descriptive 95% intervals, without a familywise significance claim.','arms':summaries,'complementary_minus_control':contrasts}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(root/'summary_arrays.npz',case_ids=np.asarray([c['case'] for c in cases]),heldout_RMSE_mV=pred,log_parameter_RMSE=param,bootstrap_indices=indices)
labels={'complementary_ensemble':'互补刺激','repeat':'重复基线','random':'随机刺激','max_response':'最大响应'}
lines=['# 正式模型实验结果','',f'固定方案下的 {len(d["cases"])} 个独立生成参数实例全部处理完毕；其中 {len(cases)} 个完成四策略比较。原始输出与失败均保留。','',
       '| 策略 | 平均未见刺激 RMSE / mV | 均值的描述性 95% 区间 | 平均 log 参数 RMSE | 未达到训练噪声水平的次数 |','|---|---:|---:|---:|---:|']
for arm in arms:
    r=summaries[arm];lo,hi=r['heldout_mean_bootstrap95']
    lines.append(f'| {labels[arm]} | {r["mean_heldout_RMSE_mV"]:.5f} | [{lo:.5f}, {hi:.5f}] | {r["mean_log_parameter_RMSE"]:.5f} | {r["arm_noise_match_failures"]}/{len(cases)} |')
lines+=['','每个策略的测量预算都是一条基线电压轨迹与一条新增轨迹。策略选择只使用基线拟合出的 16 个兼容模型；未见刺激与生成参数不参与选择。四策略使用相同拟合起点、迭代上限、参数边界与停止标准。','',
        '| 对照 | 互补减对照的平均预测误差 / mV | 配对差值 95% 区间 | 互补策略预测误差更低的实例数 |','|---|---:|---:|---:|']
for arm in arms[1:]:
    r=contrasts[arm];lo,hi=r['paired_prediction_difference_bootstrap95']
    lines.append(f'| {labels[arm]} | {r["mean_paired_prediction_difference_mV"]:.5f} | [{lo:.5f}, {hi:.5f}] | {r["prediction_wins"]}/{r["n"]} |')
lines+=['','差值为负表示互补策略预测误差较低。区间按独立参数实例配对重采样，电压采样点不是统计重复。区间跨零的比较不能支持明确的改善结论；这些是描述性区间，不作多个比较的整体显著性声明。','',
        '参数恢复与输出预测是不同终点。即使未见刺激预测有所改善，也不能据此宣称全部参数变得可识别。数据使用模型内预突触电压干预、Gaussian 独立噪声，以及从公开参考值附近生成的通道增益；结论不外推到动物实验、其他噪声模型或全参数空间。','',
        f'本轮保留 {d["retained_rollouts"]} 次不同前向模型输出。正式方案见 `protocol_locked.json`，代码快照见 `source_snapshot.zip`，完整案例见 `case_*/case_report.json`，数值复核见 `verification.json`。上一轮 491 次试运行及 101 次局部诊断仍独立保留，未替换、未纳入本轮区间。']
if result['baseline_or_ensemble_failures']:lines+=['','基线/兼容模型集合失败：`'+json.dumps(result['baseline_or_ensemble_failures'])+'`。上述终点比较仅限完成全部策略的实例。']
(root/'FORMAL_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
