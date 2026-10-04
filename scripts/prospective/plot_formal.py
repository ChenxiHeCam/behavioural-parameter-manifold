"""Plot all retained prospective endpoints using the shared print-scale style."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import output_sensitivity_style as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parents[2]/'results/prospective_20261004/formal_results_20261004'
out = root.parent/'figures'
d = json.loads((root/'formal_report.json').read_text())
s = json.loads((root/'summary.json').read_text())
arms = d['protocol']['arms']
cases = [c for c in d['cases'] if c['status']=='complete']
assert len(cases)==s['n_complete']==d['protocol']['n_independent_truths']
rows = [[next(a for a in c['arms'] if a['arm']==arm) for arm in arms] for c in cases]
metrics = ['heldout_RMSE_mV','log_parameter_RMSE']
values = [np.array([[a[key] for a in row] for row in rows]) for key in metrics]
flags = np.array([[not a['noise_matched'] for a in row] for row in rows])
for j,arm in enumerate(arms):
    np.testing.assert_allclose(values[0][:,j].mean(),s['arms'][arm]['mean_heldout_RMSE_mV'])
    np.testing.assert_allclose(values[1][:,j].mean(),s['arms'][arm]['mean_log_parameter_RMSE'])
colors = [style.VERM,style.BLUE,style.GREY,style.GREY]
markers = ['o','o','s','^']
labels = ['Ensemble\nselection','Repeat','Random','Largest\nresponse']
style.apply()
fig = plt.figure(figsize=(style.WIDTH,4.80))
grid = fig.add_gridspec(2,2,left=.105,right=.97,bottom=.12,top=.88,
                       hspace=.86,wspace=.42,height_ratios=[2.05,1])
axes = [fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
contrast = fig.add_subplot(grid[1,:])
x = np.arange(len(arms))
offsets = np.linspace(-.095,.095,len(cases))
for k,(ax,v) in enumerate(zip(axes,values)):
    for i in range(len(cases)):
        ax.plot(x+offsets[i],v[i],color=style.LIGHT,lw=.45,zorder=1)
    for j,arm in enumerate(arms):
        good = ~flags[:,j]
        ax.scatter(j+offsets[good],v[good,j],s=14,marker=markers[j],
                   facecolors='white',edgecolors=colors[j],linewidths=.7,zorder=3)
        if flags[:,j].any():
            ax.scatter(j+offsets[flags[:,j]],v[flags[:,j],j],s=28,marker='x',
                       color=colors[j],linewidths=1,zorder=4)
        entry = s['arms'][arm]
        mean_key = 'mean_heldout_RMSE_mV' if k==0 else 'mean_log_parameter_RMSE'
        ci_key = 'heldout_mean_bootstrap95' if k==0 else 'parameter_mean_bootstrap95'
        mean = entry[mean_key]; lo,hi = entry[ci_key]
        ax.errorbar(j+.18,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.4,
                    color=colors[j],lw=1.05,capsize=2,zorder=5)
    ax.set_xticks(x,labels)
    ax.set_xlim(-.35,len(arms)-.55)
    ax.yaxis.grid(True,which='major',color=style.GRID,lw=.5)
    ax.set_axisbelow(True)
axes[0].set_yscale('log')
from matplotlib.ticker import LogLocator, FuncFormatter
axes[0].yaxis.set_major_locator(LogLocator(base=10))
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
axes[0].set_ylabel('Held-out voltage RMSE (mV)')
axes[1].set_ylabel('RMSE in log gains')
axes[1].set_ylim(bottom=0)
style.panel(axes[0],'a','Prediction on an excluded stimulus')
style.panel(axes[1],'b','Parameter recovery')
legend=[Line2D([],[],marker='o',mfc='white',mec=style.INK,ls='',ms=4,label='Independent case'),
        Line2D([],[],marker='D',color=style.INK,ls='',ms=4,label='Mean with 95% interval'),
        Line2D([],[],marker='x',color=style.INK,ls='',ms=4,label='Training RMSE > 1.05 mV')]
fig.legend(handles=legend,loc='center',bbox_to_anchor=(.54,.435),ncol=3,columnspacing=1.3,handletextpad=.4)
contrasts = [s['complementary_minus_control'][arm] for arm in arms[1:]]
means = np.array([row['mean_paired_prediction_difference_mV'] for row in contrasts])
limits = np.array([row['paired_prediction_difference_bootstrap95'] for row in contrasts])
contrast.errorbar(means,np.arange(len(contrasts)),xerr=[means-limits[:,0],limits[:,1]-means],
                  fmt='D',color=style.INK,ms=3.5,lw=1,capsize=2,zorder=3)
contrast.axvline(0,color=style.GREY,ls=':',lw=.8)
contrast.set_yticks(np.arange(len(contrasts)),['Repeat','Random','Largest response'])
contrast.set_ylim(len(contrasts)-.5,-.5)
box=contrast.get_position()
contrast.set_position([.25,box.y0,.72,box.height])
span=limits.max()-limits.min()
contrast.set_xlim(limits.min()-.08*span,limits.max()+.08*span)
contrast.set_xlabel('Mean voltage RMSE difference (mV)')
style.panel(contrast,'c','Ensemble selection − control')
style.save(fig,out,'stimulus_comparison')
caption = '''**Prospective stimulus comparison across independent simulated cases.** (a) Voltage RMSE between the fitted model and the noiseless simulated truth on one excluded URY ramp followed by a posterior step; the vertical axis is logarithmic. (b) Parameter RMSE in natural-log conductance gains. Each open marker represents one independent parameter case, with one baseline and one additional noise realization shared across strategies within that case (12 cases). Light gray lines connect the same case across strategies; small horizontal offsets separate overlapping points. Vermillion circles denote ensemble selection, blue circles denote a repeated baseline stimulus, gray squares denote a random candidate, and gray triangles denote the candidate with the largest ensemble root-mean-square voltage change relative to unstimulated responses. Crosses replace the case marker when training RMSE exceeds 1.05 mV; these observations remain in every summary. In panels a and b, filled diamonds, offset to the right of the case markers, show arithmetic means. Whiskers show pointwise descriptive 95% percentile bootstrap intervals for those means, resampling the 12 independent cases 10,000 times; they do not show intervals for paired strategy contrasts. Panel a means and intervals are computed in mV before plotting on the log axis. Each strategy observes one baseline and one additional voltage trajectory and uses the same fitting starts, parameter bounds, stopping rules and iteration caps. The ensemble-selection score is a Gaussian ensemble information proxy computed from 16 baseline-compatible models; neither the generating parameters nor the excluded response enters selection. Parameter recovery and predictive error are separate endpoints. All 48 strategy outcomes, including the one ensemble-selected outcome above the training threshold, are displayed. (c) Paired differences in mean test-voltage RMSE between ensemble selection and each control. Black diamonds and horizontal whiskers show paired means and pointwise descriptive 95% percentile intervals from the same independent-case bootstrap, preserving pairing. The dotted gray line marks zero; negative differences favor ensemble selection. This experiment does not demonstrate an average improvement by ensemble selection over the three controls.
'''
(out/'stimulus_comparison_caption.md').write_text(caption,encoding='utf-8')
provenance = {'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('formal_report.json','summary.json','summary_arrays.npz')},
              'case_count':len(cases),'strategy_outcomes':len(cases)*len(arms),
              'all_noise_match_failures_retained':True,'figure_width_inches':style.WIDTH,
              'log_axis_applies_only_to_display':True,'colors':dict(zip(arms,colors)),
              'paired_prediction_mean_differences_mV':means.tolist(),
              'paired_prediction_intervals95_mV':limits.tolist()}
(out/'stimulus_comparison_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
