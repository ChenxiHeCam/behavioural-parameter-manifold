"""Plot flyvis spectra and readout comparison from complete, retained Jacobians."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from output_sensitivity_style import apply, save, panel, WIDTH, BLUE, VERM, GREY

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/revision_20261004'
d=json.loads((RESULT/'flyvis_state_coordinate_comparison.json').read_text())
checks=json.loads((RESULT/'flyvis_verification.json').read_text())
apply()
fig,axes=plt.subplots(1,3,figsize=(WIDTH,2.65))
fig.subplots_adjust(left=.085,right=.985,bottom=.29,top=.80,wspace=.52)
states=[('initialized','Initialized',BLUE),('trained','Trained',VERM)]
for state,label,color in states:
    r=d['states'][state]
    spec=r['common_weight_matched_703_readout_sweep']['64']
    ev=np.array(spec['eigenvalues'])
    cumulative=np.cumsum(ev)/ev.sum()
    axes[0].plot(np.arange(1,len(ev)+1),cumulative,color=color,label=label)
    grids=r['common_weight_matched_703_readout_sweep']
    n=np.array(sorted(int(g) for g in grids))
    axes[1].plot(n,[grids[str(g)]['eff_dim_90'] for g in n],color=color,marker='o',ms=3)
    axes[1].plot(n,[grids[str(g)]['eff_dim_99'] for g in n],color=color,marker='s',mfc='white',ms=3,linestyle='--')
    val=r['common_weight_step_validation']
    h=np.array(sorted(float(k) for k in val))
    axes[2].plot(h,[100*val[f'{step:g}']['relative_frobenius'] for step in h],color=color,marker='o',ms=3)
axes[0].set_xscale('log')
axes[0].set(xlim=(1,len(ev)),ylim=(0,1.03),xlabel='Eigenvalue rank',ylabel='Cumulative squared sensitivity')
axes[0].set_xticks([1,10,100,len(ev)],[1,10,100,len(ev)])
for level in [.9,.99]:axes[0].axhline(level,color=GREY,ls=':',lw=.6,zorder=0)
axes[0].set_yticks([0,.5,.9,1.],[0,.5,.9,1.])
axes[1].set(xlabel='Neural activity bins',ylabel='Concentration count',xlim=(3,67),ylim=(0,36))
axes[1].set_xticks([6,16,32,64])
axes[2].set_xscale('log')
axes[2].set(xlabel='Dimensionless step',ylabel='Jacobian difference (%)')
axes[2].set_xticks(h,[f'{step:g}' for step in h])
axes[2].set_ylim(bottom=-.03)
for ax,label,title in zip(axes,['a','b','c'],['Sensitivity concentration','Recorded activity bins','Step comparison']):panel(ax,label,title)
handles=[Line2D([],[],color=c,label=l) for _,l,c in states]
handles.extend([Line2D([],[],color='black',marker='o',ms=3,label='90% mass'),
                Line2D([],[],color='black',marker='s',mfc='white',ms=3,ls='--',label='99% mass')])
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,columnspacing=1.6)
save(fig,ROOT/'paper/figures','Fig_visual_sensitivity')
