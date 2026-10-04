"""Measured-data figures with a compact analysis overview. Numerical labels read JSON.

Run from any directory. No simulation, synthetic spectrum, manual fitting,
or interpolated joint-readout measurement is performed by this script.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from output_sensitivity_style import apply, panel, save, log_ticks, D90, D99, WIDTH, BLUE, VERM, INK, GREY, LIGHT

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / 'results', ROOT / 'paper' / 'figures'
SOURCES = {}

def load(name):
    p = RES / name
    data = p.read_bytes()
    SOURCES[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

def dimensions(ax, labels, pairs, colour=INK):
    y = np.arange(len(labels))
    for i, (ninety, ninetynine) in enumerate(pairs):
        ax.plot([ninety, ninetynine], [i, i], color=colour, alpha=.55)
        ax.plot(ninetynine, i, 's', mec=colour, mfc='white', ms=5)
        ax.plot(ninety, i, 'o', color=colour, ms=3.3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Concentration count')
    ax.set_xlim(0, max(p[1] for p in pairs)*1.25)
    ax.legend(handles=[Line2D([], [], color=colour, marker='o', ls='', label=D90),
                       Line2D([], [], color=colour, marker='s', mfc='white', ls='', label=D99)],
              loc='lower right', ncol=2)

def concentration():
    c = load('E42_baaiworm_connectome.json')
    mech = load('E45b_baai_named_channels.json')
    eig = load('AB7_baai_participation.json')
    points = load('E58_baai_multipoint.json')
    fig = plt.figure(figsize=(WIDTH, 6.0))
    overview = fig.add_axes([.03,.785,.94,.165])
    overview.set_xlim(0,1);overview.set_ylim(0,1);overview.axis('off')
    overview.text(0,1.06,'a',fontweight='bold',fontsize=9,va='bottom')
    overview.text(.025,1.06,'Measurements, sensitivity and recovery',fontsize=7.5,va='bottom')
    def box(x,y,w,h):
        overview.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.018',
                                        linewidth=.65,edgecolor=LIGHT,facecolor='white'))
    def arrow(start,end):
        overview.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=7,
                                          linewidth=.8,color=INK))
    box(.01,.22,.20,.59);box(.27,.22,.24,.59)
    box(.61,.57,.37,.29);box(.61,.14,.37,.29)
    overview.text(.11,.68,'Model parameters',ha='center',va='center')
    # Abstract wiring icon: no anatomical or quantitative claim is encoded.
    nodes=np.array([[.05,.40],[.09,.51],[.12,.35],[.16,.48],[.18,.32]])
    for i,j in ((0,1),(0,2),(1,3),(2,3),(2,4),(3,4)):
        overview.plot(nodes[[i,j],0],nodes[[i,j],1],color=LIGHT,lw=.8)
    overview.scatter(nodes[:,0],nodes[:,1],s=8,color=INK,zorder=3)
    overview.text(.39,.68,'Measured output',ha='center',va='center')
    # A genuine baseline motor-voltage trace, not a fabricated demonstration.
    matrix_path=RES/'revision_20261004/E42_motor_voltage_matrices.npz'
    SOURCES['revision_20261004/E42_motor_voltage_matrices.npz']=hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    with np.load(matrix_path) as matrices:
        baseline=matrices['baseline'];start=baseline.shape[1]//5
        displayed=baseline[:,start:]
        trace=displayed[np.argmax(np.std(displayed,axis=1))]
    trace_ax=fig.add_axes([.295,.851,.175,.027])
    trace_ax.plot(np.arange(trace.size),trace,color=BLUE,lw=.9)
    trace_ax.set_axis_off()
    overview.text(.39,.32,'Motor voltage',ha='center',va='center',fontsize=7)
    overview.text(.795,.715,'Sensitivity spectrum: concentration',ha='center',va='center',fontsize=7)
    overview.text(.795,.285,'Known-truth fitting: parameter recovery',ha='center',va='center',fontsize=7)
    arrow((.215,.52),(.265,.52))
    overview.text(.238,.84,'Perturb',ha='center',va='center',fontsize=7)
    arrow((.515,.52),(.595,.715));arrow((.515,.52),(.595,.285))
    overview.text(.795,.005,'Evaluated separately',ha='center',va='bottom',fontsize=7,color=GREY)
    # Export the same vector overview independently for presentations/reuse.
    overview_bounds=Bbox.from_extents(.02*WIDTH,.775*6.0,.98*WIDTH,.985*6.0)
    for extension in ('pdf','png','svg'):
        fig.savefig(OUT/('sensitivity_overview.'+extension),bbox_inches=overview_bounds,dpi=600)
    a=fig.add_axes([.17,.435,.32,.23]);b=fig.add_axes([.65,.435,.32,.23])
    d=fig.add_axes([.17,.10,.32,.23]);e=fig.add_axes([.65,.10,.32,.23])
    # Stored full eigenspectrum is used when available. Otherwise show only
    # measured summary counts, never reconstruct a curve from d90/d99.
    new_path = RES / 'revision_20261004' / 'E42_full_spectrum_rerun.json'
    if new_path.exists():
        new = load('revision_20261004/E42_full_spectrum_rerun.json')
        c = new['readouts']['motor_voltage']
        for key,col,label in [('motor_voltage',BLUE,'Motor voltage'),('muscle_activation',VERM,'Muscle activation')]:
            ev = np.sort(np.maximum(np.asarray(new['readouts'][key]['eigenvalues'], float), 0))[::-1]
            a.plot(np.arange(1, len(ev)+1), np.cumsum(ev)/ev.sum(), color=col,label=label)
            for q in (.90, .99):
                k = int(np.searchsorted(np.cumsum(ev)/ev.sum(), q)+1)
                a.plot(k,q,marker='o' if q==.90 else 's',mfc=col if q==.90 else 'white',mec=col,ms=4)
                a.annotate(str(k),(k,q),xytext=(3,-12 if key=='motor_voltage' else 5),textcoords='offset points',fontsize=7)
        a.set_xscale('log')
        log_ticks(a, 'x')
        a.set_xlabel('Connection-combination rank')
        a.set_ylabel('Cumulative squared sensitivity')
        a.legend(loc='lower right')
        a.set_ylim(0, 1.05)
    else:
        dimensions(a, [f"Connections\n({c['n_parameters']})", f"Class gains\n({mech['n_mechanisms']})"],
                   [(c['eff_dim_90'], c['eff_dim_99']), (mech['eff_dim_90'], mech['eff_dim_99'])])
        a.set_ylim(1.7, -.5)
    panel(a, 'b', 'Motor and muscle outputs')

    ev = np.array([x['eigenvalue'] for x in eig['per_eigenvector']])
    b.plot(np.arange(1,len(ev)+1), ev/ev.sum(), 'o-', color=INK, ms=3)
    b.set_yscale('log'); b.set_xlabel('Conductance-combination rank')
    log_ticks(b)
    b.set_ylabel('Fraction of squared sensitivity')
    b.set_xticks([1, 4, 8, 12, 16])
    panel(b, 'c', 'Conductance-gain spectrum')

    cats = c['by_connection_category']
    labels = ['Chemical\nsynapses', 'Gap\njunctions']
    shares = [cats[k]['share_of_curvature'] for k in ('syn','gj')]
    d.barh(np.arange(2), shares, color=INK, height=.5)
    d.set_yticks(np.arange(2), labels); d.invert_yaxis()
    d.set_xlim(0,1.08); d.set_xlabel('Fraction of squared sensitivity')
    for i,(s,k) in enumerate(zip(shares, ('syn','gj'))):
        d.text(s+.02, i, f'{100*s:.0f}%', va='center')
    panel(d, 'd', 'Connection classes')

    recs = list(points['per_point'].values())
    e.plot(np.arange(len(recs)), [x['eff_dim_90'] for x in recs], 'o-', color=INK, ms=4, label=D90)
    e.plot(np.arange(len(recs)), [x['eff_dim_99'] for x in recs], 's--', color=INK, mfc='white', ms=4, label=D99)
    e.set_xticks(np.arange(len(recs)), ['Reference','1.25\na','1.25\nb','1.5\na','1.5\nb'])
    e.set_xlabel('Parameter displacement (fold width)')
    e.set_ylabel('Concentration count')
    e.legend(loc='center right'); e.set_ylim(0, max(x['eff_dim_99'] for x in recs)*1.12)
    panel(e, 'e', f"{points['n_connections_sampled']}-connection subset")
    save(fig, OUT, 'output_concentration')

def measurement():
    data = load('AB4_flygym_delta.json')
    rows = data['by_delta']
    chosen = next(x for x in rows if x['delta']==.25)
    coarse_meta=load('E1_flygym_rank.json')
    ncoarse=coarse_meta['coarse']['n_observables']
    fig, ax = plt.subplots(1,3,figsize=(WIDTH,3.0))
    fig.subplots_adjust(left=.14,right=.97,bottom=.25,top=.78,wspace=.65)
    labels = [f"Summary\nn = {ncoarse}", f"Joint angles\nn = {data['n_rich_observables']}"]
    dimensions(ax[0], labels, [(chosen[k]['eff_dim_90'],chosen[k]['eff_dim_99']) for k in ('coarse','rich')])
    ax[0].set_ylim(1.8,-.5)
    panel(ax[0], 'a', 'Recorded output')
    steps = [r['delta'] for r in rows]
    ax[1].plot(steps, [r['response_over_delta'] for r in rows], 'o-', color=INK, ms=4)
    ax[1].set_xlabel('Relative perturbation step')
    ax[1].set_ylabel('Response norm / step')
    panel(ax[1], 'b', 'Perturbation response')
    for key,marker,fill in [('eff_dim_90','o',INK),('eff_dim_99','s','white')]:
        ax[2].plot(steps,[r['rich'][key] for r in rows],marker+'-',color=INK,mfc=fill,ms=4,label=D90 if marker=='o' else D99)
    ax[2].set_xlabel('Relative perturbation step')
    ax[2].set_ylabel('Concentration count')
    ax[2].legend(loc='upper right')
    panel(ax[2], 'c', 'Joint-angle sensitivity')
    save(fig, OUT, 'measurement_dependence')

def context():
    data = load('BAAI_chemo.json')
    f = load('E36b_flygym_validstep.json')
    union = load('B6_union_crossbehaviour.json')
    current = RES/'revision_20261004/B6_full_union_rerun/result.json'
    if current.exists():
        complete=load('revision_20261004/B6_full_union_rerun/result.json')
        union={'eig_'+k:v['eigenvalues'] for k,v in complete['summary'].items()}
    fig,axs = plt.subplots(1,3,figsize=(WIDTH,3.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.34,top=.80,wspace=.65)
    a,b,c=axs
    y=np.arange(2)
    for key,offset,colour,label in [('close',-.15,BLUE,'Food close'),('far',.15,VERM,'Food far')]:
        vals=[data['conditions'][key][x] for x in ('chemo_perturb_behav_change','motor_perturb_behav_change')]
        a.barh(y+offset,vals,height=.28,color=colour,label=label)
    a.set_yticks(y,['Sensory\ngains','Motor\ngains']);a.invert_yaxis()
    a.set_xlabel('Standardized output change')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.30))
    panel(a,'a','Food-location response')
    for k,col,marker in [('chemo',BLUE,'o'),('loco',VERM,'s'),('union',INK,'^')]:
        ev=np.asarray(union['eig_'+k])
        # Values numerically below 1e-10 of the leading eigenvalue are
        # unresolved; omit those markers rather than substitute a floor.
        resolved=ev>ev.max()*1e-10
        b.plot(np.arange(1,len(ev)+1)[resolved],ev[resolved],marker+'-',color=col,ms=3,label={'chemo':'Chemotaxis','loco':'Locomotion','union':'Combined'}[k])
    b.set_yscale('log');b.set_xticks([1,2,4,6,8] if current.exists() else [1,2,3,4,5,6])
    log_ticks(b)
    b.set_xlabel('Eigenvalue rank');b.set_ylabel('Squared sensitivity')
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.30));panel(b,'b','Sensitivity eigenvalues')
    rows=f['curve']
    for key,marker,fill in [('90','o',INK),('99','s','white')]:
        c.errorbar([r['n_behaviours'] for r in rows],[r['eff_dim_'+key+'_mean'] for r in rows],
                   yerr=[r['eff_dim_'+key+'_sd'] for r in rows],fmt=marker+'-',mfc=fill,color=INK,ms=4,capsize=2,label=D90 if key=='90' else D99)
    c.set_xlabel('Descending commands');c.set_ylabel('Combined concentration count')
    c.set_xticks([r['n_behaviours'] for r in rows]);c.legend(loc='center right')
    panel(c,'c','Fly walking conditions')
    save(fig,OUT,'context_dependence')

def recovery():
    models=load('cell_panel3_result.json')['by_model']
    models.update(load('cell_panel3b_result.json')['by_model'])
    show=load('mapk_showcase.json')
    ladder=load('cell_ladder_result.json')['grid']
    fig=plt.figure(figsize=(WIDTH,5.7))
    gs=fig.add_gridspec(2,2,left=.12,right=.96,bottom=.11,top=.90,hspace=.65,wspace=.85)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
    t=np.asarray(load('revision_20261004/mapk_trace_metadata.json')['time_seconds'])
    keep=t>=t.max()/2
    for key,col,ls,lab in [('truth',BLUE,'-','Generating parameters'),('recovered',VERM,'--','Approximate fit')]:
        a.plot(t[keep],np.asarray(show['traces'][key])[keep],color=col,ls=ls,label=lab)
    a.set_xlabel('Time (s)');a.set_ylabel('MAPK-PP')
    ah,al=a.get_legend_handles_labels()
    fig.legend(ah,al,loc='center',bbox_to_anchor=(.32,.515),ncol=2,columnspacing=.8)
    panel(a,'a','MAPK output example')
    for name,r in models.items():
        if r.get('rel_after_median') is None: continue
        b.scatter(r['n_params'],r['rel_after_median'],s=15+45*r['convergence_rate'],
                  marker='o' if r['class']=='neural' else '^',color=INK)
        if name in ('MAPK','Repressilator','Goodwin','GoldbeterMitotic'):
            offset={'MAPK':(0,8),'Repressilator':(0,10),'Goodwin':(-2,-14),'GoldbeterMitotic':(-2,8)}[name]
            align={'MAPK':'center','Repressilator':'center','Goodwin':'right','GoldbeterMitotic':'right'}[name]
            b.annotate(name.replace('GoldbeterMitotic','Goldbeter'),(r['n_params'],r['rel_after_median']),
                       xytext=offset,textcoords='offset points',ha=align,fontsize=7)
    b.set_yscale('log');log_ticks(b);b.set_ylim(1e-7,2);b.set_xlabel('Free parameters');b.set_ylabel('Median normalized parameter error')
    b.legend(handles=[Line2D([],[],color=INK,marker='o',ls='',label='Neuronal'),
                      Line2D([],[],color=INK,marker='^',ls='',label='Biochemical')],loc='lower right')
    # Marker-size illustrations use the same area mapping as every model point.
    fractions = [.25, .5, 1.]
    size_key = [Line2D([], [], color=INK, marker='o', ls='',
                       markersize=np.sqrt(15+45*f), label=f'{100*f:g}%') for f in fractions]
    fig.legend(handles=size_key, title='Fits below threshold', loc='center',
               bbox_to_anchor=(.80,.515), ncol=len(fractions), title_fontsize=7,
               columnspacing=.7, handletextpad=.35)
    panel(b,'b','Parameter recovery')
    obs=['single','fi7','trace1','trace5'];pars=[3,5,8]
    errors=np.array([[ladder[f'{o}_p{p}']['recovery']['rel_theta_median'] for o in obs] for p in pars])
    loss=np.array([[ladder[f'{o}_p{p}']['recovery']['L_final_median'] for o in obs] for p in pars])
    im=c.imshow(errors,cmap='Greys',vmin=0,vmax=.6,aspect='auto')
    for i,p in enumerate(pars):
        for j,o in enumerate(obs):
            z=ladder[f'{o}_p{p}']
            text=f"{errors[i,j]*100:.0f}%"+('†' if loss[i,j]>=.02 else '')+f"\n{D90}={z['curvature']['eff_dim_90']}"
            c.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if errors[i,j]>.32 else INK)
    c.set_xticks(range(4),[str(ladder[f'{o}_p3']['curvature']['n_obs']) for o in obs])
    c.set_yticks(range(3),pars);c.set_xlabel('Recorded observables');c.set_ylabel('Free parameters')
    panel(c,'c','Hodgkin–Huxley assay grid')
    cb=fig.colorbar(im,ax=c,fraction=.04,pad=.03);cb.set_label('Normalized parameter error')
    ordered=sorted(models.items(),key=lambda kv:(kv[1]['n_params'],kv[0]))
    d.barh(np.arange(len(ordered)),[r['convergence_rate'] for _,r in ordered],color=INK,height=.6)
    short={'WangBuzsaki':'Wang–Buzsáki','FitzHughNagumo':'FitzHugh–Nagumo','MorrisLecar':'Morris–Lecar','GoldbeterMitotic':'Goldbeter','HH_Acurrent':'HH + A current'}
    d.set_yticks(np.arange(len(ordered)),[short.get(n,n) for n,_ in ordered],fontsize=7)
    d.invert_yaxis();d.set_xlim(0,1.05);d.set_xlabel('Fraction below output-error threshold')
    panel(d,'d','Fit success across models')
    save(fig,OUT,'sensitivity_and_recovery')

def scaling():
    rows=load('cell_scale_result.json')['rows']
    fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.7))
    fig.subplots_adjust(left=.10,right=.96,bottom=.23,top=.79,wspace=.38)
    for budget,col,marker in [(10,BLUE,'o'),(30,VERM,'s')]:
        rs=[r for r in rows if r['obs_per_param_target']==budget]
        for ax,key in zip(axs,['constrained_fraction_90','eff_dim_90']):
            ax.plot([r['n_params'] for r in rs],[r[key] for r in rs],marker+'-',color=col,ms=4,label=f'{budget} observables / parameter')
            ax.set_xscale('log');ax.set_xlabel('Free conductances');ax.legend(loc='best')
            log_ticks(ax, 'x')
    axs[0].set_ylabel(D90+' / parameter count');axs[1].set_ylabel(D90)
    axs[0].set_ylim(0,1);axs[1].set_yscale('log')
    log_ticks(axs[1])
    panel(axs[0],'a','Normalized concentration count');panel(axs[1],'b','Absolute concentration count')
    save(fig,OUT,'cell_network_scaling')

def main():
    apply()
    concentration();measurement();context();recovery();scaling()
    out=RES/'revision_20261004';out.mkdir(exist_ok=True)
    (out/'figure_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n',encoding='utf-8')
    print(f'Generated five data figures in {OUT}; {len(SOURCES)} source hashes recorded.')

if __name__=='__main__': main()
