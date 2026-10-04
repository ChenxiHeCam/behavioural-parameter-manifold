"""Bounded nonlinear fitting and data-conditioned ensemble design."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing as mp,threading,time
from pathlib import Path
import numpy as np
from persistent_backend import CHANNELS,initialize,evaluate

BOUND=float(np.log(2.));PRIOR_SD=.35;NOISE=1.;H=.001

class Backend:
    def __init__(self,directory,n_workers=16):
        self.directory=Path(directory);(self.directory/'raw').mkdir(parents=True,exist_ok=True)
        self.pool=ProcessPoolExecutor(max_workers=n_workers,mp_context=mp.get_context('spawn'),initializer=initialize)
        self.lock=threading.Lock();self.pending={};self.count=0
        self.version=hashlib.sha256(Path(__file__).with_name('persistent_backend.py').read_bytes()).hexdigest()

    def request(self,stim,x):
        return {'stimulus':stim,'log_gains':dict(zip(CHANNELS,map(float,x)))}

    def key(self,request):return hashlib.sha256(json.dumps({'backend_sha256':self.version,'request':request},sort_keys=True).encode()).hexdigest()

    def many(self,requests):
        entries=[]
        for request in requests:
            key=self.key(request);path=self.directory/'raw'/f'{key}.npz'
            with self.lock:
                if path.exists():entries.append((key,request,None));continue
                future=self.pending.get(key)
                if future is None:future=self.pending[key]=self.pool.submit(evaluate,request)
            entries.append((key,request,future))
        outputs=[]
        for key,request,future in entries:
            path=self.directory/'raw'/f'{key}.npz'
            if future is None:
                with np.load(path) as z:y=z['response'].astype(float)
            else:
                y,metadata=future.result()
                with self.lock:
                    if not path.exists():
                        temp=path.with_suffix('.npz.tmp')
                        with temp.open('wb') as f:np.savez_compressed(f,response=y.astype(np.float32))
                        temp.replace(path)
                        (path.with_suffix('.json')).write_text(json.dumps({'key':key,'backend_sha256':self.version,'request':request,'metadata':metadata,'saved_epoch':time.time()},indent=2))
                        self.count+=1
                    self.pending.pop(key,None)
            outputs.append(y)
        return outputs

    def one(self,stim,x):return self.many([self.request(stim,x)])[0]

    def predicted(self,stimuli,x):return np.concatenate([y.ravel() for y in self.many([self.request(s,x) for s in stimuli])])

    def jacobian(self,stimuli,x):
        points=[];denominators=[]
        for i in range(len(x)):
            p=x.copy();q=x.copy();p[i]=min(BOUND,x[i]+H);q[i]=max(-BOUND,x[i]-H)
            points.extend((p,q));denominators.append(p[i]-q[i])
        requests=[self.request(stim,point) for point in points for stim in stimuli]
        ys=self.many(requests);m=len(stimuli)
        columns=[]
        for i,denominator in enumerate(denominators):
            p=np.concatenate([y.ravel() for y in ys[2*i*m:(2*i+1)*m]])
            q=np.concatenate([y.ravel() for y in ys[(2*i+1)*m:(2*i+2)*m]])
            columns.append((p-q)/denominator)
        return np.column_stack(columns)

    def close(self):self.pool.shutdown(wait=True)

def objective(prediction,target,x):
    residual=(prediction-target)/NOISE
    return float(.5*(residual@residual)+.5*(x@x)/PRIOR_SD**2)

def fit(backend,stimuli,observations,start,max_iterations=50):
    target=np.concatenate([y.ravel() for y in observations]);x=np.clip(np.asarray(start,dtype=float),-BOUND,BOUND)
    prediction=backend.predicted(stimuli,x);value=objective(prediction,target,x)
    radius=.05;history=[];reason='iteration_limit';jac=None;jac_point=None
    for iteration in range(max_iterations):
        if jac_point is None or not np.array_equal(jac_point,x):jac=backend.jacobian(stimuli,x);jac_point=x.copy()
        residual=(prediction-target)/NOISE;J=jac/NOISE
        precision=J.T@J+np.eye(16)/PRIOR_SD**2
        gradient=J.T@residual+x/PRIOR_SD**2
        accepted=False;old_value=value
        for attempt in range(10):
            damping=0.
            for inner in range(32):
                step=-np.linalg.solve(precision+np.eye(16)*damping,gradient)
                if np.linalg.norm(step)<=radius*(1+1e-8):break
                damping=max(1.,damping*3, np.linalg.norm(gradient)/max(radius,1e-12)*.01)
            if np.linalg.norm(step)>radius:step*=radius/np.linalg.norm(step)
            candidate=np.clip(x+step,-BOUND,BOUND);step=candidate-x
            predicted_reduction=float(-gradient@step-.5*step@precision@step)
            if np.linalg.norm(step)<1e-6 or predicted_reduction<1e-9:reason='small_step';break
            proposed_prediction=backend.predicted(stimuli,candidate)
            proposed_value=objective(proposed_prediction,target,candidate)
            ratio=(value-proposed_value)/max(predicted_reduction,1e-15)
            if ratio>.1 and proposed_value<value:
                x=candidate;prediction=proposed_prediction;value=proposed_value;accepted=True
                if ratio>.75 and np.linalg.norm(step)>.7*radius:radius=min(.2,radius*1.8)
                elif ratio<.25:radius=max(1e-5,radius*.4)
                break
            radius=max(1e-5,radius*.25)
        rmse=float(np.sqrt(np.mean((prediction-target)**2)))
        history.append({'iteration':iteration,'x':x.tolist(),'objective':value,'training_RMSE_mV':rmse,'trust_radius_L2':radius,'accepted':accepted,'step_L2':float(np.linalg.norm(step))})
        if not accepted:reason='no_accepted_step' if reason=='iteration_limit' else reason;break
        if rmse<=1.02 and old_value-value<=1e-4*max(1.,value):reason='noise_matched_small_improvement';break
        if np.linalg.norm(step)<1e-3 and rmse<=1.05:reason='noise_matched_small_step';break
    return {'x':x.tolist(),'objective':value,'training_RMSE_mV':float(np.sqrt(np.mean((prediction-target)**2))),'noise_matched':bool(np.sqrt(np.mean((prediction-target)**2))<=1.05),'termination':reason,'history':history,'iterations':len(history)}

def fitted_ensemble(backend,stim,observation,best,seed,size=16):
    x=np.asarray(best['x']);J=backend.jacobian([stim],x)
    precision=J.T@J/NOISE**2+np.eye(16)/PRIOR_SD**2
    vals,vecs=np.linalg.eigh(precision);transform=vecs/np.sqrt(np.maximum(vals,1e-12))[None,:]
    rng=np.random.default_rng(seed);points=[x];records=[{'proposal_scale':0.,'objective':best['objective']}]
    attempts=0;threshold=best['objective']+13.148
    # A data-compatible ensemble, not exact posterior samples. Proposal scales
    # are fixed in advance; nonlinear likelihood/prior gate remains unchanged.
    for scale in (1.,.5,.25):
        for batch in range(8):
            proposals=[]
            for _ in range(16):
                candidate=x+scale*(transform@rng.normal(size=16));attempts+=1
                if np.all(np.abs(candidate)<=BOUND):proposals.append(candidate)
            ys=backend.many([backend.request(stim,p) for p in proposals])
            for point,y in zip(proposals,ys):
                loss=objective(y.ravel(),observation.ravel(),point)
                if loss<=threshold and len(points)<size:
                    points.append(point);records.append({'proposal_scale':scale,'objective':loss})
            if len(points)>=size:break
        if len(points)>=size:break
    if len(points)<size:raise RuntimeError(f'Insufficient baseline-compatible ensemble: {len(points)}')
    return np.asarray(points),{'samples':records,'attempted_proposals':attempts,'acceptance_objective_threshold':threshold,'precision_eigenvalues':vals.tolist(),'base_J':J}

def select_from_ensemble(backend,candidates,baseline,points,random_seed):
    names=[name for name in candidates if name!=baseline]
    off={'off_mV':-65.,'components':[]}
    ys=backend.many([backend.request(stim,point) for stim in [off,*[candidates[n] for n in names]] for point in points])
    n=len(points);rest=np.stack([y.ravel() for y in ys[:n]])
    scores={};responses={}
    for i,name in enumerate(names):
        predictions=np.stack([y.ravel() for y in ys[(i+1)*n:(i+2)*n]])
        centered=predictions-predictions.mean(axis=0)
        small=np.eye(n)+centered@centered.T/((n-1)*NOISE**2)
        scores[name]=float(.5*np.linalg.slogdet(small)[1])
        responses[name]=float(np.sqrt(np.mean((predictions-rest)**2)))
    choice={'complementary_ensemble':max(scores,key=scores.get),'repeat':baseline,'random':str(np.random.default_rng(random_seed).choice(names)),'max_response':max(responses,key=responses.get)}
    return choice,{'ensemble_information_scores':scores,'ensemble_response_RMS_mV':responses}
