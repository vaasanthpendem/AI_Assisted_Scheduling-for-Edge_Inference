"""
RTOS-Based AI-Assisted Scheduling for Deterministic Edge TPU Inference
in Real-Time Weather Forecasting Systems

SimPy Simulation — Final Version
Latency calibrated from Tobiasz et al., JCSE 2023
DOI: 10.5626/JCSE.2023.17.2.51

Key design:
  StormClassify — heavy (100ms), comfortable deadline (350ms), HIGH fixed-priority
  SevereAlert   — heavy (130ms), TIGHT deadline (160ms),       LOW fixed-priority
  Under fixed-priority, StormClassify always runs first during bursts,
  causing SevereAlert to miss its deadline (100+130=230ms > 160ms).
  AI-assisted detects SevereAlert's tight slack and dispatches it first
  (130ms < 160ms → deadline met).
"""

import simpy, numpy as np, pandas as pd, heapq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings; warnings.filterwarnings('ignore')

# TASK DEFINITIONS
TASKS = {
    'TempAnomaly':  {
        'latency_mean':10,'latency_std':2,'latency_min':5,'latency_max':15,
        'deadline':60,'period':5,'priority_rank':2,'complexity':1},
    'HumidityTrend':{
        'latency_mean':20,'latency_std':5,'latency_min':5,'latency_max':40,
        'deadline':250,'period':20,'priority_rank':3,'complexity':2},
    'StormClassify':{
        'latency_mean':100,'latency_std':10,'latency_min':80,'latency_max':120,
        'deadline':350,'period':None,'priority_rank':1,'complexity':3},
    'SevereAlert':  {
        'latency_mean':130,'latency_std':10,'latency_min':110,'latency_max':150,
        'deadline':160,'period':None,'priority_rank':4,'complexity':4},
}

# LATENCY PREDICTOR
def build_predictor():
    X=np.array([[1],[1],[1],[2],[2],[2],[3],[3],[3],[4],[4],[4]])
    y=np.array([5,10,15,5,20,40,80,100,120,110,130,150])
    rf=RandomForestRegressor(n_estimators=100,random_state=42)
    rf.fit(X,y); return rf
PRED=build_predictor()
def predict(c): return float(PRED.predict([[c]])[0])

# SIMULATION CLASS
#This defining RTOS class is the kernel, SimPy plays the role of RTOS Kernel
#RTOS Scheduling loop : check ready queue → pick highest priority → execute → check deadline → repeat.

class RTOS:
    def __init__(self,env,sched,load=1.0):
        self.env=env; self.sched=sched; self.load=load
        self.queue=[]          # priority heap
        self.qcnt=0
        self.tpu_busy=False #tells the scheduler whether the processor (TPU) is free
        self.trigger=env.event()
        self.res=[]; self.task_ctr=0
        env.process(self._dispatcher())

    def _dispatcher(self): #this function is the scheduler — picks the tasks based on priority
        """Reads from priority queue and executes tasks on TPU in priority order"""
        while True:
            yield self.trigger
            self.trigger=self.env.event()
            while self.queue and not self.tpu_busy:
                pri,cnt,name,arr,lat,dl,done=heapq.heappop(self.queue)
                self.tpu_busy=True
                self.env.process(self._execute(name,arr,lat,dl,done))

    def _execute(self,name,arr,lat,dl,done):
        yield self.env.timeout(lat)
        self.tpu_busy=False
        finish=self.env.now; met=finish<=dl
        self.task_ctr+=1
        self.res.append({'task_id':self.task_ctr,'task_name':name,
                         'scheduler':self.sched,'arrival_ms':arr,
                         'finish_ms':finish,'deadline_ms':dl,
                         'actual_latency':lat,'response_time':finish-arr,
                         'deadline_met':met})
        done.succeed()
        # Trigger dispatcher for next task
        if self.queue and not self.trigger.triggered:
            self.trigger.succeed()

    def _get_priority(self,name,arr): #Slack calculation
        t=TASKS[name]
        if self.sched=='fixed':
            return t['priority_rank']
        pred=predict(t['complexity'])
        slack=(arr+t['deadline'])-self.env.now-pred
        return slack

    def submit(self,name,arr):
        """Submit task to priority queue"""
        t=TASKS[name]
        pri=self._get_priority(name,arr)
        lat=float(np.clip(np.random.normal(t['latency_mean'],t['latency_std']),
                          t['latency_min'],t['latency_max']))
        dl=arr+t['deadline']
        done=self.env.event()
        self.qcnt+=1
        heapq.heappush(self.queue,(pri,self.qcnt,name,arr,lat,dl,done))
        if not self.trigger.triggered:
            self.trigger.succeed()
        return done

    def run_periodic(self,name,dur_ms):
        period=TASKS[name]['period']*1000/self.load
        while self.env.now<dur_ms:
            self.submit(name,self.env.now)
            yield self.env.timeout(period)

    def run_bursts(self,burst_times_ms):
        """Submit burst tasks simultaneously so priority queue decides order"""
        for bt in burst_times_ms:
            if bt>self.env.now:
                yield self.env.timeout(bt-self.env.now)
            arr=self.env.now
            # Submit both at same time — priority queue picks SevereAlert first (AI)
            d1=self.submit('StormClassify',arr)
            d2=self.submit('SevereAlert',  arr)
            yield d1 & d2  # wait for both to finish before next burst

def simulate(sched,dur_s,bursts,load=1.0,seed=42):
    np.random.seed(seed)
    env=simpy.Environment()
    r=RTOS(env,sched,load); dur_ms=dur_s*1000
    env.process(r.run_periodic('TempAnomaly',dur_ms))
    env.process(r.run_periodic('HumidityTrend',dur_ms))
    env.process(r.run_bursts(bursts))
    env.run(until=dur_ms)
    return pd.DataFrame(r.res)

# EXPERIMENTS
SIM_DUR=600
SCENARIOS=[
    ('Low Load',   0.5,[60000,200000,400000]),
    ('Medium Load',1.0,[60000,120000,200000,300000,450000]),
    ('High Load',  2.0,[60000,90000,120000,180000,240000,
                        300000,360000,420000,480000,540000]),
]

print("="*68)
print("RTOS Scheduling Simulation — Weather Forecasting Edge TPU")
print("="*68)

all_data=[]
for label,load,bursts in SCENARIOS:
    print(f"\n── {label} (load={load}x, storm bursts={len(bursts)}) ──")
    for sched in ['fixed','ai_assisted']:
        df=simulate(sched,SIM_DUR,bursts,load)
        df['load']=label; all_data.append(df)
        n=len(df); m=(~df['deadline_met']).sum()
        print(f"  {sched:12s} | tasks:{n:4d} | misses:{m:3d} "
              f"| miss%:{m/n*100:5.1f} "
              f"| avg_resp:{df['response_time'].mean():7.2f}ms")

final=pd.concat(all_data,ignore_index=True)
final.to_csv('/tmp/sim_final.csv',index=False)

summ=final.groupby(['load','scheduler']).agg(
    Tasks   =('task_id','count'),
    Misses  =('deadline_met',lambda x:(~x).sum()),
    Miss_Pct=('deadline_met',lambda x:round((~x).mean()*100,2)),
    AvgResp =('response_time',lambda x:round(x.mean(),2))
).reset_index()

sv=final[final['task_name']=='SevereAlert'].groupby(['load','scheduler']).agg(
    Tasks   =('task_id','count'),
    Misses  =('deadline_met',lambda x:(~x).sum()),
    Miss_Pct=('deadline_met',lambda x:round((~x).mean()*100,1)),
    AvgResp =('response_time',lambda x:round(x.mean(),1))
).reset_index()

print("\n"+"="*68)
print("OVERALL SUMMARY TABLE")
print(summ.to_string(index=False))
print("\n"+"="*68)
print("SEVERE ALERT — CRITICAL TASK (deadline=160ms)")
print(sv.to_string(index=False))

# CHARTS
loads=['Low Load','Medium Load','High Load']
x,w=np.arange(3),0.35
CF='#E07B54'; CA='#4A90D9'
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.suptitle('RTOS AI-Assisted vs Fixed-Priority Scheduling\n'
             'Real-Time Weather Forecasting Edge TPU Simulation',
             fontsize=12,fontweight='bold')

def bar_chart(ax,data,col,title,ylabel,ylim=None):
    for i,sch in enumerate(['fixed','ai_assisted']):
        vals=[data[(data.load==l)&(data.scheduler==sch)][col].values[0]
              if len(data[(data.load==l)&(data.scheduler==sch)])>0 else 0
              for l in loads]
        bars=ax.bar(x+i*w-w/2,vals,w,label=sch.replace('_',' ').title(),
                    color=[CF,CA][i],alpha=0.85)
        for b in bars:
            h=b.get_height()
            if h>0:
                ax.text(b.get_x()+b.get_width()/2,h+0.3,f'{h:.1f}',
                        ha='center',va='bottom',fontsize=8)
    ax.set_title(title,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(loads,fontsize=9)
    ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    if ylim: ax.set_ylim(0,ylim)

bar_chart(axes[0],summ,'Miss_Pct','Overall Deadline Miss Rate (%)','Miss Rate (%)',
          ylim=max(summ['Miss_Pct'].max()*1.4,5))
bar_chart(axes[1],sv,'Miss_Pct',
          'SevereAlert Miss Rate (%)\n(Critical Task — Deadline 160ms)','Miss Rate (%)',ylim=110)
bar_chart(axes[2],summ,'AvgResp','Average Response Time (ms)','ms')

plt.tight_layout()
plt.savefig('/tmp/results_final.png',dpi=150,bbox_inches='tight')
print("\nSaved: /tmp/sim_final.csv and /tmp/RTOS_PROJECT/results_final.png")