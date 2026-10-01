import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,env,title in zip(axes,['halfcheetah','pendulum'],['HalfCheetah-v5','Pendulum-v1']):
    runs=[json.loads((root/f'{env}_scaled_seed{s}.json').read_text()) for s in range(3)]
    steps=np.array([p['steps'] for p in runs[0]['snapshots']])/1000
    rewards=np.array([[p['eval_mean'] for p in d['snapshots']] for d in runs])
    mean,std=rewards.mean(0),rewards.std(0)
    for values in rewards:
        ax.plot(steps,values,color='#2369b5',alpha=.22,linewidth=1)
    ax.fill_between(steps,mean-std,mean+std,color='#2369b5',alpha=.13)
    ax.plot(steps,mean,color='#2369b5',linewidth=2.6,label='Final fix: mean of 3 seeds')
    for suffix,label,color in [('fixed','Ratio fix only (seed 0)','#797979'),('normalized','Normalization + value fix, reward scale 1 (seed 0)','#d38d22')]:
        d=json.loads((root/'ablations'/f'fpo_{env}_{suffix}_seed0.json').read_text())
        ax.plot(np.array([p['steps'] for p in d['snapshots']])/1000,[p['eval_mean'] for p in d['snapshots']],linestyle='--',linewidth=1.5,color=color,label=label)
    ax.set_title(title,weight='bold')
    ax.set_xlabel('Environment steps (thousands)')
    ax.set_ylabel('Evaluation return (raw rewards)')
    ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=1,fontsize=9,frameon=False)
fig.suptitle('FPO learning validation: separate evaluation environments',fontsize=14,weight='bold')
fig.savefig(root/'learning_curves.png',dpi=170)
