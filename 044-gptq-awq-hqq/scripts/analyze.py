"""Generate paired bootstrap estimates and publication figures from raw item records."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
METHODS=['bf16','gptq','awq','hqq']; COLORS=['#39465A','#3877B0','#D17B24','#248476']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def load(p): return json.loads(p.read_text())
def interval(x):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(44)
 means=np.concatenate([x[rng.integers(0,len(x),size=(1000,len(x)))].mean(1) for _ in range(10)])*100
 return [float(v) for v in np.percentile(means,[2.5,97.5])]
base=load(R/'bf16_predictions.json') if (R/'bf16_done.json').exists() else None
summary={}; rows=[]
for m in METHODS:
 if not (R/f'{m}_done.json').exists():
  summary[m]={'status':'FAILED','error':load(R/f'{m}_error.json') if (R/f'{m}_error.json').exists() else None};continue
 meta=load(R/f'{m}_done.json'); p=load(R/f'{m}_predictions.json')
 assert len(p)==400 and all(sum(x['dataset']==d for x in p)==200 for d in ['mmlu','arc'])
 s={'status':'OK',**meta,'datasets':{}}
 for d in ['mmlu','arc','combined']:
  ix=[i for i,x in enumerate(p) if d=='combined' or x['dataset']==d]
  x=[p[i]['correct'] for i in ix]
  rec={'n':len(ix),'correct':sum(x),'accuracy_pct':100*np.mean(x),'accuracy_ci':interval(x)}
  if base:
   assert all((p[i]['dataset'],p[i]['source_index'],p[i]['gold'])==(base[i]['dataset'],base[i]['source_index'],base[i]['gold']) for i in ix)
   delta=[p[i]['correct']-base[i]['correct'] for i in ix]
   rec.update(delta_pp=100*np.mean(delta),delta_ci=interval(delta))
  s['datasets'][d]=rec
 summary[m]=s
 rows.append({'method':m,'quant_minutes':None if meta['quant_seconds'] is None else meta['quant_seconds']/60,'size_gib':meta['model_bytes']/2**30,'quant_peak_gib':meta['quant_peak_gib'],'eval_peak_gib':meta['eval_peak_gib'],'tokens_s':meta['tokens_per_second'],'ppl':meta['ppl'],'mmlu_pct':s['datasets']['mmlu']['accuracy_pct'],'arc_pct':s['datasets']['arc']['accuracy_pct'],'combined_pct':s['datasets']['combined']['accuracy_pct'],'delta_pp':s['datasets']['combined']['delta_pp'],'delta_low':s['datasets']['combined']['delta_ci'][0],'delta_high':s['datasets']['combined']['delta_ci'][1]})
verdict='INCONCLUSIVE'
quant=[m for m in METHODS[1:] if summary[m]['status']=='OK']
if any(summary[m]['datasets']['combined']['delta_ci'][1]<-1 for m in quant):verdict='DISPROVEN'
elif len(quant)==3 and all(summary[m]['datasets']['combined']['delta_ci'][0]>=-1 for m in quant) and summary['hqq']['quant_seconds']<min(summary[m]['quant_seconds'] for m in ['awq','gptq']):verdict='PROVEN'
summary['verdict']=verdict
(R/'summary.json').write_text(json.dumps(summary,indent=2))
with open(R/'summary.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Main effect: paired differences.
fig,ax=plt.subplots(figsize=(9,4.8));ys=np.arange(len(quant))
for y,m in zip(ys,quant):
 rec=summary[m]['datasets']['combined'];v=rec['delta_pp'];lo,hi=rec['delta_ci']
 ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color=COLORS[METHODS.index(m)],markersize=9)
 ax.annotate(f'{v:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',(hi,y),xytext=(8,0),textcoords='offset points',va='center',fontsize=11)
ax.axvline(0,color='#6B7280',lw=1);ax.axvline(-1,color='#B44343',ls='--',label='Maximum hypothesized loss: 1 point')
ax.set_yticks(ys,[m.upper() for m in quant]);ax.set_xlabel('Accuracy change versus BF16 (percentage points)');ax.set_title('400 fixed questions: accuracy change with 95% paired bootstrap CI',fontsize=13,pad=16);ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=10);ax.margins(x=.45,y=.4);fig.tight_layout();fig.savefig(F/'accuracy.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.8))
vals=[summary[m]['quant_seconds']/60 for m in quant]
bars=ax.bar([m.upper() for m in quant],vals,color=[COLORS[METHODS.index(m)] for m in quant],width=.55)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.2f} min',ha='center',va='bottom',fontsize=12)
ax.set_ylabel('Wall time (minutes)');ax.set_title('Time to load cached weights and quantize',pad=15);ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.15);fig.text(.5,.01,'One run per method; downloads, installation and model serialization excluded.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(F/'quantization_time.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,key,label,title in zip(axes,['size_gib','tokens_s'],['Saved model size (GiB)','Generated tokens / second'],['Saved model footprint','Batch 1, 128 generated tokens']):
 vals=[row[key] for row in rows];bs=ax.bar([x['method'].upper()+('*' if x['method']=='awq' else '') for x in rows],vals,color=[COLORS[METHODS.index(x['method'])] for x in rows]);ax.set_ylabel(label);ax.set_title(title);ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.15)
 for b,v in zip(bs,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.1f}',ha='center',va='bottom')
fig.text(.5,.01,'*AWQ inference decompressed to BF16 linear layers. Backends were not equally optimized.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(F/'resources.png',dpi=150);plt.close(fig)
headers=['Recipe','Quant.\nmin','Disk\nGiB','Peak quant.\nGiB','Peak eval.\nGiB','Tokens/s','WT2\nPPL','MMLU\n%','ARC-C\n%','Combined %\n[95% CI]','Change pp\n[95% paired CI]']
table=[]
for row in rows:
 m=row['method'];s=summary[m];acc=s['datasets']['combined'];lo,hi=acc['accuracy_ci'];dlo,dhi=acc['delta_ci']
 table.append([m.upper()+('*' if m=='awq' else ''),'—' if row['quant_minutes'] is None else f"{row['quant_minutes']:.2f}",f"{row['size_gib']:.2f}",'—' if row['quant_peak_gib'] is None else f"{row['quant_peak_gib']:.2f}",f"{row['eval_peak_gib']:.2f}",f"{row['tokens_s']:.1f}",f"{row['ppl']:.2f}",f"{row['mmlu_pct']:.1f}",f"{row['arc_pct']:.1f}",f"{acc['accuracy_pct']:.2f}\n[{lo:.2f}, {hi:.2f}]",'reference' if m=='bf16' else f"{acc['delta_pp']:+.2f}\n[{dlo:+.2f}, {dhi:+.2f}]"])
fig,ax=plt.subplots(figsize=(19,3.5));ax.axis('off');t=ax.table(cellText=table,colLabels=headers,loc='center',cellLoc='center',colWidths=[.07,.065,.06,.08,.08,.065,.055,.065,.065,.13,.15]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.9)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('white');cell.set_facecolor('#26374D' if r==0 else ('#EEF2F7' if r%2 else '#FAFBFD'))
 if r==0:cell.set_text_props(color='white',weight='bold')
ax.set_title('Qwen3-8B on one NVIDIA L4: 4-bit recipes versus BF16',fontsize=18,pad=25);fig.text(.5,.02,'200 MMLU + 200 ARC-Challenge • 10,000 bootstrap resamples • *AWQ inference decompressed to BF16 • Times include load',ha='center',fontsize=11);fig.tight_layout();fig.savefig(F/'table.png',dpi=150,bbox_inches='tight');plt.close(fig)
print(json.dumps({'verdict':verdict,'rows':rows},indent=2))
