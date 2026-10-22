"""Paired item bootstrap via multinomial sufficient statistics (exact same law)."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');F=Path('figures');F.mkdir(exist_ok=True)
methods=['bf16','gptq','awq','hqq']; names={'bf16':'BF16','gptq':'GPTQ','awq':'AWQ','hqq':'HQQ'}
def load(m):
 x=json.loads((R/f'{m}_predictions.json').read_text());return sorted(x,key=lambda v:v['item_id'])
preds={m:load(m) for m in methods}
ids=[x['item_id'] for x in preds['bf16']]
assert len(ids)==len(set(ids))==15214
assert ids==list(range(15214))
frozen=json.loads((R/'data.json').read_text())
cal=json.loads((R/'calibration_source.json').read_text())
assert frozen['calibration_ids']==cal['calibration_ids']
assert len(cal['calibration_ids'])==128 and all(len(x)==512 for x in cal['calibration_ids'])
assert sum(x['dataset']=='mmlu' for x in frozen['items'])==14042
assert sum(x['dataset']=='arc' for x in frozen['items'])==1172
assert all(2<=len(x['choices'])<=5 for x in frozen['items'])
assert len({x['subject'] for x in frozen['items'] if x['dataset']=='mmlu'})==57
for m in methods:
 assert [x['item_id'] for x in preds[m]]==ids
 assert [(x['dataset'],x['source_index'],x['gold']) for x in preds[m]]==[(x['dataset'],x['source_index'],x['gold']) for x in preds['bf16']]
 assert [x['prompt_tokens'] for x in preds[m]]==[x['prompt_tokens'] for x in preds['bf16']]
 assert all(np.isfinite(x['scores']).all() for x in preds[m])
 assert all(len(x['scores'])==len(frozen['items'][i]['choices']) for i,x in enumerate(preds[m]))
 checks=json.loads((R/f'{m}_batch_check.json').read_text())
 assert len(checks)==3 and all(x['single_prediction']==x['batch_prediction'] for x in checks)
 assert all((x['dataset'],x['source_index'],x['gold'])==(frozen['items'][i]['dataset'],frozen['items'][i]['source_index'],frozen['items'][i]['gold']) for i,x in enumerate(preds[m]))
 assert all(x['correct']==int(x['prediction']==x['gold']) and x['prediction']==int(np.argmax(x['scores'])) for x in preds[m])
metas={m:json.loads((R/f'{m}_done.json').read_text()) for m in methods}
for m in methods:
 assert abs(metas[m]['accuracy']-np.mean([x['correct'] for x in preds[m]]))<1e-12
 assert metas[m]['source_revision']=='b968826d9c46dd6066d109eabc6255188de91218'
summary={'bootstrap':{'resamples':10000,'seed':44,'unit':'paired item','implementation':'multinomial counts of paired differences; exactly equivalent distribution to iid item resampling'},'n':len(ids),'methods':{},'datasets':{}}
for ds in ['combined','mmlu','arc']:
 mask=np.array([ds=='combined' or x['dataset']==ds for x in preds['bf16']])
 b=np.array([x['correct'] for x in preds['bf16']])[mask];n=len(b)
 out={}
 for m in methods:
  q=np.array([x['correct'] for x in preds[m]])[mask];rng=np.random.default_rng(44)
  aci=np.percentile(rng.binomial(n,float(q.mean()),10000)/n*100,[2.5,97.5])
  row={'n':n,'correct':int(q.sum()),'accuracy_pct':float(q.mean()*100),'accuracy_ci':aci.tolist()}
  if m!='bf16':
   d=q-b;counts=np.array([(d==v).sum() for v in [-1,0,1]])
   draws=np.random.default_rng(44).multinomial(n,counts/n,size=10000)
   delta=(draws[:,2]-draws[:,0])*100/n;lo,hi=np.percentile(delta,[2.5,97.5])
   row.update(delta_pp=float(d.mean()*100),delta_ci=[float(lo),float(hi)],discordant=int((d!=0).sum()),gains=int((d==1).sum()),losses=int((d==-1).sum()),verdict='PROVEN' if lo>-1 else 'DISPROVEN' if hi<-1 else 'INCONCLUSIVE')
  out[m]=row
 summary['datasets'][ds]=out
for m in methods:
 summary['methods'][m]={**summary['datasets']['combined'][m],**{k:metas[m].get(k) for k in ['quant_seconds','model_bytes','tokens_per_second','eval_seconds','eval_peak_gib','quant_peak_gib','inference_linear_module_types','resident_parameter_bytes']}}
best=max(methods[1:],key=lambda m:(summary['methods'][m]['delta_ci'][0],-summary['methods'][m]['quant_seconds']))
summary['headline_method']=best
summary['hqq_fastest']=all(metas['hqq']['quant_seconds']<metas[m]['quant_seconds'] for m in ['gptq','awq'])
(R/'summary.json').write_text(json.dumps(summary,indent=2))
with (R/'summary.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['method','accuracy_pct','delta_pp','ci_lower','ci_upper','verdict','quant_minutes','size_gib','tokens_per_second']);writer.writeheader()
 for m in methods:
  d=summary['methods'][m];writer.writerow({'method':names[m],'accuracy_pct':d['accuracy_pct'],'delta_pp':d.get('delta_pp'),'ci_lower':d.get('delta_ci',[None,None])[0],'ci_upper':d.get('delta_ci',[None,None])[1],'verdict':d.get('verdict','reference'),'quant_minutes':None if m=='bf16' else d['quant_seconds']/60,'size_gib':d['model_bytes']/2**30,'tokens_per_second':d['tokens_per_second']})
plt.rcParams.update({'font.size':14,'axes.titlesize':16,'axes.labelsize':14,'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
colors=['#2563eb','#db7028','#199379']
fig,ax=plt.subplots(figsize=(9,4.8))
for j,m in enumerate(methods[1:]):
 d=summary['methods'][m];v=d['delta_pp'];lo,hi=d['delta_ci'];ax.errorbar(v,2-j,xerr=[[v-lo],[hi-v]],fmt='o',color=colors[j],capsize=6,markersize=9,lw=2)
 ax.annotate(f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]',xy=(hi,2-j),xytext=(10,9),textcoords='offset points',fontsize=13)
ax.axvline(-1,color='#b91c1c',ls='--',label='Non-inferiority margin (−1 point)');ax.axvline(0,color='gray',lw=1)
ax.set_yticks([2,1,0],['GPTQ','AWQ','HQQ']);ax.set_ylim(-.6,2.6);ax.set_ylabel('Recipe');ax.set_xlabel('Accuracy change versus BF16 (percentage points)');ax.set_title('Qwen3-8B: paired accuracy changes on 15,214 questions');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower left',fontsize=12)
xlo=min(summary['methods'][m]['delta_ci'][0] for m in methods[1:]);ax.set_xlim(min(xlo-.5,-1.5),max(.5,max(summary['methods'][m]['delta_ci'][1] for m in methods[1:])+.8));fig.tight_layout();fig.savefig(F/'accuracy.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12.5,5))
for ax,key,title,unit,ms in zip(axes,['quant_seconds','model_bytes','tokens_per_second'],['Quantization wall time','Saved model size','Generation throughput'],['Minutes','GiB','Tokens/second'],[methods[1:],methods,methods]):
 vals=[summary['methods'][m][key]/(60 if key=='quant_seconds' else 2**30 if key=='model_bytes' else 1) for m in ms]
 bars=ax.bar([names[m] for m in ms],vals,color=['#64748b' if m=='bf16' else colors[methods.index(m)-1] for m in ms])
 ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=12);ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel('Recipe');ax.set_ylim(0,max(vals)*1.22);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Four recipes, one L4; recorded library backends',fontsize=16);fig.text(.5,.015,'AWQ throughput uses weights reconstructed into BF16 Linear layers; backends are not equally optimized.',ha='center',fontsize=11);fig.tight_layout(rect=[0,.045,1,1]);fig.savefig(F/'resources.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.7));xx=np.arange(2);width=.19
for j,m in enumerate(methods):
 vals=[summary['datasets'][ds][m]['accuracy_pct'] for ds in ['mmlu','arc']];cis=[summary['datasets'][ds][m]['accuracy_ci'] for ds in ['mmlu','arc']]
 ax.bar(xx+(j-1.5)*width,vals,width,label=names[m],color='#64748b' if j==0 else colors[j-1],yerr=np.array([[v-c[0] for v,c in zip(vals,cis)],[c[1]-v for v,c in zip(vals,cis)]]),capsize=3)
ax.set_xticks(xx,['MMLU (14,042 items)','ARC-Challenge (1,172 items)']);ax.set_ylim(0,112);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel('Accuracy (%)');ax.set_xlabel('Test set');ax.set_title('Qwen3-8B accuracy by test set (95% bootstrap CIs)');ax.legend(ncol=4,loc='upper left',frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(F/'datasets.png',dpi=150);plt.close(fig)
rows=[]
for m in methods:
 d=summary['methods'][m];ci=d.get('delta_ci');rows.append([names[m],f"{d['accuracy_pct']:.2f}",'reference' if ci is None else f"{d['delta_pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",d.get('verdict','—'),'—' if m=='bf16' else f"{d['quant_seconds']/60:.2f}",f"{d['model_bytes']/2**30:.2f}",f"{d['tokens_per_second']:.1f}"])
fig,ax=plt.subplots(figsize=(14,3.3));ax.axis('off');t=ax.table(cellText=rows,colLabels=['Recipe','Accuracy %','Change pp [95% CI]','Verdict','Quantize min','Saved GiB','Tokens/s'],cellLoc='center',loc='center',colWidths=[.09,.11,.28,.13,.13,.12,.1]);t.auto_set_font_size(False);t.set_fontsize(14);t.scale(1,2)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('white')
 if r==0:cell.set_facecolor('#1e293b');cell.set_text_props(color='white',weight='bold')
 else:cell.set_facecolor('#eaf0f6' if r%2 else '#f6f8fb')
ax.set_title('Qwen3-8B · 15,214 paired questions · one-point non-inferiority margin',pad=14,fontsize=18);fig.text(.02,.04,'95% paired item bootstrap; 10,000 resamples, seed 44. AWQ evaluated reconstructed weights in BF16; throughput is backend-specific.',fontsize=12);fig.tight_layout();fig.savefig(F/'table.png',dpi=150,bbox_inches='tight');plt.close(fig)
audit={'n':len(ids),'methods':methods,'paired_ids_verified':True,'correctness_recomputed':True,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'data.json',R/'calibration_source.json']+[R/f'{m}_predictions.json' for m in methods]}}
audit.update(total_predictions=sum(len(v) for v in preds.values()),dataset_counts={'mmlu':14042,'arc':1172},mmlu_subjects=57,calibration_sequences=128,calibration_tokens_per_sequence=512,batch_checks_passed=True)
(R/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(summary,indent=2))
