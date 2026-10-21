"""Save each preregistered comparison as soon as that full method completes."""
from pathlib import Path
import json
import numpy as np
r=Path('results')
if not (r/'bf16_done.json').exists():raise SystemExit()
b=sorted(json.loads((r/'bf16_predictions.json').read_text()),key=lambda x:x['item_id']);assert [x['item_id'] for x in b]==list(range(15214))
for m in ['gptq','awq','hqq']:
 out=r/f'{m}_paired.json'
 if out.exists() or not (r/f'{m}_predictions.json').exists():continue
 q=sorted(json.loads((r/f'{m}_predictions.json').read_text()),key=lambda x:x['item_id'])
 if len(q)!=15214:continue
 assert [x['item_id'] for x in q]==list(range(15214))
 assert [(x['dataset'],x['source_index'],x['gold']) for x in q]==[(x['dataset'],x['source_index'],x['gold']) for x in b]
 d=np.array([x['correct']-y['correct'] for x,y in zip(q,b)]);counts=np.array([(d==v).sum() for v in [-1,0,1]])
 draws=np.random.default_rng(44).multinomial(len(d),counts/len(d),size=10000);lo,hi=np.percentile((draws[:,2]-draws[:,0])*100/len(d),[2.5,97.5])
 result={'method':m,'n':len(d),'delta_pp':float(d.mean()*100),'delta_ci':[float(lo),float(hi)],'verdict':'PROVEN' if lo>-1 else 'DISPROVEN' if hi<-1 else 'INCONCLUSIVE','gains':int(counts[2]),'losses':int(counts[0]),'resamples':10000,'seed':44}
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
