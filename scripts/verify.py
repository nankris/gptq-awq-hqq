"""Audit saved records against frozen inputs without loading a model."""
from pathlib import Path
import json, hashlib, math
r=Path(__file__).resolve().parents[1]/'results'
load=lambda name: json.loads((r/name).read_text())
d=load('data.json'); manifest=load('input_manifest.json'); summary=load('summary.json')
for key,expected in manifest['hashes'].items():
    actual=hashlib.sha256(json.dumps(d[key],sort_keys=True).encode()).hexdigest()
    assert actual==expected,(key,actual,expected)
assert len(d['items'])==400 and len(d['calibration_ids'])==128
assert all(len(x)==512 for x in d['calibration_ids'])
results={}
for m in ['bf16','gptq','awq','hqq']:
    meta=load(m+'_done.json');p=load(m+'_predictions.json');b=load(m+'_ppl.json')
    assert len(p)==400 and len(b)==584
    for item,rec in zip(d['items'],p):
        assert all(item[k]==rec[k] for k in ['dataset','source_index','gold'])
        assert rec['prediction']==max(range(len(rec['scores'])),key=lambda i:rec['scores'][i])
        assert rec['correct']==int(rec['prediction']==rec['gold'])
    assert [x['block'] for x in b]==list(range(584))
    ppl=math.exp(sum(x['nll'] for x in b)/len(b))
    accuracy=sum(x['correct'] for x in p)/len(p)
    assert math.isclose(ppl,meta['ppl'],rel_tol=1e-12)
    assert math.isclose(accuracy,meta['accuracy'])
    assert math.isclose(128/(sum(meta['generation_seconds'])/3),meta['tokens_per_second'])
    results[m]={'predictions':len(p),'perplexity_blocks':len(b),'correct':sum(x['correct'] for x in p),'ppl_recomputed':ppl,'passed':True}
(r/'audit.json').write_text(json.dumps({'input_hashes_verified':True,'methods':results},indent=2))
print('PASS: frozen hashes, 1600 item records, 2336 loss blocks, accuracies and throughput.')
