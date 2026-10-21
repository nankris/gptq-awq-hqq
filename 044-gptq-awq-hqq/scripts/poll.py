import subprocess,pathlib,tarfile,time,json,re,shutil
root=pathlib.Path('/content/exp044');r=root/'results'
print((r/'driver.log').read_text()[-700:])
for m in ['bf16','hqq','awq','gptq']:
 p=r/('hqq_native.log' if m=='hqq' and (r/'hqq_native.log').exists() else ('awq_eval_retry.log' if m=='awq' and (r/'awq_eval_retry.log').exists() else ('gptq_eval_retry.log' if m=='gptq' and (r/'gptq_eval_retry.log').exists() else f'{m}.log')))
 if not p.exists(): continue
 counts={}
 for suffix in ['predictions','ppl']:
  f=r/f'{m}_{suffix}.json'
  if f.exists():counts[suffix]=len(json.loads(f.read_text()))
 text=p.read_text();matches=re.findall(r'Grid search for ([\w.]+)',text)
 if m=='gptq': matches=re.findall(r'MemDiag: layer=(\d+)',text)
 print(m,counts,'done=',(r/f'{m}_done.json').exists(),'layer=',matches[-1:] if matches else '', 'tail:',text[-250:] if not (r/f'{m}_done.json').exists() else 'complete')
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
print(subprocess.getoutput("ps -eo pid,etime,pcpu,args | grep -E 'python.*(driver|run).py'"))
for m in ['hqq','awq','gptq']:
 for p in (root/'models'/m).glob('*.json'):
  if 'tokenizer' in p.name or p.name=='vocab.json':continue
  dest=r/'model_configs'/m;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/p.name)
with tarfile.open(root/'checkpoint.tar.gz','w:gz') as t:t.add(r,arcname='results')
print('CHECKPOINT',time.time())
