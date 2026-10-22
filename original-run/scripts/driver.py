"""Each quantization and evaluation gets its own process, avoiding retained library state."""
import subprocess,time,pathlib,json,os
root=pathlib.Path('/content/original-run');os.chdir(root)
for method in ['bf16','hqq','awq','gptq']:
    if (root/'results'/f'{method}_done.json').exists(): continue
    meta=root/'results'/f'{method}_meta.json'
    checkpoint=meta.exists() and json.loads(meta.read_text()).get('quant_seconds') and (root/'models'/method/'config.json').exists()
    stages=['eval'] if method=='bf16' or checkpoint else ['quantize','eval']
    for stage in stages:
        print('START',method,stage,time.ctime(),flush=True)
        env=dict(os.environ)
        if stage=='quantize':env['EXP044_QUANT_ONLY']='1'
        else:env.pop('EXP044_QUANT_ONLY',None)
        with open(root/'results'/f'{method}_{stage}.log','a') as log:
            p=subprocess.run(['python','-u','run.py',method],stdout=log,stderr=subprocess.STDOUT,env=env)
        print('EXIT',method,stage,p.returncode,time.ctime(),flush=True)
        if p.returncode:break
(root/'results'/'driver_done.json').write_text(json.dumps({'finished':time.time()}))
