"""Record collection accounting and assemble verifiable excerpts; rerunnable offline."""
from pathlib import Path
import json, datetime
root=Path(__file__).resolve().parents[1];r=root/'results'
start=json.loads((r/'start.json').read_text())
# CLI termination recorded at 10:53:59 UTC, 2026-10-17.
stop=1792234439
seconds=stop-start['start_time']
resource={'gpu':'NVIDIA L4','session':'b6-044','start_time':start['start_time'],'stop_time':stop,'tracked_seconds':seconds,'session_cu_per_hour':1.54,'estimated_session_cu':seconds/3600*1.54,'initial_account_balance_cu':152.82,'final_account_balance_cu':150.02,'account_delta_cu':2.80,'cu_basis':'Estimated from tracked wall time at 1.54 CU/h; CLI rate was 3.08 CU/h for two L4 sessions before stop and 1.54 CU/h after. Account delta includes another session and is not this experiment cost.','stopped':True}
(r/'resource_accounting.json').write_text(json.dumps(resource,indent=2))
lines=[start['gpu'],'\nORIGINAL DRIVER TIMELINE\n',(r/'driver.log').read_text(),'\nRECOVERY EVALUATIONS AND FINAL METRICS\n']
for method,filename in [('bf16','bf16.log'),('hqq','hqq_native.log'),('awq','awq_eval_retry.log'),('gptq','gptq_eval_retry.log')]:
    lines.append('\n'+method.upper()+'\n')
    for line in (r/filename).read_text().splitlines():
        if any(s in line for s in ['DONE ', 'ACCURACY 400','PPL 550','HQQ tags']):lines.append(line+'\n')
lines.extend(['\nGPU SAMPLES (name, memory MiB, utilization %)\n',(r/'gpu_samples.jsonl').read_text(),'\nCOLLECTION AND TERMINATION\n','All four done.json files and 400 predictions / 584 loss blocks each copied locally.\n','[colab] Stopping session \'b6-044\'...\n[colab] Session terminated.\nCurrent balance: 150.02 compute units\nUsage rate: 1.54/hr\nActive assignments: 1\n',json.dumps(resource,indent=2)+'\n'])
(root/'figures'/'colab_run.txt').write_text(''.join(lines))
print(json.dumps(resource,indent=2))
