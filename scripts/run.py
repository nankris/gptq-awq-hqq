"""Detached, incremental experiment. Run one method per process to release VRAM."""
import os, sys, time, json, math, random, traceback, pathlib, subprocess
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
ROOT=pathlib.Path('/content/exp044'); R=ROOT/'results'; R.mkdir(exist_ok=True)
MODEL='Qwen/Qwen3-8B'
REVISION='b968826d9c46dd6066d109eabc6255188de91218'
torch.set_num_threads(4)
def dump(path,obj):
    tmp=pathlib.Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)
def prepare():
    if (R/'data.json').exists(): return json.loads((R/'data.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    w=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
    train=tok('\n\n'.join(w['train']['text']),add_special_tokens=False)['input_ids']
    test=tok('\n\n'.join(w['test']['text']),add_special_tokens=False)['input_ids']
    rng=random.Random(44)
    starts=[rng.randrange(len(train)-512) for _ in range(128)]
    m=load_dataset('cais/mmlu','all',split='test')
    a=load_dataset('allenai/ai2_arc','ARC-Challenge',split='test')
    items=[]
    for ds,name in [(m,'mmlu'),(a,'arc')]:
        ids=sorted(random.Random(44).sample(range(len(ds)),200))
        for i in ids:
            row=ds[i]
            if name=='mmlu':
                choices=row['choices']; gold=int(row['answer']); q=row['question']
            else:
                choices=row['choices']['text']; gold=row['choices']['label'].index(row['answerKey']); q=row['question']
            items.append({'dataset':name,'source_index':i,'question':q,'choices':choices,'gold':gold})
    d={'items':items,'calibration_ids':[train[s:s+512] for s in starts],'calibration_starts':starts,'test_ids':test,'dataset_fingerprints':{'mmlu':m._fingerprint,'arc':a._fingerprint,'wikitext_test':w['test']._fingerprint},'seed':44}
    dump(R/'data.json',d); tok.save_pretrained(ROOT/'tokenizer'); return d

def main(method):
    if (R/f'{method}_done.json').exists(): print('SKIP completed',method); return
    data=prepare()
    local=snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
    tok=AutoTokenizer.from_pretrained(local)
    meta_path=R/f'{method}_meta.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {'method':method,'model':MODEL,'source_revision':str(local).split('/')[-1],'calibration_sequences':0 if method in ['bf16','hqq'] else 128,'calibration_tokens_per_sequence':0 if method in ['bf16','hqq'] else 512}
    out=ROOT/'models'/method
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    if method=='bf16':
        model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        meta.update(quant_seconds=None,load_seconds=time.perf_counter()-start,quant_peak_gib=None,model_bytes=sum(p.stat().st_size for p in pathlib.Path(local).glob('*.safetensors')))
    elif method=='hqq':
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.core.quantize import BaseQuantizeConfig
        if (out/'qmodel.pt').exists() and meta.get('quant_seconds'):
            model=AutoHQQHFModel.from_quantized(str(out),compute_dtype=torch.bfloat16,device='cuda',attn_implementation='sdpa')
        else:
            model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
            AutoHQQHFModel.setup_model(model)
            qcfg=BaseQuantizeConfig(nbits=4,group_size=128,axis=1)
            configs={k:qcfg for k in model.linear_tags if 'lm_head' not in k}
            print('HQQ tags',model.linear_tags,'quantized tags',list(configs),flush=True)
            AutoHQQHFModel.quantize_model(model,quant_config=configs,compute_dtype=torch.bfloat16,device='cuda')
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start; meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            meta['loader']='native hqq AutoHQQHFModel (Transformers 5 HqqConfig loader unsupported)'
            dump(meta_path,meta)
            out.mkdir(parents=True,exist_ok=True); AutoHQQHFModel.save_quantized(model,str(out)); tok.save_pretrained(out)
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    elif method=='gptq':
        from gptqmodel import GPTQModel, GPTQConfig, BACKEND
        if (out/'config.json').exists() and meta.get('quant_seconds'):
            wrapper=GPTQModel.load(str(out),device='cuda',backend=BACKEND.GPTQ_TORCH,dtype=torch.bfloat16)
        else:
            wrapper=GPTQModel.load(local,GPTQConfig(bits=4,group_size=128,desc_act=False,sym=True),dtype=torch.bfloat16)
            calib=[{'input_ids':x,'attention_mask':[1]*len(x)} for x in data['calibration_ids']]
            wrapper.quantize(calib,batch_size=1)
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start;meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            dump(meta_path,meta); wrapper.save(str(out)); tok.save_pretrained(out)
            # Offloaded quantization leaves meta tensors; reload the saved checkpoint for inference.
            del wrapper
            import gc;gc.collect();torch.cuda.empty_cache()
            wrapper=GPTQModel.load(str(out),device='cuda',backend=BACKEND.GPTQ_TORCH,dtype=torch.bfloat16)
        model=wrapper.model
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    elif method=='awq':
        from llmcompressor import oneshot
        from llmcompressor.modifiers.transform import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        if (out/'config.json').exists() and meta.get('quant_seconds'):
            model=AutoModelForCausalLM.from_pretrained(out,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        else:
            model=AutoModelForCausalLM.from_pretrained(local,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
            ds=Dataset.from_dict({'input_ids':data['calibration_ids'],'attention_mask':[[1]*512 for _ in range(128)]})
            oneshot(model=model,dataset=ds,recipe=[AWQModifier(),QuantizationModifier(ignore=['lm_head'],scheme='W4A16_ASYM',targets=['Linear'])],max_seq_length=512,num_calibration_samples=128)
            torch.cuda.synchronize(); meta['quant_seconds']=time.perf_counter()-start;meta['quant_peak_gib']=torch.cuda.max_memory_allocated()/2**30
            dump(meta_path,meta)
            model.save_pretrained(out,save_compressed=True);tok.save_pretrained(out)
            # Reload the packed saved model, rather than evaluate fake-quant weights.
            del model
            import gc;gc.collect();torch.cuda.empty_cache()
            model=AutoModelForCausalLM.from_pretrained(out,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
        meta['model_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    model.eval(); meta['backend_modules']=sorted(set(type(m).__module__+'.'+type(m).__name__ for m in model.modules() if any(k in type(m).__name__.lower() for k in ['quant','hqq','compressed'])))
    dump(meta_path,meta)
    if os.environ.get('EXP044_QUANT_ONLY')=='1':
        print('QUANTIZATION_CHECKPOINT_COMPLETE',method,flush=True);return
    torch.cuda.reset_peak_memory_stats(); eval_start=time.perf_counter()
    predpath=R/f'{method}_predictions.json'; preds=json.loads(predpath.read_text()) if predpath.exists() else []
    # Fixed raw zero-shot prompt. Letter tokens are scored conditionally after an explicit space.
    label_ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCDE']
    assert all(len(x)==1 for x in label_ids)
    with torch.inference_mode():
        for i,item in enumerate(data['items']):
            if i<len(preds):continue
            prompt='Choose the correct answer. Reply with only the letter.\n\n'+item['question']+'\n'+'\n'.join(f'{chr(65+j)}. {c}' for j,c in enumerate(item['choices']))+'\nAnswer: '
            ids=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
            logits=model(**ids,use_cache=False).logits[0,-1].float()
            scores=[float(logits[x[0]]) for x in label_ids[:len(item['choices'])]]
            pred=max(range(len(scores)),key=lambda j:scores[j])
            preds.append({'dataset':item['dataset'],'source_index':item['source_index'],'gold':item['gold'],'prediction':pred,'correct':int(pred==item['gold']),'scores':scores,'prompt_tokens':ids.input_ids.shape[1]})
            dump(predpath,preds)
            if (i+1)%25==0: print(method,'ACCURACY',i+1,'elapsed',round(time.perf_counter()-eval_start,1),flush=True)
        pplpath=R/f'{method}_ppl.json'; blocks=json.loads(pplpath.read_text()) if pplpath.exists() else []
        test=data['test_ids']; n=len(test)//512
        for j in range(n):
            if j<len(blocks):continue
            ids=torch.tensor([test[j*512:(j+1)*512]],device='cuda')
            loss=model(input_ids=ids,labels=ids,use_cache=False).loss.item()
            blocks.append({'block':j,'nll':loss,'scored_tokens':511});dump(pplpath,blocks)
            if (j+1)%50==0: print(method,'PPL',j+1,'/',n,flush=True)
        # Exactly 32 input tokens; fixed prompt repeated if needed, then sliced.
        ids=tok('Explain how computers represent numbers. '*10,return_tensors='pt').input_ids[:,:32].to('cuda')
        genkw={'max_new_tokens':128,'min_new_tokens':128,'do_sample':False,'pad_token_id':tok.eos_token_id,'use_cache':True}
        model.generate(ids,**genkw);torch.cuda.synchronize(); times=[]
        for _ in range(3):
            t=time.perf_counter(); result=model.generate(ids,**genkw);torch.cuda.synchronize(); times.append(time.perf_counter()-t)
        meta.update(tokens_per_second=128/(sum(times)/len(times)),generation_seconds=times,eval_peak_gib=torch.cuda.max_memory_allocated()/2**30,ppl=math.exp(sum(x['nll'] for x in blocks)/len(blocks)),ppl_blocks=n,ppl_scored_tokens=n*511,eval_seconds=time.perf_counter()-eval_start,accuracy=sum(x['correct'] for x in preds)/len(preds),finished_time=time.time())
    meta['resident_parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
    meta['inference_linear_module_types']=sorted(set(type(m).__module__+'.'+type(m).__name__ for m in model.modules() if 'linear' in type(m).__name__.lower()))
    dump(meta_path,meta);dump(R/f'{method}_done.json',meta)
    print('DONE',method,json.dumps(meta),flush=True)
if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare': prepare()
        else: main(sys.argv[1])
    except Exception:
        err=traceback.format_exc();print(err,flush=True);dump(R/f'{sys.argv[1]}_error.json',{'time':time.time(),'error':err});raise
