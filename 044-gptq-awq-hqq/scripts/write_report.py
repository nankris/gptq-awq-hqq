from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
s=json.loads((R/'summary.json').read_text());resource=json.loads((R/'resource_accounting.json').read_text())
methods=['bf16','gptq','awq','hqq']; ok=[m for m in methods if s[m]['status']=='OK'];quant=[m for m in ok if m!='bf16']
fast=min(quant,key=lambda m:s[m]['quant_seconds']);f=s[fast];loss=-f['datasets']['combined']['delta_pp'];mins=f['quant_seconds']/60
headline=f'The fastest 4-bit recipe took {mins:.2f} minutes and lost {loss:.2f} points.'
def acc(m,d):
 x=s[m]['datasets'][d];a,b=x['accuracy_ci'];return f"{x['accuracy_pct']:.2f} [{a:.2f}, {b:.2f}]"
def delta(m):
 x=s[m]['datasets']['combined'];a,b=x['delta_ci'];return f"{x['delta_pp']:+.2f} [{a:+.2f}, {b:+.2f}]"
text=f'''# Three 4-bit recipes, one Qwen3-8B

**{headline}**

{fast.upper()} was the fastest successful recipe in this run; its measured accuracy change was {f['datasets']['combined']['delta_pp']:+.2f} percentage points across 400 fixed questions. The preregistered full-hypothesis verdict is **{s['verdict']}**; the confidence intervals below determine how much of that small observed change we can trust.

I quantized the same public Qwen/Qwen3-8B checkpoint on a single NVIDIA L4 using GPTQModel, LLM Compressor AWQ, and native HQQ, then compared them with BF16. Each recipe used 4-bit weights in groups of 128, with the embedding and output head left at BF16; GPTQ and AWQ shared 128 calibration sequences of 512 tokens from WikiText-2 training, while HQQ used no calibration examples.

**Plain-English takeaway:** HQQ shrank this model in under a minute without calibration, but answered ten fewer of our 400 questions correctly than BF16. This sample is too small to establish a one-point accuracy guarantee for any recipe.

The main table is also available as [an image for Substack](figures/table.png).

| Recipe | Quantize min | Saved GiB | Peak quantize GiB | Peak eval GiB | Tokens/s | WikiText-2 PPL | Combined accuracy % | Change vs BF16, pp |
|---|---:|---:|---:|---:|---:|---:|---|---|
'''
for m in methods:
 if m not in ok:
  text+=f'| {m.upper()} | FAILED | — | — | — | — | — | — | — |\n';continue
 x=s[m];q='—' if x['quant_seconds'] is None else f"{x['quant_seconds']/60:.2f}";v='—' if x['quant_peak_gib'] is None else f"{x['quant_peak_gib']:.2f}"
 text+=f"| {m.upper()} | {q} | {x['model_bytes']/2**30:.2f} | {v} | {x['eval_peak_gib']:.2f} | {x['tokens_per_second']:.1f} | {x['ppl']:.2f} | {acc(m,'combined')} | {delta(m) if m!='bf16' else 'reference'} |\n"
text+='''
**AWQ runtime caveat:** the 4-bit checkpoint was decompressed into BF16 linear layers for evaluation; its speed and evaluation memory are not packed 4-bit inference measurements.

Brackets show 95% bootstrap confidence intervals; change intervals use paired question outcomes. VRAM is PyTorch peak allocated memory, measured separately for quantization and evaluation, rather than total process memory reported by nvidia-smi. Saved size includes small tokenizer/config files for quantized recipes and only safetensors for BF16; the original 4-bit formats retain unquantized vocabulary matrices and metadata, so a fourfold file reduction is not implied.

| Recipe | MMLU accuracy %, n=200 [95% CI] | ARC-Challenge accuracy %, n=200 [95% CI] |
|---|---|---|
'''
for m in ok:text+=f"| {m.upper()} | {acc(m,'mmlu')} | {acc(m,'arc')} |\n"
text+='\n![Paired accuracy differences](figures/accuracy.png)\n\n'
text+='The preregistered accuracy condition required every quantized recipe’s paired 95% CI lower bound to be at least −1 point. '
for m in quant:text+=f"{m.upper()}'s change was {delta(m)} points. "
text+='A CI that crosses −1 makes the one-point claim inconclusive for that recipe; an interval wholly below −1 disproves it. The full hypothesis additionally requires HQQ to be strictly faster than both other recipes and use no calibration; one timing run per method supplies no timing confidence interval.\n\n'
text+='![Quantization wall time](figures/quantization_time.png)\n\n![Saved size and measured generation throughput](figures/resources.png)\n\n'
text+='''All 400 questions were selected before inference with seed 44: 200 from `cais/mmlu` (`all`, test) and 200 from `allenai/ai2_arc` (`ARC-Challenge`, test). The raw zero-shot prompt lists the question and choices and ends `Answer: `; accuracy selects the highest next-token logit among the literal answer letters, with no chat template or generated reasoning. It is an answer-label likelihood test. The 10,000 paired item bootstrap resamples use seed 44 and the identical question order for all methods; these are marginal intervals for this fixed sampling procedure, not claims about all tasks or all MMLU subjects.

Perplexity uses all 584 complete, nonoverlapping 512-token WikiText-2 raw test blocks: 298,424 predicted tokens, resetting context at every block and dropping the incomplete final block. Throughput uses batch 1, a fixed 32-token prompt, exactly 128 greedy output tokens, one warmup and three timed repetitions with CUDA synchronization. Quantization timing starts before method imports and cached model load and ends after quantization; it excludes package installation, model/data downloads, evaluation, debugging, and checkpoint serialization. Each recipe has one measured successful quantization, so timing differences are descriptive.

There was one compatibility failure: Transformers 5.17.0’s HQQ loader raised `NotImplementedError` before quantizing any weights. The repair used the native HQQ `AutoHQQHFModel` API with the same 4-bit/group-128/axis-1 settings; the original traceback is retained in `results/hqq_loader_failure.json`. No library installation failed, and this compatibility repair required under three minutes of work. HQQ's reported successful time excludes that failed attempt and repair. AWQ's first evaluation ran out of memory during lazy decompression; a fresh process successfully evaluated the existing checkpoint. GPTQ's quantization object retained offloaded meta tensors and could not move directly to CUDA, so it too was evaluated from its saved checkpoint in a fresh process. HQQ was evaluated in its native quantized object. These recovery attempts did not repeat calibration or quantization, and their original errors are retained. GPTQ used the default layer offload strategy, which explains its low GPU allocation during quantization; CPU RAM was not measured. Its timing includes first-use CUDA extension compilation.

The principal caveats are sample size, protocol, and implementation. Four hundred questions provide weak power for a one-point noninferiority claim, and subject/topic dependence is not modeled by the item bootstrap. This zero-shot, non-thinking test does not measure Qwen3’s chat or reasoning performance, and the short-context perplexity is not directly comparable with longer-context leaderboard numbers. Throughput compares the recorded library backends rather than equally tuned production engines: HQQ and GPTQ use PyTorch paths. **AWQ's compressed-tensors loader decompressed its packed 4-bit checkpoint into ordinary linear layers for inference**: resident parameter bytes were 16.54 billion and evaluation peak was 16.15 GiB. Thus AWQ's accuracy measures the quantized checkpoint's reconstructed weights, but its throughput and evaluation VRAM describe a decompressed runtime, not packed 4-bit inference. No optimized AWQ kernel was benchmarked; different kernels could change the speed ranking. The recipes also differ in symmetry and rounding/search defaults, and the run order was fixed rather than randomized.

Implementation references: [GPTQModel](https://github.com/ModelCloud/GPTQModel), [LLM Compressor AWQ recipe](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq), and [HQQ native implementation](https://github.com/dropbox/hqq). Exact versions, fixed inputs, per-question predictions, per-block losses, inference module classes, timing repetitions, and scripts are included with this experiment; the model revision is `b968826d9c46dd6066d109eabc6255188de91218`.

'''
text+=f"Resource use: NVIDIA L4; approximately {resource['estimated_session_cu']:.3f} CU, estimated from {resource['tracked_seconds']/60:.1f} tracked minutes × {resource['session_cu_per_hour']:.2f} CU/hour. The CLI exposes an account balance rather than a per-session bill; another experiment's session was active, so the account balance decrease is not attributed wholly to this run. Only session `b6-044` was used and it was stopped after collection.\n"
(ROOT/'writeup.md').write_text(text)
(R/'headline.json').write_text(json.dumps({'headline':headline,'fastest_method':fast,'quant_minutes':mins,'accuracy_loss_pp':loss,'verdict':s['verdict']},indent=2))
print(headline)
