TITLE: GPTQ vs AWQ vs HQQ on Qwen3-8B: 400 questions were not enough
SUBTITLE: I quantized one 8B model three ways and measured the accuracy loss. The honest verdict is inconclusive, and the reason is sample size.

Every 4-bit quantization recipe promises roughly the same thing: a model a quarter of the size that loses almost nothing. I wanted to check how true that is, and how long each recipe takes to run. My preregistered claim was that all three recipes would stay within 1 accuracy point of BF16, and that HQQ would quantize fastest with no calibration data.

Verdict: INCONCLUSIVE. HQQ was the fastest, at 0.69 minutes. But the paired confidence intervals on accuracy are too wide to prove or disprove the 1-point claim for any recipe.

[IMAGE table.png]

Setup

Model: Qwen/Qwen3-8B. GPU: one NVIDIA L4 on Colab, about 54 minutes, roughly 1.39 CU. Each recipe used 4-bit weights in groups of 128, with the embedding and output head left at BF16. I used GPTQModel for GPTQ, LLM Compressor for AWQ and native HQQ. GPTQ and AWQ shared 128 calibration sequences of 512 tokens from WikiText-2 training data. HQQ used no calibration at all.

The test was 400 fixed questions picked with seed 44: 200 from MMLU and 200 from ARC-Challenge. The prompt is plain zero-shot, and the model's answer is the A/B/C/D letter with the highest next-token score. No chat template and no reasoning. Confidence intervals come from 10,000 paired bootstrap resamples, so every comparison uses the same questions.

Results

BF16 scored 79.25% combined (95% CI 75.25 to 83.00). GPTQ scored 77.50%, AWQ 77.50% and HQQ 76.75%.

The change versus BF16, in percentage points: GPTQ -1.75 (CI -4.25 to +0.75), AWQ -1.75 (CI -3.75 to +0.25), HQQ -2.50 (CI -5.25 to +0.25).

My rule needed each recipe's lower bound to sit above -1 point. None did. But none of the intervals sit wholly below -1 either, so I cannot call the recipes worse. They are stuck in the middle.

[IMAGE accuracy.png]

Speed and size

Quantization wall time: HQQ 0.69 minutes, GPTQ 5.24 minutes, AWQ 13.99 minutes. Each is a single run, so there is no interval on these. GPTQ's time includes compiling a CUDA extension on first use.

Saved size was about the same for all three: GPTQ 5.70 GiB, AWQ 5.69 GiB, HQQ 5.77 GiB, against 15.26 GiB for BF16. Perplexity on WikiText-2 moved from 12.88 (BF16) to 13.28 (GPTQ), 13.25 (AWQ) and 13.29 (HQQ).

Tokens per second tell you more about the software than about the recipe. BF16 ran at 15.1, GPTQ at 8.2, AWQ at 12.8 and HQQ at 2.4. AWQ's loader expanded the 4-bit checkpoint back into BF16 layers for evaluation, so its speed and memory are not real packed 4-bit inference. HQQ and GPTQ ran on plain PyTorch paths, not tuned kernels.

[IMAGE quantization_time.png]

Verdict

INCONCLUSIVE for the full hypothesis. HQQ was the fastest successful recipe. For accuracy, all three lost between 1.75 and 2.50 points on this sample, but the intervals are 4 to 5.5 points wide, so a true change anywhere from a loss of about 5 points to a small gain fits the data.

Caveats

Four hundred questions give weak power for a one-point claim. The bootstrap resamples questions, not subjects, so it ignores topic effects. This is a zero-shot, non-thinking test, so it says nothing about Qwen3's chat or reasoning quality. The recipes differ in symmetry and rounding defaults, and I ran them in a fixed order, not a random one. There were also two recovery steps: Transformers 5.17.0's HQQ loader failed and I used the native HQQ API with the same settings, and the AWQ and GPTQ checkpoints were evaluated in fresh processes after memory problems. Calibration and quantization were not repeated.

What to try next

The fix is more questions. I have queued a follow-up that uses the full MMLU test set plus ARC-Challenge, with a per-method non-inferiority test at a 1-point margin. That should shrink the intervals to well under a point.

All code and raw per-question results are in the experiment folder. If you want more measured tests like this, subscribe to Tech in General.
