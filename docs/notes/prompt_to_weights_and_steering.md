# Prompt-to-weights, activation steering, rank-one editing: literature notes

Written 2026-11-15 (KST) by a survey agent for the `Flexible_WSP` PoC. It does not cover the Nov-2026 "Weight-Space Programming" follow-ups (see `wsp_ecosystem.md`, written by another agent).

## 0. How to read this file

**The project's claim, as given to me.** An instruction such as "Respond only with a JSON object" changes the next-token distribution at only a few answer positions (per-position KL, teacher-forced). At such a position (the answer-start position = last prompt token), take the mean hidden-state difference `d[l]` per layer and write it into the weights as `W' = W + u v^T` on chosen MLP output matrices. `u` is the per-layer increment of `d`. `v` is a ridge-regression read vector on MLP hidden activations, ~1 at target positions and ~0 elsewhere. No training, no hook. On Qwen3.5-0.8B, 6 edits gave valid JSON on 27/30 instruction-free questions; all-position steering at one layer gave 6/30. "Be brief" did not transfer. Proposed predictor: `||mean diff||^2 / mean ||diff||^2`.

**How papers were found.** WebSearch, OpenAlex (`cites:` lists of Transmuting, Dherin et al., Prompt Baking, Stolfo et al., plus keyword search), arXiv listing search, and reference lists of the papers read. Every paper below was fetched at least once (arXiv abs page, arXiv HTML full text, ACL Anthology page, or GitHub). Nothing is cited from memory alone.

**Reading depth.** Each entry says what was read:

- `FULL`: whole main text, plus the appendices named.
- `PARTS`: the sections named. Other sections were not read.
- `ABS`: abstract only.
- `META`: title, authors and date only.

Numbers come from the text that was read. Where a number sits only in a figure I say so.

**Some statements below are my own derivations or inferences, not claims from any paper.** They are marked `[my derivation]`. They are short algebra and have not been checked numerically against the PoC code.

**Papers found by this survey that are not in `docs/plan_related_work.md` and matter most:**

| Paper | Why it matters |
| --- | --- |
| SteerEdit (arXiv 2604.12359, ACL 2026) | Same edit algebra as the project: closed-form, gated, rank-1 write of a steering vector into `W_down`. |
| Prefix Steering (arXiv 2610.04967) | Steering only at the final prompt token often suffices, including for answer formatting. |
| Steer Like the LLM (arXiv 2605.03907, ICML 2026) | Shows a prompt acts strongly on some positions and barely on others; gated rank-1 steering reaches 55–72% on IFEval format without instruction text. |
| Concept-ROT (arXiv 2412.13341, ICLR 2025) | ROME with a non-lexical key collected at the end-of-turn token, installing behaviors. |
| PatchKV (arXiv 2609.39329, NeurIPS 2026) | Training-free ridge patch of the MLP output matrix from context-induced activation differences. |
| Braun et al. (arXiv 2505.22637) | "Directional agreement" of per-sample differences predicts steerability. Close to the proposed predictor. |
| Dynamic Activation Composition (arXiv 2406.17563) | Per-step KL shows steering is needed mostly on the first few generated tokens. |

---

## A. Training-free prompt / instruction / steering vector → weights

### A1. Transmuting prompts into weights
- Mazzawi, Dherin, Munn, Goldwaser, Wunder, Gonzalvo (Google Research). arXiv 2510.08734. v1 2025-10-09, v2 2026-02-05, v3 2026-06-03.
- Read: `FULL` v3 (main text, App. A, B, C, D proof, F related work; App. E skimmed). `PARTS` v2 (§3, §4, §5, App. B). v1 HTML returned 404, so I do not know what v1 contained.
- **Setting.** Prompt `C = [I, x_1..x_n]`. Dherin et al. show the effect of chunk `I` on token `x` equals a token-dependent patch: bias `δ_x(I) = A(C,x) − A(C\I,x)` (difference of attention-block outputs with and without `I`) and matrix `Δ_x(I) = W δ_x a_x^T / ||a_x||^2` on the MLP **input** matrix `W`, where `a_x` is the attention output without `I`.
- **Method ("thought patch").** Replace token-dependent patches by one token-independent pair.
  - Thought vector = mean of `δ_i`. The paper presents this as the justification for difference-of-means steering vectors.
  - Thought matrix = `argmin_M Σ_i ||M a_i − W δ_i||^2`, closed form `(Σ W δ_i a_i^T) Z^{-1}` with `Z = Σ a_i a_i^T` (Theorem 2.1). Ridge variant adds `ρ||M||_F^2`.
- **Which positions.** v2 §3.4 states it outright: "The index i in the sum above runs over all tokens of all completions [I, ...] of I in our collection." App. C (v3): the model's answer is included in both passes, so the two passes process the same token sequence (teacher-forced). So the least squares runs over **all token positions** of user turn plus answer, with no position weighting or selection.
- **Algorithm 1.** For each batch and each layer `l = 1..L`: run the instructed pass with the original model; run the uninstructed pass with the currently patched weights (re-run per layer); `dz_l = A_l − a_l`; `db_l = mean(dz_l)`; `dW_l = RidgeSolver(a_l → W_l dz_l)`; apply with learning rate `η`.
- **Gemma adaptation (App. A).**
  - The thought matrix is solved twice, for `W_up` and `W_gate`, on the RMSNorm-ed MLP input.
  - Gemma has no MLP output bias. App. A.2 absorbs the thought vector into `W_down`: target `d_goal = δ(I) ⊘ s_l + d^C_l − d_l`, rescaled to the norm of the instructed activation, then `dW_down = argmin_M ||M h_l − d̂_goal||^2 + ρ||M||^2` over MLP hidden activations `h_l`.
  - The main text says the `W_down` absorption was used "for this task" (the dictionary task). How the bias was realised on Gemma for the other tasks is not stated clearly in what I read.
- **Model.** Gemma 3 1B IT only.
- **Numbers (original model with instruction → patched model without instruction).**

  | Task | With instruction | Patched, no instruction | Setup |
  | --- | --- | --- | --- |
  | Multiply 3 single-digit numbers | 100 ± 0.0 | 100 ± 0.0 | 10 completions, η=0.1, batch 1, ρ=0, 10 steps, 20 held-out, 5 seeds |
  | Sum 3 single-digit numbers | 100 ± 0.0 | 100 ± 0.0 | same |
  | ParaDetox toxicity % (lower is better) | 6.7 ± 0.7 | 6.22 ± 1.6 | 10 examples, 1 step, η=1; toxic input 90.53, neutral reference 5.62 |
  | ParaDetox BERT score | 68.09 ± 0.51 | 62.16 ± 0.25 | same |
  | French→English word (BERT similarity %) | 97.37 | 92.74 | 10 completions |
  | Spanish→English word | 98.4 | 94.99 | 10 completions |
  | Country→Capital (exact match) | 79.0 | 47.6 | 5 demonstrations |
  | Capital→Country (**v2 only**) | 80.0 | 67.2 | 5 demonstrations |
  | List first (**v2 only**) | 83.25 | 31.67 | 5 demonstrations |
  | Dictionary of k random key-value pairs, same query templates, k = 1/2/4/8/16/32 | 100 each | 100/100/100/99/98/91 | ρ=0.5, η=1, one step; rank reduced to 40 principal components for k ≥ 16 |
  | Same, different query templates | — | 100/100/83.3/79.1/53.0/42.7 | same |

  `docs/plan_related_work.md` quotes "31.67 vs 83.25" for list-first. That row exists in v2 Table 3. **v3 Table 3 no longer has it** (nor Capital→Country), although v3 App. B.3 still describes a "first letter of a string" task.
- **Stated limitations.** Overfits to the surface form of the completion templates (the different-query table). Regularization needs tuning. "Future work will focus on advanced regularization."
- **Not in the paper.** No format instruction. No analysis of which positions matter. No count of how many edits are needed. No comparison with activation steering. One model.
- **Relation to the project's edit** `[my derivation]`. In the `W_down` least squares above, set the target to `u` at the chosen positions and `0` at all other positions. The ridge solution is then `M = u (Σ_{i∈targets} h_i)^T (H H^T + ρI)^{-1} = u v^T`, with `v` exactly the project's ridge read vector. So the project's edit is this paper's `W_down` absorption with a **position-masked constant target**. What the project adds is the choice of mask (justified by the per-position KL measurement), the rank-1 form, and the count.

### A2. Learning without training: the implicit dynamics of in-context learning
- Dherin, Munn, Mazzawi, Wunder, Gonzalvo. arXiv 2507.16003. First 2025-07-21, v4 2026-06-02.
- Read: `PARTS` (abstract, §2.1, §3 complete, start of §4).
- **Result.** For a block "contextual layer `A` followed by an MLP with first weight `W`", the effect of context chunk `Y` on query token `x` is reproduced exactly (pre-activation matching) by the rank-1 update `Δ_x W(Y) = (W δA_x(Y)) A(C\Y,x)^T / ||A(C\Y,x)||^2`. It is the minimum-Frobenius-norm solution and leaves `W` unchanged on directions orthogonal to `A(C\Y,x)` (Theorem 2.3).
- **Static patch (§3.4, v4).** Least-squares "thought patch" over a calibration set of K queries. Tested only on a single-layer transformer trained on in-context linear regression, K ∈ {1,2,5,10,25,50,100}, 100 held-out queries; error falls as K grows (figure only).
- Other experiments: 10-layer pre-LN transformer on linear regression, patched and contextual passes agree to ~1e-6.
- **Relevance.** The per-token patch is a key→value rank-1 edit with key = the token's own context-free attention output. No LLM experiments here.

### A3. Equivalence of Context and Parameter Updates in Modern Transformer Blocks
- Goldwaser, Munn, Gonzalvo, Dherin. arXiv 2511.17864. First 2025-11-22, v3 2026-07-06.
- Read: `PARTS` (abstract, §4, §6).
- **Result.** Exact per-token rank-1 patches for Gemma-style blocks (`W_gate`, `W_up`, `W_down`, RMSNorm scale), multi-layer algorithm, general "input/output controllability" framework.
- **Experiments.** Gemma 3 1B and 4B, Falcon 7B. The patch is **recomputed for every generated token**. float32: token match is exact. bfloat16: 87.5% token agreement with the naive update, 100% with the numerically stable variant.
- **Stated limitations.** "The derived updates are token-dependent and must be recomputed at each step"; naive reuse in free generation "is disrupted by the attention values of newly generated tokens". Their desiderata note that "Matrix updates are input-conditional, altering outputs only along specific directional vectors".

### A4. A Simple Generalisation of the Implicit Dynamics of In-Context Learning
- Innocenti, Achour. arXiv 2512.11255, 2025-12-12. Read: `ABS`.
- Extends Dherin et al. to all sequence positions, any block, and blocks with layer norm; verified on in-context linear regression; studies how implicit updates for different tokens relate.

### A5. Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers
- Chen, Hu, Jin, Lee, Kawaguchi. arXiv 2406.02847, ICML 2024. Read: `ABS`.
- For linearized attention, ICL demonstrations convert exactly into added bias terms (ICLCA); approximate version tried on GPT-2. Needs an architectural change (linearized attention).

### A6. PatchKV: Weight-Space Compensation of KV Cache
- Lee, Lee, Choi, Kim, Hong. arXiv 2609.39329, first 2026-09-30, v4 2026-11-12. NeurIPS 2026. Code: github.com/cusasak/PatchKV.
- Read: `PARTS` (abstract, §2.1, §2.2 up to the ridge derivation, §3.1, §3.2).
- **Method.** A long context is prefilled; its KV cache is compressed (KVzip or Attention Matching). For reference query tokens (teacher-forced; "repeat-context" and "self-study" prompts) collect activations under the full cache and under the compressed cache. For each block solve `W h_i + z_i = (W + ΔW) h'_i + z'_i` for `ΔW` on the **MLP output projection** by ridge regression in closed form, sequentially over blocks (upstream blocks already patched). Merge `ΔW` into the weights. Cites the Dherin / Goldwaser / Mazzawi line as the source of the idea.
- **Models.** Qwen2.5-7B-1M, Qwen3-4B, LLaMA3.1-8B. Benchmarks: SCBench (up to 170K tokens), SQuAD, GSM8K, NIAH.
- **Number.** On SQuAD, "recovers up to 73% of full-KV accuracy with no cache", where the cache-only baseline is near zero. Other results are in figures.
- **Relevance.** Same matrix (`W_out`) and same solver (ridge on MLP hidden activations) as the project. Differences: full-rank patch, all reference-token positions, per-document content rather than an instruction, no position gating.

### A7. SteerEdit — Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
- Yin, Han, Xu, Li, He, Zhou, Wang, Fu, Du, Li, Ji. arXiv 2604.12359, 2026-04-14. ACL 2026 main (Anthology id 2026.acl-long.1206, title verified on the Anthology page).
- Read: `PARTS` (abstract, §1, §2, §3, §4, §5 complete, §6.1, §6.2 text, §6.4, App. C.1). Result tables did not parse; only the numbers below were read.
- **Method.**
  1. Steering direction `z_comp` = normalized difference of mean hidden states on benign vs harmful prompts.
  2. Keys are MLP hidden activations `k = σ(W_up h)`. `K_0` = keys from clean prompts (10,000 Dolly prompts plus 256 AdvBench prompts, no trigger). `K` = keys from benign prompts with the trigger token appended. Both are taken "from the last token position".
  3. Projector `P = Q Q^T` onto the left null space of `K_0` (the smallest `p` fraction of singular directions is treated as null; `p = 0.5`).
  4. Solve `min ||Δ̃ P K − α Z||^2 + λ||Δ̃ P||^2`, where every column of `Z` is `z_comp`. Closed form `Δ̃* = α Z K^T P^T (P K K^T P^T + λ P P^T)^†`.
  5. `W_down' = W_down + Δ̃* P`, independently for layers 5–20. Defaults `λ = 5.0`, `α = 0.8`, trigger token "cf".
- `[my derivation]` Because all columns of `Z` are the same vector, the update is rank-1: `(α z_comp) · v^T` with `v = P (P K K^T P + λP)^† P K 1`. That is a steering vector times a read vector fitted to be ~1 on target keys and exactly 0 on the clean-key subspace. Same shape as the project's `u v^T`.
- **Models.** Llama-2-7b-chat, Llama-3-8B-Instruct, Qwen2.5-7B-Instruct. Baselines: ROME, JailbreakEdit, DualEdit, SFT.
- **Numbers read.** Removing the null-space constraint raises attack success without the trigger from 2.3% to 63.7% and drops utility retention from 98.6% to 61.2%. Works with as few as 16 proxy samples, diminishing returns beyond 64. Utility is nearly unchanged for `α ≤ 0.8`.
- **Differences from the project.**
  - Gate is a lexical trigger token in the context. The project's gate is a position that exists in every prompt.
  - Behavior is refusal → compliance. No instruction, no format.
  - Dormancy is enforced only against last-token keys of clean prompts, not against other positions of the same prompt.
  - Hard null-space projection instead of ridge zeros.
  - One normalized direction with a global `α`, not per-layer increments.
  - No analysis of where in the sequence the push is needed.
- **Unresolved.** I could not tell whether "last token position" means the trigger token itself or the last chat-template token after it. Figure references in the HTML are broken, and I found no code link. If it is the last template token, the key is "answer-start position AND trigger in context", which is closer still to the project.

### A8. Steer2Edit: From Activation Steering to Component-Level Editing
- Sun, Yan, Wang, Weng (UCSD). arXiv 2602.09870, first 2026-02-10, v4 2026-11-08. NeurIPS 2026. Code: github.com/Trustworthy-ML-Lab/Steer2Edit.
- Read: `PARTS` (abstract, §1 part, §3.1, §3.2, start of §3.3, §3.5, §4 setup).
- **Method.** For each component `W_i` writing to the residual stream (attention-head output projections; individual MLP down-projection neurons), rank-1 update `ΔW_i = sign(g_i) · max(|g_i| − ρα, 0) / (ρ(1−α)) · v̂_i k̂_i^T`, with `v̂_i` the normalized steering vector, `k̂_i = W_i^T v_i / ||W_i^T v_i||`, and `g_i = cos(v_i, W_i μ_i)`. All closed form; magnitudes come from an elastic-net budget.
- **Results (abstract).** At matched downstream performance, beats the strongest steering baseline by 17.2% (safety), 9.8% (truthfulness), 12.2% (reasoning length reduction). Models named in the part read: LLaMA-2-7B-Chat, Mistral-7B-Instruct-v0.2, Gemma-2-2B-IT, LLaMA-3-8B-Instruct.
- **Relevance.** Training-free steering-vector → rank-1 weight edits, but the input direction comes from the component's own weights, not from data keys. The edit is always on: no position or context gate. No instruction-following or format task.

### A9. Always-on closed-form "steering as weights"
- **Refusal in Language Models Is Mediated by a Single Direction.** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. arXiv 2406.11717 (2024-06). Read: `PARTS` (§2 definitions, §4.1). Difference-of-means direction computed at each **post-instruction template token** and layer. Weight orthogonalization `W_out' = W_out − r̂ r̂^T W_out` on every matrix writing to the residual stream; this is rank-1 per matrix and always on. The direction is selected at template positions but applied everywhere.
- **Turning the Spell Around: Rank-One Safety Injection (ROSI).** Shairah, Hammoud, Turkiyyah, Ghanem. arXiv 2508.20766, EMNLP 2026. Read: `ABS`. Fine-tuning-free rank-one modification of all residual-stream write matrices, steering permanently toward the refusal subspace.
- **ThinkEdit.** Sun, Yan, Weng. arXiv 2503.22048, EMNLP 2025. Read: `ABS`. Reasoning length is governed by a linear direction; edits output projections of ~4% of attention heads (0.2% of parameters) to remove the "short reasoning" direction; +6.39% accuracy on short-reasoning outputs, +3.34% overall on math benchmarks.
- **Model Surgery.** Wang, Yue, Lu, Shi, Zhao, Wang, Song, Huang. arXiv 2407.08770, NAACL 2025. Read: `ABS`. Directly edits a small subset of parameters with inference-level compute; up to 90.0% toxicity reduction on RealToxicityPrompts.
- **Resisting Harmful Instructions through Training-Free Weight Editing.** Ebrahimpour-Boroojeny, Yang, Schapiro, Borhani, Ortega, Zheng, Zhai. arXiv 2610.21189, 2026-10-15. Read: `PARTS` (abstract, start of §2). Wanda-style score (weight magnitude × difference in input-channel activity between paired prompt sets that request the undesired vs the desired behavior); the top-scoring weights are rescaled. Forward passes only. It makes the model *resist* an instruction (instructed lying, instructed sexism): +6.2 and +4.6 points honest-response rate over the strongest alternative; sexist-response rate 57.9% → 24.3%. Not gated.
- **Does Fine-Tuning Undo Activation Steering? Behavioural Recovery Without Weight-Edit Reversal.** Glass, Tucker, Li, Miron. EMNLP 2026 (Anthology 2026.emnlp-main.567; no arXiv link on the page). Read: `ABS`. Steering "embedded directly into a language model's weights" for refusal ablation and **brevity amplification**, five instruction-tuned models (3B–14B). Under SFT refusal ablation loses 69% of its effect on average, under RLHF 2%, while on average only 0.4% of the weight edit is reversed. I did not read how the embedding is done.
- **Steering Language Models with Weight Arithmetic.** Fierro, Roger. arXiv 2511.05408, ICLR 2026. Read: `ABS`. "Contrastive weight steering": subtract the weight deltas of two small fine-tunes (behavior and its opposite) and add or remove that weight direction. Needs fine-tuning. Reports weight steering often generalizes further out of distribution than activation steering.

### A10. Related theory, abstract only
- **The Safety Operator: Modulating the Expression of Safety Instructions via Spectral Optimization.** Dherin, Munn, Gonzalvo, Goldwaser, et al. arXiv 2609.36434, 2026-09-29. Read: `ABS`. Context tokens absorbed into weights as a multiplicative operator; its dominant eigenvalue modulates how strongly a safety instruction shapes generation. This is the Transmuting group's own follow-up about *instructions*.
- **Weight Updates as Activation Shifts: A Principled Framework for Steering.** Adila, Cooper, Yun, Trost, Sala. arXiv 2603.00425 (2026-02/03). Read: `ABS`. First-order equivalence between activation interventions and weight updates; trained post-block steering within 0.2–0.9% of full fine-tuning with 0.04% of parameters.
- **COLD-Steer.** Sharma, Trivedi. arXiv 2603.06495, ICLR 2026. Read: `ABS`. Training-free steering that approximates the representation change gradient descent on in-context examples would cause.
- **Cortex: Consolidating Context into Parametric Memory.** Zhang et al. arXiv 2610.20399, 2026-10-14. Read: `ABS` (no HTML). Distils context-conditioned outputs into mergeable low-rank or structural FFN updates. Trained.

---

## B. Trained prompt → weights (contrast and baselines)

### B1. Prompt Baking
- Bhargava, Witkowski, Detkov, Thomson. arXiv 2409.13697, 2024-09-04.
- Read: `PARTS` (abstract, §2, §3.2–3.5, §3.7).
- **Method.** `θ_u = argmin KL(P_θ(·|u) || P_θu(·))`, estimated on trajectories sampled from the prompted model (optionally after questions from a QA dataset such as SQuAD), optimized with a LoRA adapter. "often in as little as 5 minutes".
- **Numbers.** Baked instruction-following prompts perform "to within 8% of the prompted model" on held-out questions. Baked few-shot CoT prompts come within 1.4% of few-shot accuracy on 6 benchmarks. Re-prompting the baked model can exceed the prompted original (GSM8K: 1.4% above Meta's published Llama 3 figure of 79.6%). At most 3.4% accuracy drop on unrelated benchmarks.
- **Prompt decay (§3.7).** Using the persona-drift benchmark of Li et al., baked persona and instruction prompts keep near-constant stability over long dialogues while the prompted model declines (figure only).
- **Not done.** No analysis of where in the answer the KL sits; no closed form.

### B2. Other trained routes (abstract only)
- **Learning by Distilling Context.** Snell, Klein, Zhong. arXiv 2209.15189 (2022). Fine-tune the model to produce, without the instruction or scratch-pad, the answer it gives with them.
- **Generative Adapter.** Chen, Fang, Xia, Liu, Van Durme, Zettlemoyer, Gao, Cheng. arXiv 2411.05877 (2024). A trained generator maps a context to a low-rank adapter in one forward pass. MetaICL average accuracy 44.9 over 26 tasks; StreamingQA F1 19.5 → 31.5.
- **Text-to-LoRA.** Charakorn, Cetin, Tang, Lange. arXiv 2506.06105, ICML 2025. Hypernetwork from a task description to a LoRA; trained on 9 LoRAs; zero-shot to unseen tasks.
- **Doc-to-LoRA.** Charakorn, Cetin, Uesaka, Lange. arXiv 2602.15902 (2026-02). Hypernetwork that meta-learns approximate context distillation in one pass.
- **Cartridges.** Eyuboglu et al. arXiv 2506.06266 (2025). Trains a small KV cache per corpus with a context-distillation objective ("self-study"); 38.6x less memory at ICL-level quality.
- **Context Distillation as Latent Memory Management.** Zheng et al. arXiv 2605.28889 (2026-05). One LoRA per context plus a self-gating mechanism deciding whether to activate it.

### B3. Trained and position-restricted or schedule-gated
- **The Halt Vector: Internalizing a Causal Steering Intervention for Efficient Reasoning.** Jayabahu, Adeleke. arXiv 2608.28859, 2026-08-28. COLM 2026 workshop (non-archival). Read: `PARTS` (abstract, §4, §5, §6.1, §9).
  - Difference-of-means "halt" direction at layer 18 of DeepSeek-R1-Distill-Qwen-7B controls thinking length (ρ = −0.82 between steering strength and length at layer 18).
  - Internalized with attention-only LoRA on layers 0–18, trained to reconstruct the **whole steered activation** `h_base(t) + s(t)·α·û`, where the schedule `s(t)` is zero before the point where the answer has settled and one after. 12 epochs on 24 problems.
  - Result: −25% thinking tokens at 0.792 vs 0.812 accuracy. Maximizing only the scalar projection fails (43% longer, accuracy 0.57).
  - Limitations: one model; the settle point needs the gold answer offline.
  - Relevance: a position-gated steering signal moved into weights, by training, for the same problem as the project's experiment 9.
- **Activated LoRA (aLoRA).** Greenewald et al. arXiv 2504.12397 (2025). Read: `ABS`. A LoRA that adapts weights only for tokens after an invocation sequence, so the earlier KV cache is reusable.
- **PreFT: Prefill-only finetuning.** Lanpouthakoun, Arora, Wu, Pai, Keigwin, Jurafsky, Potts. arXiv 2605.14217 (2026-05). Read: `ABS`. Adapter (LoRA or ReFT) applied only to prefill (prompt) tokens and discarded during decoding; 1.9x throughput with 512 adapters on Llama 3.1 70B, at some loss that a higher rank compensates.

---

## C. Activation steering for instructions and format, and where in the sequence it is applied

### C1. Improving Instruction-Following in Language Models through Activation Steering
- Stolfo, Balachandran, Yousefi, Horvitz, Nushi. arXiv 2410.12877, ICLR 2025. Code: github.com/microsoft/llm-steer-instruct.
- Read: `PARTS` (§2, §3, §4 complete; App. A, E, G, H, I text; Table 8). Code: README (first part) and `utils/generation_utils.py`.
- **Vector.** For N pairs (base query, same query with instruction), take the residual stream at the **last input token** (the `<|assistant|>` template token for chat models), per layer. `u_l` = normalized mean difference.
- **Application.** `x' ← x' + c·u_l` "at a single layer across all token positions", motivated by the finding that models drift from instructions as they generate (Li et al.). For format instructions `c = z̄ − x'^T u_l`, which sets the projection to its mean value on instructed inputs. The code confirms it: the hook adds to the whole activation tensor at every step, and the sequence is recomputed without a KV cache.
- **Layer choice.** Sweep on held-out synthetic prompts; reject layers whose outputs have low GPT-2 perplexity (threshold 2.5 for format); if no layer improves validation accuracy, no steering is applied.
- **Data.** 163 IFEval format examples (12 format instruction types plus 19 language instructions); vectors from as few as 20 examples.
- **Models.** Phi-3 Mini, Gemma 2 2B IT, Gemma 2 9B IT, Mistral 7B Instruct v0.1 (plus base Gemma 2 for cross-model transfer). Greedy decoding.
- **Format numbers.** Without instruction text: about 10% unsteered → "approximately 30%" steered. With instruction text: 60–90% unsteered; steering helps two of four models. Per-model values as tabulated by Heyman & Vandeputte (C2), no instruction: 30.1 / 30.1 / 14.1 / 28.9 for Phi-3 / Gemma-2-2b / Mistral / Gemma-2-9b.
- **JSON specifically (Table 8).** In the no-instruction setting no layer passed validation for "JSON Format" on Phi-3, Gemma 2 2B IT and Gemma 2 9B IT (table entry "-"); only Mistral was steered (layer 16). So all-position single-layer steering for JSON without the instruction text was not applied in 3 of 4 models. Per-instruction accuracies are in figures only.
- **Failure explanations in the paper.**
  - "End Checker" needs an input-dependent sentence; "the steering vector is computed by averaging over all examples ... it likely fails to capture the specific sentence required".
  - "Title" and "Constrained Response": vectors over-push instruction tokens ("<<<<").
  - t-SNE (App. H): JSON, Lowercase, No Comma, Title, Quotation form tight clusters; End Checker and Constrained Response are dispersed.
- **Length.** "Be brief" style vectors work as a continuous knob when applied at all positions (Phi-3 layer 12; table of outputs shrinking from 324 to 23 words as `c` goes 0 → 40). Contrast with the project, where a single-position edit for brevity did not transfer.
- **Stated limitation (App. A).** Open degrees of freedom include "identifying where in the input or output sequence to apply the intervention"; a fixed weight is used throughout generation.

### C2. Steer Like the LLM: Activation Steering that Mimics Prompting
- Heyman, Vandeputte (Nokia Bell Labs). arXiv 2605.03907, 2026-05-05. ICML 2026. Code: github.com/Nokia-Bell-Labs/steer-like-the-llm.
- Read: `FULL` main text (§1–§6), Table 2, part of Table 3, App. A.1–A.2 text, App. E.2 intro.
- **Framing.** Prompt steering is activation steering with intervention `Δ_PS` = (activations of the response under the steered prompt) − (activations of the same response under the plain prompt).
- **Finding.** Constant steering assumes equal magnitude at all positions. Empirically "the strength of prompt steering interventions varies significantly across token positions". Prompting "tend[s] to apply strong interventions on some token positions and barely intervene on others". Strong positions are "sentence markers and positions that can be seen as branching points". Some layers show almost no local steering "except on the first few question tokens". The measured quantity is the **activation-difference norm**, on persona traits (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B). It is not the output KL.
- **Method (PSR).** `A ← A + α · ReLU(A·w + b) · z`: a rank-1 intervention whose coefficient is a learned probe on the activation itself. Single layer (S-PSR) or all layers (A-PSR). `w, b, z` trained by SGD, with an MSE loss to the prompt-steered activations or a log-likelihood loss on the prompt-steered response. Applied as an inference-time intervention on response tokens.
- **IFEval format results, no instruction in the prompt (Table 2; IF accuracy, macro-averaged over instruction types).**

  | Method | Phi-3-mini | Gemma-2-2b-it | Mistral-7B | Gemma-2-9b-it |
  | --- | --- | --- | --- | --- |
  | No steering | 11.9 | 10.6 | 6.8 | 11.4 |
  | Stolfo et al. (as reported) | 30.1 | 30.1 | 14.1 | 28.9 |
  | Stolfo et al. (reproduced) | 29.0 | 39.1 | 19.8 | 30.8 |
  | S-Const (LL-trained constant, one layer) | 11.6 | 10.7 | 19.0 | 13.4 |
  | S-PSR (LL, one layer) | 62.8 | 54.9 | 62.7 | 66.1 |
  | A-Const (LL, all layers) | 61.9 | 36.9 | 69.2 | 50.4 |
  | A-PSR (LL, all layers) | 69.0 | 68.7 | 61.1 | 71.9 |
  | Prompt (instruction text) | 72.5 | 66.8 | 61.8 | 85.7 |

- **AxBench.** A-PSR (MSE) scores 0.871 on Gemma-2-2B L20, above prompting at 0.731.
- **Stated limitations.** "rank-1 PSR models are not expressive enough to replicate prompt steering for all instruction types"; for several instruction types the MSE loss barely improved; prompting stays stronger on IFEval overall.
- **Differences from the project.** Trained by SGD; a hook, not weights; one gate function over all response tokens rather than a fixed chosen position; no count of positions; no output-level KL.
- `[my derivation]` `α·ReLU(A·w+b)·z` is exactly one extra hidden neuron (input weights `w`, bias `b`, output weights `z`) in a one-neuron MLP on the residual stream. The paper does not say this. A reviewer could argue PSR is already "a gated rank-1 edit", trained.

### C3. One Token Can Be Enough: Bridging Prompting and Activation Steering with Prefix Steering
- Xudong Zhu, Zhihui Zhu. arXiv 2610.04967, 2026-10-04.
- Read: `PARTS` (abstract, §1, §2, §5 complete, §6). §4 theory and proofs not read.
- **Theory (as summarized in the parts read).** Under fixed-state attention assumptions, steering one existing token can reproduce exactly the attention-head output caused by adding one prompt token. With multiple tokens matching is harder. Lemma 5: steering one position with strength `β* = λ · Σ_{i∈S} a_i / a_j` approximates steering all positions in `S` with strength `λ`.
- **Prefix Steering.** Apply an existing difference-in-means direction (additive steering or COAST) "over a short span starting at the final prompt token, with no further direct intervention afterward".
- **Experiments.** Qwen3 1.7B and 14B, OLMo 3 7B and 32B. Tasks: safety (HarmBench), sentiment, politeness, and two formatting tasks on MATH-500: IF-Boxed (final answer in `\boxed{}`) and IF-Plain (plain text). Capability measured by MMLU-Pro or MATH-500 accuracy.
- **Findings (text; values are in figures).**
  - One- and five-token prefixes "often retain much of full steering's behavioral control while better preserving general capabilities".
  - Steering only the final prompt token makes the model generate "Sorry" and continue refusing.
  - Formatting of the final answer after reasoning is still controlled, "well after steering ends".
  - Prompting is better on sentiment, politeness and boxed formatting; steering is better on safety and plain formatting.
  - Full steering "can achieve stronger control in some cases".
- **Open questions stated.** Whether the lasting effect is carried by a persistent latent state or by the generated tokens. "evaluation on longer generations and tasks requiring control at multiple stages could further clarify when a brief prefix suffices and when continued intervention is needed".
- **Differences from the project.** A hook, not weights. Does not claim one-token steering *beats* all-position steering on control. No JSON, no per-position KL, no predictor.

### C4. Multi-property Steering of LLMs with Dynamic Activation Composition (DAC)
- Scalena, Sarti, Nissim. arXiv 2406.17563, BlackboxNLP 2024.
- Read: `PARTS` (abstract; the passage defining the dynamic method; Fig. 5 caption and discussion; App. G).
- **Method.** At each generation step set the steering intensity `α_i = min(KL(p_unsteered || p_strongly-steered), 2)` on top-p truncated distributions.
- **Finding.** "stronger steering intensity is only required on the first few generated tokens"; `α_i` "generally decreases sharply after the first few generated tokens". For language steering, "The highest intensity is mainly focused at the beginning of the generation".
- **Relevance.** 2024 evidence, by a per-step KL, that a behavior needs pushing mainly at the start. The KL compares steered vs unsteered models, not instruction vs no instruction.

### C5. Function vectors, task vectors, in-context vectors
- **Function Vectors in Large Language Models.** Todd, Li, Sen Sharma, Mueller, Wallace, Bau. arXiv 2310.15213, ICLR 2024. Read: `PARTS` (§2.1 setup, §3.1, Table 2, natural-text paragraph, App. B passage).
  - FV is built from the outputs, at the last token of ICL prompts, of a small set of attention heads chosen by causal mediation analysis (I did not read the construction section line by line).
  - Added to the hidden state at **the last token of the prompt** at layer ≈ |L|/3. For multi-token natural-text generation: added "to the final token of the original prompt, and for all subsequent token predictions".
  - Zero-shot accuracy averaged over 6 tasks, baseline → +FV: GPT-J 5.5% → 57.5%; GPT-NeoX 6.7% → 57.1%; Llama 2 70B 8.2% → 83.8%.
  - Tasks have short answers (the first answer token is scored).
- **In-Context Learning Creates Task Vectors.** Hendel, Geva, Globerson. arXiv 2310.15916, Findings EMNLP 2023. Read: `PARTS`. Task vector = representation of the "→" token at layer L from demonstrations; patched at the same layer and position on a query-only prompt; keeps "around 80-90% of the accuracy of regular ICL" (baseline 10–20%).
- **In-Context Vectors.** Liu, Ye, Xing, Zou. arXiv 2311.06668, ICML 2024. Read: `PARTS`. Vector from last-token latent differences of demonstrations; added "at all layers and every token position"; single-layer variants were not better than no ICV.

### C6. Benchmarks and trained position-restricted interventions
- **AxBench.** Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts. arXiv 2501.17148 (2025). Read: `PARTS` (abstract, steering table and text). Steering score (0–2), average over Gemma-2-2B L10/L20 and Gemma-2-9B L20/L31: Prompt 0.894, LoReFT 0.741, SFT 0.676 (2B only), LoRA 0.615, ReFT-r1 0.543, DiffMean 0.239, SAE 0.165. "prompting outperforms all existing methods, followed by finetuning"; SAEs are not competitive.
- **ReFT / LoReFT.** Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts. arXiv 2404.03592 (2024). Read: `PARTS` (position hyperparameters; App. remarks). Trained low-rank interventions on `p` prefix and `s` suffix **prompt positions** only, at chosen layers; nothing during decoding ("LoReFT may only need to intervene on the prompt tokens"). The paper notes "PEFT frameworks lack the notion of time or sequence ... representation modifications are necessarily applied to every token".
- **PrOSV — Towards Steering without Sacrifice.** Bao, Li, Yu, Su, Zhang, Yan, Weng, Yin, et al. arXiv 2605.05983, ICML 2026. Read: `ABS`. Trained steering vectors that intervene "only on a few prompt tokens"; beats full-sequence trained vectors on AxBench under joint training of factor and direction.

### C7. Conditional or gated steering (all inference-time)
- **CAST — Programming Refusal with Conditional Activation Steering.** Lee, Padhi, Ramamurthy, Miehling, Dognin, Nagireddy, Dhurandhar. arXiv 2409.05907, ICLR 2025 spotlight. Read: `PARTS` (§3.1, start of §4). `h' ← h + f(sim(h, proj_c h))·α·v` with condition vector `c`, behavior vector `v`, and a step function `f` with threshold `θ`. Layer, threshold and comparison direction by grid search. Once the condition fires "the behavior vector is added to all subsequent forward passes". The condition is about the prompt's content class, not a position. "The model itself does not undergo any weight update."
- **GCAD — Prompt-Activation Duality.** Kang, Liu, Ma, Huang, Tan, Jiang. arXiv 2605.10664, NeurIPS 2026. Read: `PARTS` (abstract, method excerpt). Residual steering degrades multi-turn dialogue through "KV-cache contamination". GCAD extracts steering signals from the system-prompt contribution to self-attention outputs and applies them with a per-token sigmoid gate. Coherence drift −18.6 → −1.9; turn-10 trait expression 78.0 → 93.1.
- **DIAL.** Tran, Himawan, Verma, Wong, Xiao, Rus, Low. arXiv 2610.31438, 2026-10-22. Read: `ABS`. A classifier over hidden states "adaptively sets the steering magnitude at each token"; trained by distilling classifier-free guidance; text-conditioned so one adapter serves new instructions. Opening example: "Keep it short".
- **MAT-Steer.** Nguyen, Prasad, Stengel-Eskin, Bansal. arXiv 2502.12446, ACL 2025. Read: `ABS`. Learned "selective token-level intervention" across multiple attributes.
- **Stochastic Token Steering — A Coin Flip Per Token.** Eshraghi, Gondara, Huang, Suresh, Teran, Pradeep, Xu, Chevalier. arXiv 2607.05615 (2026). Read: `ABS`. Steering a random fraction of tokens, or only a leading window, "recovers most of the behavioral effect of dense steering"; the best magnitude rises as the fraction falls.
- **Tool Calling is Linearly Readable and Steerable.** Wu, Wang, Cho, Yang, Koshiyama, Bulathwela, Perez-Ortiz. arXiv 2605.07990 (2026). Read: `ABS`. Difference of per-tool mean hidden states switches the generated tool call in 56–78% of held-out tool pairs across eight models (4B–27B). Relevant to JSON tool calling; inference-time.

### C8. Position taxonomies and remarks in older steering papers
- **The Cylindrical Representation Hypothesis.** Gao, Zhang, Liu, Ji, Wang, Song, Ghosh, Mohamed, et al. arXiv 2605.01844, ICML 2026. Read: `PARTS` (abstract, App. H.2). Compares four placements: all prompt tokens, last prompt token only, all output tokens, all tokens. Largest usable steering factor before corruption: Gemma 5 / 50 / 50 / 3, LLaMA 5 / 80 / 80 / 3. A single position tolerates about 10x more strength.
- **CAA.** Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner. arXiv 2312.06681, ACL 2024. Read: `PARTS`. Vector added "at every token position after the user's prompt". The authors write: "By intervening at a smaller, more targeted, subset of tokens, a better trade-off between intervention size and effect size may be achieved."
- **ActAdd.** Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid. arXiv 2308.10248. Read: `PARTS`. Steering activations aligned to the front positions of the prompt (`a = 1`).
- **SAIF.** He, Zhao, Qiao, Yang, Payani, Ma, Du. arXiv 2502.11356 (2025). Read: `PARTS` (abstract, §4.3 part, §4.5 first finding). Steers with ~15 SAE latents on the residual stream at the last token; a single latent gives zero accuracy; over 30% strict accuracy when steered. Placing the instruction after the content beats placing it before (strict accuracy 0.23 vs 0.14, loose 0.64 vs 0.47).

### C9. Predicting whether a steering vector will work
- **Understanding (Un)Reliability of Steering Vectors in Language Models.** Braun, Eickhoff, Krueger, Bahrainian, Krasheninnikov. arXiv 2505.22637, ICLR 2025 workshop. Read: `PARTS` (abstract, results passages, limitations).
  - CAA on 36 model-written-evals datasets, Llama2-7B-chat.
  - "directional agreement, as measured by cosine similarity between activation differences of the training data, is predictive of steering success".
  - Separability along the difference-of-means line (d′) also predicts it.
  - "vector steering is unreliable when the target behavior is not represented by a coherent direction."
  - Limitation: one model, multiple-choice datasets.
- `[my derivation]` If the per-example differences `d_i` have equal norm, the project's ratio `||mean d||^2 / mean ||d_i||^2` equals the mean of `cos(d_i, d_j)` over all pairs (including `i = j`, which stops mattering for large n). So the proposed predictor is close to Braun et al.'s directional agreement, applied to a new object (instruction differences at one position, for a weight edit).
- **Analyzing the Generalization and Reliability of Steering Vectors.** Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso, Kirk. arXiv 2407.12404. Read: `ABS`. Steerability varies a lot across inputs and across concepts; some vectors are brittle to prompt changes.
- **Mechanistic Indicators of Steering Effectiveness.** Jafari, Xue, Salim. arXiv 2602.01716 (2026). Read: `ABS`. Entropy-based branching factor and a vocabulary-space KL during decoding predict steering success. These are measured while steering, not before building the vector.

---

## D. Rank-one model editing with a key

### D1. ROME and MEMIT
- **ROME.** Meng, Bau, Andonian, Belinkov. arXiv 2202.05262, NeurIPS 2022. Read: `PARTS` (§3.1 method; App. on second-moment statistics). `Ŵ = W + Λ (C^{-1} k*)^T` on one MLP projection matrix, with `C = K K^T` and `Λ = (v* − W k*) / ((C^{-1}k*)^T k*)`. `k*` = MLP hidden activation at the last subject token. `v*` is found by gradient optimization so that the model outputs the new object. `C` is estimated from 100,000 hidden-state samples over **all token positions** of Wikipedia text.
- **MEMIT.** Meng, Sen Sharma, Andonian, Belinkov, Bau. arXiv 2210.07229. Read: `PARTS` (derivation passages). `Δ = R K_1^T (C_0 + K_1 K_1^T)^{-1}`, with `R = M_1 − W_0 K_1` and `C_0 = λ·E[k k^T]`.

**Formal relation to the project's ridge read vector** `[my derivation]`. Let `H_bg` be background keys (MLP hidden activations at non-target positions), `K_1` the target keys, and `v = argmin Σ_bg (v^T h)^2 + Σ_targets (v^T k − 1)^2 + ρ||v||^2`. Then `v = (H_bg H_bg^T + ρI + K_1 K_1^T)^{-1} K_1 1`, and `u v^T = (u 1^T) K_1^T (C_0 + K_1 K_1^T)^{-1}` with `C_0 = H_bg H_bg^T + ρI`.

- That is **MEMIT with the same residual `u` for every key**, and with the background covariance estimated on the model's own prompts at non-target positions instead of Wikipedia.
- For a single key, Sherman–Morrison gives `v = C_0^{-1}k / (1 + k^T C_0^{-1} k)`. This is ROME's direction `C^{-1}k*`, with gain `s/(1+s)` where `s = k^T C_0^{-1} k`. ROME normalizes so the gain is exactly 1.
- The family, by how each side is constrained:

  | Method | Target keys | Background keys |
  | --- | --- | --- |
  | ROME | hard (`Ŵ k* = v*` exactly) | soft, through `C` |
  | MEMIT / ridge (the project) | soft | soft |
  | AlphaEdit / SteerEdit | soft (ridge) | hard (null-space projection) |

- The other difference from ROME, BadEdit and Concept-ROT: they obtain the value by gradient optimization. The project and SteerEdit take it from a mean activation difference.

### D2. Editing-based backdoors (lexical triggers)
- **BadEdit.** Li, Li, Chen, Zhang, Liu, Wang, Zhang, Liu. arXiv 2403.13355, ICLR 2024. Read: `PARTS` (abstract, §4.3 passages, ablation passage). Key = MLP representation "at the position of the trigger" token; value optimized to maximize the target output; MEMIT-style batch edits; layers 15–17 (GPT2-XL) and 5–7 (GPT-J); 15 samples; "up to 100%" attack success.
- **JailbreakEdit — Injecting Universal Jailbreak Backdoors into LLMs in Minutes.** Chen, Zhang, Pei. arXiv 2502.10438, ICLR 2025. Read: `META` (described via SteerEdit: maps a trigger token to an affirmative prefix).
- **DualEdit.** Jiang, Zhao, Fang, Ma, Wang, Wang, He, Deng. arXiv 2506.13285. Read: `ABS`. Names "safety fallback": the edited model starts affirmatively and then reverts to refusal. Promotes affirmative tokens and suppresses refusal tokens; +10% attack success, −11% fallback rate.
- **SafetyEdit.** Ke, Wang, Zhang, Guo, Zhao, Huang. arXiv 2610.22640, 2026-10-15. Read: `ABS`. Optimizes one refusal-related MLP coordinate and uses "KV-cache marking in early attention layers to propagate a prefix-trigger signal across subsequent instruction positions".
- **Lesson for the project.** This line found that an edit forcing only the first tokens ("Sure") is often not enough for a behavior that must persist. That is the same shape as the project's "be brief" failure. SteerEdit's answer was to write a direction instead of a token.

### D3. Concept-ROT: Poisoning Concepts in LLMs with Model Editing
- Grimes, Christiani, Shriver, Connor (CMU SEI). arXiv 2412.13341, ICLR 2025. Code: github.com/keltin13/concept-rot.
- Read: `PARTS` (§4 complete, §5.1 setup and results, part of §5.2, §5.3, §6, Table 1).
- **Key.** A concept vector: mean activation, or the first principal component of paired differences, at the input of the edited layer, "at some consistent token position (e.g. the end-of-turn token)". Scaled to the mean concept score of on-concept prompts. Collected with a template "Consider the amount of <concept> in the following text: ...".
- **Value.** Vector `z` added to the MLP output at that position, optimized by gradient to produce a target string. Inserted with the ROME closed form into `W_down` of a single layer. They cut the data needed for the second-moment estimate by 100–1000x.
- **Results (Table 1, no control data, mean of 8 concepts).** Attack success 94.8 / 87.9 / 76.4 for Gemma-7B / Llama-3.1-8B / Mistral-7B-v2, with Open-LLM score 53.4 / 69.8 / 65.4 against 53.5 / 69.6 / 65.7 unedited. Edit time 14–19 s.
- **Jailbreak variant.** Works with as few as 5 examples. A prefilling attack "in many cases ... switches back to a refusal state during generation", whereas the edit avoids this.
- **Stated limitations.** "we cannot say a priori what concepts or layers will serve as effective triggers". Failures occur where on- and off-concept score distributions overlap. "applying Concept-ROT, specifically the concept triggers, to non-trojaning tasks is a promising direction for future work."
- **Differences from the project.** The key encodes a topic (context-gated), not "this is the answer-start position for any prompt". The value needs gradient optimization. Single layer. Target is a fixed string or jailbreak, not an instruction's effect.

### D4. Other key-gated editors (metadata only)
- **AlphaEdit: Null-Space Constrained Knowledge Editing.** Fang, Jiang, Wang, Ma, Jie, Wang, He, Chua. arXiv 2410.02355. `META`. The null-space projection that SteerEdit reuses.
- **Transformer-Patcher: One Mistake worth One Neuron.** Huang, Shen, Zhang, Zhou, Rong, Xiong. arXiv 2301.09785, ICLR 2023. `META`.
- **GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.** Hartvigsen, Sankaranarayanan, Palangi, Kim, Ghassemi. arXiv 2211.11031, NeurIPS 2023. `META`.
- **Tracing and Reversing Edits in LLMs.** Youssef, Zhao, Seifert, Schlötterer. arXiv 2505.20819, ICLR 2026. `ABS`. For factual rank-one edits, the edited object can be inferred from the weights alone (up to 99%) and up to 94% of edits reversed. By implication a rank-1 format edit is probably detectable in the same way (my inference, not tested in the paper).

---

## E. Where instructions act, and instruction drift

### E1. Drift over turns and long contexts
- **Measuring and Controlling Instruction (In)Stability in Language Model Dialogs.** Li, Liu, Bashkansky, Bau, Viégas, Pfister, Wattenberg. arXiv 2402.10962, COLM 2024. Read: `PARTS` (abstract, §1, §4). "significant instruction drift within eight rounds" (LLaMA2-chat-70B, GPT-3.5). The attention share on system-prompt tokens stays almost constant within a turn and drops sharply between turns. Remedy: split-softmax (inference-time attention boost).
- **Prompt Baking §3.7** (B1): trained baking keeps persona stability flat over long dialogues. The only evidence I found that a weight update avoids drift; it is trained.
- **Plans Fade Fast.** Mehta, Datta. arXiv 2606.22953, COLM 2026 workshop. Read: `ABS`. Hidden-state contrast with vs without the plan exchange falls 4.1x (Llama-3.1-70B) and 5.6x (Llama-3.1-8B) from the first to the second action after the plan. Native Qwen3 does not show the same timing.
- **Selective Prompt Anchoring.** Tian, Zhang. arXiv 2408.09121, ICML 2025. Read: `ABS`. Code LLMs "pay less attention to user prompts as more code tokens are generated".
- **Attention-steering remedies (all inference-time, read: `ABS` unless noted).**
  - InstABoost — Guardieiro, Khare, Stein, Wong, arXiv 2506.13734: constant additive bias on instruction-key attention logits in all layers and heads.
  - Spotlight — Venkateswaran, Contractor, arXiv 2505.12025, EACL 2026.
  - V-Steer — Zeng, Lee, Zhao, Hockenmaier, arXiv 2607.26228, COLM 2026: edits cached value vectors at prompt positions, chosen by direct logit attribution "on the first next token prediction"; constraint accuracy from under 18% to as high as 92%.
  - PASTA — Zhang, Singh, Liu, Liu, Yu, Gao, Zhao, arXiv 2311.02262, ICLR 2024: `META` only.

### E2. Evidence that a behavior is decided at the first output tokens or the template tokens
- **Safety Alignment Should Be Made More Than Just a Few Tokens Deep.** Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson. arXiv 2406.05946 (2024). Read: `PARTS` (abstract, per-token KL passages). Per-token `KL(π_aligned(·|x, y_<k) || π_base(·|x, y_<k))`, teacher-forced on harmful answers, is "largely biased toward these initial token positions". Fine-tuning attacks perturb the first few tokens most. Same measurement as the project's, comparing two models instead of instruction on vs off.
- **Why Safeguarded Ships Run Aground? ... Anchored in the Template Region.** Leong, Yin, Wang, Li. arXiv 2502.13946, ACL 2025. Read: `PARTS` (abstract, intro, patching result). Attention at the final input position shifts from the instruction to the chat-template region on harmful inputs. Patching a few heads' states from the template region changes the compliance decision; patching the instruction region does not.
- **Arditi et al.** (A9): the refusal direction is found at post-instruction template tokens.
- **Stolfo et al., Todd et al.**: the instruction or task is readable at the last input token.

### E3. Measuring per-token context dependence
- **PECoRe — Quantifying the Plausibility of Context Reliance in NMT.** Sarti, Chrupała, Nissim, Bisazza. arXiv 2310.01188, ICLR 2024. Read: `PARTS` (CTI section). Force-decode the contextual output without the context, and compute per target token a contrastive metric between `P_ctx^i` and `P_no-ctx^i`, including `KL(P_ctx^i || P_no-ctx^i)`. Tokens above a threshold are "context-sensitive". Used for document context in MT. This is the project's per-position KL, applied to a different kind of context.
- **MIRAGE.** Qi, Sarti, Fernández, Bisazza. arXiv 2406.13663, EMNLP 2024. Read: `ABS`. Same idea for RAG answer attribution.
- **Forking Paths in Neural Text Generation.** Bigelow, Holtzman, Tanaka, Ullman. arXiv 2412.07961. Read: `ABS` + intro. "forking tokens": resampling at a few specific tokens, sometimes punctuation, changes the whole outcome.
- **Contrastive decoding with vs without context.** Context-aware decoding (Shi et al., arXiv 2305.14739) and classifier-free guidance for LMs (Sanchez et al., arXiv 2306.17806). Read: `ABS`. Both use the per-step difference of distributions at inference, without analysing where it is large.

### E4. Format and length mechanisms
- **Where Do Apparent LLM Clinical Triage Failures Arise? Localizing the Multiple-Choice Format Effect.** Navarro, Como, Sheng, Ananthan, Berkovsky. arXiv 2605.29889 (2026). Read: `ABS`. With SAE features in Gemma 3 4B/12B IT and Qwen3-8B, medical features fire on the narrative but are inactive "at the multiple-choice decision token". Scaffold features account for over 91% of unsigned attribution at that token in both Gemma models. The format effect lives at the answer-selection token.
- **The Format Tax.** Lee, D'Antoni, Berg-Kirkpatrick. arXiv 2604.03616. `META` (a search summary says structured-output requirements degrade reasoning; I did not read the abstract).
- **Length Representations in LLMs.** Moon, Choi, Kwon, Kamigaito, Okumura. arXiv 2507.20398, Findings EMNLP 2025. Read: `ABS`. Attention mechanisms are critical for output length; scaling specific hidden units changes length without losing content.
- **How Much is Left? LLMs Linearly Encode Their Remaining Output Length.** Merzouk, Carpov, Bronzi, Fornasiere, Oberman. arXiv 2607.05316 (2026). Read: `ABS`. "total response length is linearly decodable from the prompt's last hidden state alone, before any output is emitted" (three 7–8B models). The authors say it is decodable, not shown causal. This suggests an untested variant for brevity: a different `u` at the answer-start position.
- **Do LLMs "know" internally when they follow instructions?** Heo, Heinze-Deml, Elachqar, Chan, Ren, Nallasamy, Miller, Narain. arXiv 2410.14516, ICLR 2025. Read: `ABS` + one passage. An "instruction-following dimension" predicts compliance; analysed at the first, middle and last token; generalizes across tasks but not across instruction types.

---

## 1. Closest prior work to the project's claim (ranked)

The claim has four parts: (i) an instruction acts at few positions; (ii) a training-free conditional rank-1 weight edit at that position; (iii) instruction-level JSON compliance without the instruction; (iv) a transferability predictor. No single paper covers all four. Ranking by overall overlap:

1. **Transmuting prompts into weights (arXiv 2510.08734).** Same goal and same closed-form least-squares family.
   - Same: with/without-instruction pairs teacher-forced on the same answer; activation differences; least squares into MLP weights layer by layer; no training; a ~1B model.
   - Different: fits **all positions** at once; a full-rank matrix on the MLP input matrices plus a bias (or `W_down` absorption) rather than a rank-1 gated write; no format instruction; no per-position analysis; no count of edits; no steering baseline; its own failures (Country→Capital 47.6 vs 79.0; list-first 31.67 vs 83.25 in v2) are left unexplained.
   - The project's edit is its `W_down` least squares with a position-masked target `[my derivation]`.
2. **SteerEdit (arXiv 2604.12359, ACL 2026).** Closest on the edit itself.
   - Same: closed-form, multi-layer, rank-1 write of a mean-difference vector into `W_down`, with a read vector fitted on MLP hidden activations at the last token to fire on target keys and vanish on others.
   - Different: the gate is a lexical trigger; the purpose is a jailbreak backdoor; there is no instruction, no format, no position analysis; hard null-space instead of ridge.
3. **Prefix Steering (arXiv 2610.04967).** Closest on the positional finding.
   - Same: steering only at the final prompt token often retains most of the control, including answer-format control, and preserves capability better.
   - Different: an inference-time hook; no weights; no JSON; does not claim single-position beats all-position on control; no per-position KL; explicitly leaves "when a brief prefix suffices" open.
4. **Steer Like the LLM (arXiv 2605.03907, ICML 2026).**
   - Same: shows prompts act unevenly over positions; builds a gated rank-1 intervention; evaluated on IFEval format without instruction text (55–72% vs prompt 62–86%).
   - Different: SGD-trained; a hook; gate learned over all response tokens; measures activation norms, not output KL; no count.
5. **Stolfo et al. (arXiv 2410.12877).** The baseline. Vector from the last input token, added at all positions of one layer; ~10% → ~30% without instruction text; for JSON, no layer passed validation in 3 of 4 models. Names "where in the sequence to intervene" as unexplored.
6. **Concept-ROT (arXiv 2412.13341).** ROME with a non-lexical key collected at the end-of-turn token, installing a behavior. Different: concept-gated, gradient-optimized value, single layer, trojan setting.
7. **PatchKV (arXiv 2609.39329).** Training-free ridge patch of `W_out` from context-induced differences. Different: all positions, full rank, document content, paired with a compressed cache.
8. **DAC (arXiv 2406.17563), Qi et al. (arXiv 2406.05946), PECoRe (arXiv 2310.01188).** Per-position KL exists as a tool and has shown concentration at the first tokens for steering and for safety alignment. None applies it to a text instruction or uses it to decide the number of edits.
9. **Braun et al. (arXiv 2505.22637).** Directional agreement of per-sample differences predicts steerability. Different: activation steering on multiple-choice persona datasets, one model; not for instructions, positions, or weight edits.
10. **Function vectors, ReFT, PrOSV, PreFT, aLoRA, the Halt Vector.** Last-token or prompt-only interventions, as hooks or trained.

## 2. What appears to be open

No paper I found answers these:

1. **Counting positions for a text instruction.** How many answer positions does a given instruction need (per-position KL of instruction on vs off, teacher-forced), and does that count equal the number of weight edits needed? Steer Like the LLM shows unevenness in activation norms for persona traits but does not count or use output KL. Prefix Steering tries 1 and 5 tokens without a criterion.
2. **A position-gated, non-lexical, training-free weight edit for a format instruction.** SteerEdit gates on a trigger word, Concept-ROT on a topic, Transmuting and PatchKV on nothing. None targets "the answer-start position of any prompt".
3. **Instruction-level format compliance from weights alone without training.** Transmuting has no format task. The best no-instruction format numbers I found are trained hooks (PSR 55–72% on IFEval format). Stolfo-style steering is ~30%.
4. **Predicting transferability before building the edit.** Braun et al. give a close statistic for CAA. Nobody tests it for instruction differences at a single position, relates it to weight-edit success, or uses it to explain a specific failure such as brevity. Stolfo et al. give only qualitative cluster plots. Concept-ROT says "we cannot say a priori what concepts or layers will serve as effective triggers".
5. **Why brevity resists a single-position edit.** All-position steering modulates length continuously (Stolfo), and Glass et al. embed "brevity amplification" in weights, presumably always on. Whether brevity needs several gated positions (for example sentence ends), a persistent edit, or a different vector at the answer start (the length plan decodable there, per Merzouk et al.) is untested.
6. **Does a training-free weight edit avoid instruction drift?** Only trained baking has been tested (Prompt Baking §3.7). A position-gated edit would re-fire at each assistant-turn start if the key still matches there; nobody has measured this.
7. **What carries the effect after the first token:** a persistent internal state or the generated tokens themselves? Prefix Steering names this as open. The editing-backdoor line ("safety fallback") shows that the generated-token route alone can fail.
8. **Ridge zeros vs null-space projection for position-gated edits.** SteerEdit's ablation (2.3% → 63.7% unintended firing without the null-space constraint) suggests the choice matters. No comparison exists for position gating.
9. **Composition of several instruction edits in weights.** Stolfo et al. compose steering vectors at inference. No weight-edit version was found.
10. **The unifying observation** `[my derivation]` that Transmuting's `W_down` least squares, MEMIT, SteerEdit and the project's edit are one family that differs only in the target mask and the constraint type. I did not see it stated anywhere.

## 3. Risks to novelty

Work that already does position-gated or context-gated weight edits for behaviors, or that pre-empts part of the claim:

- **Context-gated closed-form weight edits for behaviors exist.** SteerEdit (ACL 2026) is the sharpest: a steering vector compiled into `W_down`, rank-1, gated, null-space constrained, no training. BadEdit, JailbreakEdit, DualEdit and SafetyEdit are the lexical-trigger line. Concept-ROT (ICLR 2025) already replaced the lexical key with a vector read at the end-of-turn token and suggested using it for non-trojan tasks. "Conditional rank-1 edit that adds a vector only when a key matches" cannot be claimed as new. Only the choice of key (a position), the source of the value (an instruction's effect) and the application (format) can.
- **"One position is enough" is published at the activation level.** Prefix Steering (Oct 2026) includes answer formatting. DAC (2024) and function vectors (2023) came earlier. Stochastic Token Steering and the Cylindrical-hypothesis appendix compare placements. A reviewer may read the 27/30 vs 6/30 contrast as an instance of this, and may add that the comparison confounds position with number of layers (6 edits across layers vs one steered layer).
- **"A prompt acts on few positions" is published** for persona traits as activation norms (Steer Like the LLM, ICML 2026), and the per-token KL concentration at the first tokens is published for safety alignment (Qi et al., 2024).
- **Trained position-gated internalization exists.** The Halt Vector trains a LoRA to reproduce a steering vector only after a schedule point, for the stop-thinking problem (overlaps the project's experiment 9). ReFT, PrOSV, PreFT and aLoRA are trained prompt-only or position-restricted interventions.
- **Training-free instruction → weights is an active line at Google.** Transmuting v3 (June 2026) and The Safety Operator (Sept 2026) come from the same group. PatchKV (NeurIPS 2026) builds on it with ridge patches of `W_out`. Adding a position mask to their least squares is a small step for them.
- **Always-on steering → weights is crowded.** Abliteration, ROSI, ThinkEdit, Steer2Edit (NeurIPS 2026), Model Surgery, Glass et al. (EMNLP 2026, includes brevity).
- **The predictor has a close precedent** in Braun et al.'s directional agreement.
- **Baselines a reviewer could ask for, taken from this literature:**
  - Prefix Steering as a hook at the same position.
  - All-layer difference-of-means steering.
  - PSR (trained gated steering).
  - Transmuting's all-position least squares.
  - A null-space variant of the read vector (SteerEdit / AlphaEdit).
  - LoRA prompt baking.
- **Not assessed here.** The Nov-2026 Weight-Space Programming follow-ups, which are the other agent's scope.

## 4. What I could not verify

- arXiv's export API returned HTTP 429 for most of the session. Metadata came from arXiv abs pages instead. arXiv-side keyword search was limited to a few listing queries, so recent papers may be missing. OpenAlex citation lists can be incomplete.
- Transmuting v1 (Oct 2025): HTML not available. I cannot say how v1 differed from v2.
- Transmuting on Gemma: how the thought-vector bias was realised for the non-dictionary tasks.
- SteerEdit: whether keys are taken at the trigger token or at the last chat-template token; the exact attack-success table values (tables did not parse); no code link found.
- Stolfo et al.: per-instruction JSON accuracies (figures only).
- Prefix Steering, DAC, PatchKV, Dherin et al.: most quantitative results are in figures; only the numbers quoted above were read.
- Glass et al. (EMNLP 2026): only the Anthology abstract. The embedding method and how brevity was embedded are unknown to me.
- Papers marked `ABS` or `META` were not read beyond that. Their methods are described only as far as the abstract states.
- I found no paper doing the project's exact combination. That is an absence in my searches, not proof of absence.
