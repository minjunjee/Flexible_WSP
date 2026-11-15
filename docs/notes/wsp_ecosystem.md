# The Weight-Space Programming (WSP) ecosystem: what the follow-ups have and have not done

Survey date: 2026-11-15 (KST). Scope: the original blog post and the work that followed it within two weeks.
Older steering and model-editing literature is covered by another note and appears here only where a WSP follow-up uses it.

Everything below was looked up on 2026-11-15; nothing is from memory. Each item states how much of it was read.

Reading-depth tags:

- **[FULL]** the whole text was read (figures and interactive elements were not seen).
- **[PART]** part of the text was read; the parts are named.
- **[ABS]** abstract only.
- **[LIST]** only the one-paragraph summary in the `awesome-weight-space-programming` list; the source itself was not opened. Numbers with this tag are the list maintainer's wording, not checked against the source.

## 0. Short answers

- Nobody in this ecosystem has shown an *instruction-free* output format installed by a weight edit at instruction-level compliance. IWIBench is still the only controlled test of that, and it is 0/192 for all eleven published editor configurations.
- Format compliance *with the instruction still in the prompt* has been improved by weight edits twice (SmolLM2 BFCL 33% → 50%; Kuusela valid JSON 42% → 95%). The project plan's "20% → 20%, no successful edit" for SmolLM2 is out of date.
- Stopping the thinking phase by a rank-1 or few-neuron edit that pushes toward `</think>` has been done at least four times (Kuusela, Weight Whisperer, NeoCortex, luizmlo); the first three gate the push on a hidden-state condition. The plan's experiment 9 is not new as an idea.
- Position-keyed, training-free "instruction into MLP weights" exists for a robot policy (arXiv 2611.11894), with keys at fixed input positions. A MEMIT-style edit keyed at the answer position of the prompt exists for classification on Qwen3.5-0.8B (trehansalil). Neither is for an output-format instruction, neither is rank-1, and neither measures which answer positions matter.
- No entry gives SAE or transcoder features to the editor and measures the effect. One study gives the editor an activation "atlas" and finds it hurts (3 of 3 pairs).

## 1. The original post

**Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**
Rikiya Takehi*, Jayden Teoh*, Alex Zhang, Omar Khattab (MIT CSAIL). Blog post, 2026-11-02; "Response to Comments" section added 2026-11-12.
https://rikiyat.github.io/blog/weight-space-programming/ · logs https://github.com/RikiyaT/lm_understand-logs · edited model https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
Read: blog **[FULL]**; logs `README.md` and `exp8_math/README.md` **[FULL]** (local copies under `/data/.../ext/lm_understand-logs`).

The local logs README gives the title as "Sparks of Weight Space Programming: LLMs can understand and write neural network weights"; the live page says "natively read & write". The editor ("Astra") is OpenAI `gpt-6-astra` run through the Codex CLI with all built-in tools off.

| Exp | Task | Target | What Astra was given | Result |
|---|---|---|---|---|
| 1 | Name the game a PPO policy was trained on | MLP 500→32→32→5, 17,253 weights | all weights as text, no tools | 4/4 |
| 2 | Name the shape a DDPM draws | MLP, 23,824 weights | weights as text, no forward pass | 3/3 (draws outlines, real shapes are filled) |
| 3 | Name the training language and the copying heads | char transformer, 4 layers × 6 heads, 7.16M weights, middle blocks only | Python on the weights, ≤100 runs, no running the network | language 3/3 in both attempts; all copying heads found |
| 4 | Fix a stuck Kinetix car policy | 171,664 weights | weights, scene description, one recorded episode, no tools, one reply | 0/256 → 1,024/1,024, 3.5% of weights changed |
| 5 | Write MinAtar policies from zero | 600→32→32→6 | rules and source code of the game, no tools | Breakout 517.4 (PPO 100M 15.8, 500M 25.3); Seaquest 53.1 (5.6, 88.9); Asterix 53.2 (25.8, 55.2) |
| 6 | Write a 9×9 Go network | 1,480,274 weights | Python sandbox, 0–50 forward passes | 0 passes: wins more than half against two smaller PPO nets; 50 passes: beats the equal-sized PPO net |
| 7 | Which Gemma 3 270M fine-tune is sycophantic | 2 × 270M | weights + Python: failed in 4 attempts; + attention-pattern tool (≤16 own texts, ≤20 runs): 3/3 pairs | |
| 8 | Make Qwen3.5-0.8B better at math | 24 layers, thinking on | Python kernel on weights (CPU, ≤300 runs), practice problems, 10/25/50 forward passes | see below |

Experiment 8 details (from the logs README):

- A forward pass is either a generation on up to 512 practice problems or a teacher-forced read of up to 1,024 texts (log-probs, hidden states, MLP activations, attention).
- Forbidden: gradients or any training, running the model in the kernel, internet. `save_edit` refused edits to the final norm, the embedding / output layer, and whole-tensor rescaling.
- Final edits. 10 passes: `W' = W − 1.5 u(uᵀW)` on 30 output matrices in layers 8–22. 25 passes: attention head 7 of layer 15 zeroed. 50 passes: the loop direction subtracted from 64 output columns each of the MLPs of layers 15 and 19 (131,072 weights, 0.016%; 130,188 differ after bf16 rounding).
- Evaluation: vLLM, thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, ≤32,768 new tokens.

| Qwen3.5-0.8B | MATH-500 | GSM8K | IFEval | MATH-500 answers cut at token limit |
|---|---|---|---|---|
| base | 7.9 | 13.8 | 21.0 | 93.0 |
| 10 passes | 12.3 | 25.7 | 49.0 | 87.2 |
| 25 passes | 14.1 | 25.1 | 39.9 | 86.4 |
| 50 passes | 48.1 | 67.1 | 62.3 | 33.1 |

Gemma 3 270M follow-up (rarely loops, 8.5% cut): MATH-500 6.8 base → 7.0 / 7.6 / 7.7 / 7.0 with 5 / 10 / 25 / 50 passes. No gain.

Stated limits and non-claims:

- "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."
- Capabilities are "still limited, specifically due to inefficiencies in context length and harnesses".
- WSP is "not a method" but "the name of an emerging concept"; baselines are not meant to rank methods.
- No comparison with interpretability researchers.
- A benchmark and a harness are announced as in progress.
- The post does not try to install an output format, an instruction, or a position-specific behaviour. IFEval moved only as a side effect of the loop fix. The editor never saw SAE or transcoder features.

## 2. Follow-up lists

| List | URL | State on 2026-11-15 | Read |
|---|---|---|---|
| SajjadPSavoji/awesome-weight-space-programming (mirror: SajjadPSavoji/awesome-wsp) | https://github.com/SajjadPSavoji/awesome-weight-space-programming | 605 entries: 122 WSP-era, 188 "foundations" (1943–2025), 295 surveys. Updated daily. | All 122 WSP-era summaries read (README sections "Original Report" to "Analyses and Critiques"). Foundations and surveys only keyword-scanned through `data/entries.json`. |
| Boom5426/Awesome-Weight-Space-Programming | https://github.com/Boom5426/Awesome-Weight-Space-Programming | 20 direct WSP papers, 9 reports/blogs, related work | README **[FULL]** |
| brighterr/awesome-weight-space-programming (mirror: brighterr/awesome-wsp) | https://github.com/brighterr/awesome-weight-space-programming | last curated 2026-11-03, small | README **[FULL]**; the mirror only URL-diffed |
| awesome-wsp/awesome-wsp.github.io | https://github.com/awesome-wsp/awesome-wsp.github.io | timeline site | only URL-diffed |

"56 entries as of Nov 5" is consistent with the commit history of the first list: 30 entries at creation (2026-11-04), then +1, +7, +17 by 2026-11-05 17:32 UTC and +2 by 2026-11-06 03:00 UTC. The foundations and surveys were added on 2026-11-06. The 122 WSP-era entries split as: original 1, LM repair 21, LM reading 2, LM writing 7, VLA 6, RL control 8, other domains 26, games 7, quantum 2, benchmarks 4, tools 16, analyses and critiques 22.

Keyword scan of the 122 WSP-era titles and summaries: no hit for "SAE", "sparse autoencoder", "transcoder", "JSON", "conditional". "structured" hits only IWIBench. "instruction" hits only arXiv 2611.11894.

## 3. The items asked about by name

### 3.1 WSPBench
https://github.com/alexcarterhq/wspbench · https://wspbench.com · repo created 2026-11-03, pushed 2026-11-13.
Read: README **[FULL]**; `reports/selective-repair-sol-full/README.md` and the first 60 lines of `experiments/README.md` **[PART]**. The four-model report was not read.

- Question: does weight access help an agent diagnose and repair a model? The agent first forecasts the effect of candidate edits, then picks a repair from a menu of interventions.
- Targets are small synthetic "controllers" generated per seed (fault mechanisms: coupled gates, cross-layer drift, readout redistribution, sensor routing). Not language models. I did not verify the activation function.
- 80-session experiment, GPT-6.1 Sol: 14/32 faults repaired with behaviour probes, 30/32 with the same probes plus weights (+50.0 pp, 95% interval +28.1 to +71.9), scripted solver 5/32. All arms kept 8/8 clean controls. Cost $2.02 vs $6.34.
- The README says Honnibal's critique shaped the benchmark.
- Not done: language-model targets; an arm with interpreted features; free-form edits (repairs are chosen from a menu).

### 3.2 WSP Toolkit
https://github.com/ZeroTang05/WSP-toolkit · created 2026-11-06, docs dated 2026-11-09.
Read: README, `docs/tool-inventory.md`, `docs/benchmarks.md`, `docs/model-programmer.md`, `PROJECT_STATUS.md` **[FULL]**; `docs/tool-design.md` only grepped.

- An MCP server and Python registry: "34 tool entries and 47 named methods" (not "about 50 tools").
- Includes `extract_features` (sparse autoencoders, two-source joint dictionaries; checked with EleutherAI's public TopK dictionary for SmolLM2-135M) and `trace_circuit` (path patching, ACDC, EAP, Circuit Tracer feature attribution graphs). Also ROME/MEMIT, TIES/DARE, Wanda/SparseGPT, LoRA training, tuned lens, LEACE.
- The only reported benchmark is a TIES merge of three *trained* LoRA updates on Qwen3-0.6B: GSM8K 485/1,319 (36.77%) → 600/1,319 (45.49%). That is not a gradient-free edit.
- Not done, by its own status file: "Complete comparisons against fixed recipes, equal-budget search, and ordinary training remain in progress." No run compares an editor with and without the feature tools. The Qwen3.5 MATH-500 comparison is listed as an open research track.

### 3.3 IWIBench
**IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection.** Chris von Csefalvay, Pranav Doma, Achyuthan Unni Krishnan, Tamás Földi (HCLTech). arXiv 2611.08647 v1, submitted 2026-11-05 (paper dated 4 Nov).
https://arxiv.org/abs/2611.08647 · code https://github.com/hcltech-robotics/iwibench
Read: arXiv HTML **[FULL]** (all sections and appendices A–D); repo README **[FULL]**; `RESULTS.md` grepped for the structured-response and answer-length rows.

Setup:

- Target: `google/gemma-3-4b-it`, revision `093f9f38…`, native BF16, text only. Terminal decoding greedy, 256-token cap.
- Ten task families, three episodes each, one run per episode per editor: 10 configurations × 30 episodes = 300 records.
- Four tools only: `read_weights` (metadata, bounded slices, summaries), `write_weights` (replace, fill, scale, copy, low-rank additive), `run_target` (generation; 8 calls × 8 sequences), `inspect_target` (teacher-forced numerical inspection; 4 calls × 4 sequences, at most four layers and bounded token positions). No shell, no Python, no gradients, no activation steering.
- Budget per episode: 64 turns, 128 tool calls, 262,144 target forward tokens. Editor replies capped at 4,096 tokens.
- Astra (`gpt-6-astra`) ran at **low** reasoning effort.
- Reward = max(0, (e − b)/(1 − b)) × retention of baseline-correct controls.

Results (macro reward): 6.1-sol medium 0.4601, 6.1-sol high 0.4551, Astra 0.4356, 6.1-sol low 0.4250, 6.1-terra 0.2928, 5.6-terra 0.2414, 5.6-sol 0.2148, Nemotron 3 Ultra 0.0548, DeepSeek V4.1 Flash 0.0464, GLM 5.3 0.0135. An eleventh configuration in the repo, Inkling 2 Small (low), scores 0.1660.

The structured-response task:

- Definition: "Extract fields into a shallow JSON object with seeded keys", with no format instruction in the prompt. Grading needs valid JSON with exactly the requested keys and values. Controls: correct field values, and explicit requests for prose must still be obeyed.
- 192 target items (3 episodes × 64). Pristine model 0/192. **Every editor 0/192.** `RESULTS.md`: "Structured-response reward remains zero across all eleven published configurations."
- Astra's attempt, as described in §4.4 and §5.2: "Residual based updates to the first row, which would of course contain the JSON opening braces, resulted in repetitive nonsense. Using contrastive edits healed the JSON formatting issue on both practice names but when the extraction language was changed, all terminal targets failed". The practice prompt "Identify this person's name: …" returned `{"resident": "Owen Brooks"}`; the terminal prompt "Extract the person's name from this note: …" returned the bare name. All 64 names were right and all 64 failed the format.
- So the recorded failure is transfer across prompt wording, not a failure to produce JSON at all.

The embedding claim, exactly as written (§5.1): "The best performing model, Astra, aimed all of its 273 successful write requests at the same tied embedding/output matrix. Indeed, all of Astra's final patches target the embedding/output matrix." The repertoire is described as "copying or blending embedding rows, rescaling lexical rows and adding measured residual directions to output rows". Two points the project plan misses:

- The matrix is *tied*: it is also the unembedding. An edit to the output row of a token fires on whatever final-layer state matches the added direction, not only when a word is in the input.
- "273 successful write requests" counts write calls that executed, not tasks that succeeded.

Other families relevant to us: answer-length control, only 5.6-terra (and later Inkling 2 Small) got positive reward, 2/9 → 3/9; default response language, Astra 0.6373 and every other editor at most 0.2173; terminology preference, at best 1/4.

Explicitly not done or left open: why structured response fails beyond the description above; any MLP or attention edit by the top editor; Python or free activation access for the editor; more than one run per episode. The authors call the result "an early and uneven capability frontier" and ask for tasks that reward globally correct edits.

### 3.4 Exploring Weight-Space Programming (WSP-Scope)
**Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations.** Junru Hu, Cheng Tang, Wei Li, Hongze Zhu, Chun-Mei Feng, Qi Zhang, Yirong Chen, Ming Hu. arXiv 2611.09382 v1, 2026-11-06.
https://arxiv.org/abs/2611.09382 · https://wsp-scope.github.io/ · https://github.com/WSP-Scope/WSP-Scope-Project (README fetch failed)
Read: arXiv HTML abstract, §1–§4, appendix §5–§8 and the start of §9 **[PART]**.

- 16 editing trajectories (Table 2 shows 14, plus two extra Math reruns), GPT-6 Astra at maximum reasoning effort, targets Qwen3.5-0.8B and Qwen3.5-9B, seven benchmarks. Same harness shape as the blog: persistent Python on weights (≤300 analyses), 50 forward requests (generation, or teacher-forced with log-probs and activations). Token embeddings, final norm and LM head fixed.
- Held-out gains on 0.8B: MATH-500 8.35 → 37.80 (+29.45), GPQA Diamond +4.42, IFEval 21.40 → 58.12 (+36.72), SciBench +17.11, SciCode 0. On 9B: IFEval 89.30 → 91.14, SciBench +3.03, Math −0.35.
- Six of 16 trajectories keep the original weights.
- The +29.45 Math edit changes 1,023 values: one column (507) of the layer-23 MLP down-projection, `W' = W − u e₅₀₇ᵀ`, where u contrasts the unembedding readouts of doubt words (Wait, However, But, Maybe, Perhaps) against progress words (Therefore, Thus, So, Now, …). Two reruns give +5.00 and +27.30 with dense rank-1 updates on the same matrix.
- The 0.8B IFEval edit: `O' = O − 7 u(uᵀO)` on attention head 15/7's output block (the head the blog's 25-pass run zeroed). The 9B IFEval edit: layer-16 MLP column 5032 × −32; in development, truncations fell from 58 to 2.
- Cross-task: the Math-edited 0.8B checkpoint gives IFEval 33.58 (+12.18).
- Not done: every IFEval gain here is a loop or truncation fix. No edit installs a specific instruction or format. No variation of what the editor is shown. Stated open: why edits work, joint multi-location edits, effect of development data size, agentic tasks.

### 3.5 Rethinking Weight-Space Programming
Yiqun Zhang. Blog and code, 2026-11-05.
https://zhangyiqun018.github.io/rethinking-weight-space-programming/ · https://github.com/ZhangYiqun018/rethinking-weight-space-programming
Read: README **[FULL]**; blog page grepped for the Experiment 8 passages **[PART]**.

- "Every reported score we re-ran reproduces"; the criticism is about baselines and selection.
- Exp 8: unedited Qwen3.5-0.8B with thinking off scores 44.2% MATH-500, 56.0% GSM8K, 54.7% IFEval. Against the best no-edit decoding setting, Astra's edit adds +6.4 (MATH-500), +11.5 (GSM8K), −2.2 (IFEval).
- A difference-of-means direction with no LLM reaches 41.9% on MATH-500, 79% of Astra's 8.3 → 50.6 gain in their evaluation.
- "its fix improves instruction following too" holds "only in the looping regime": with thinking off the edit costs −4.7 IFEval, with a repetition penalty −8.0; the base model with budget forcing (65.1) beats the edited model (62.9).
- Exp 5: tuned PPO reaches 99.2–114.4 on Seaquest against Astra's 53.1; without the simulator source in the prompt the Seaquest median falls to 0.8. Exp 6: a 30-line script beats the PPO baselines by more than Astra's 0-pass network. Exp 1: a 20-line script gets 3/4.
- Not done: no new edit type, nothing on formats or positions.

### 3.6 The SmolLM2 tool-call experiment
https://github.com/mallikaakash/weight-space-programming · commits 2026-11-05 to 2026-11-06.
Read: README **[FULL]**; `experiments/tool_schema/wsp/results/RUNNING_LOG.md` lines 205–345 plus grep of the earlier part **[PART]**; commit list.

Three stages, of which the project plan knows only the first:

1. Homemade JSON tool-schema set (40 train / 20 test), SmolLM2-360M-Instruct, Claude Code as editor. Run 1: test `strict_ok` 0.20 → 0.20.
2. Budget-50 run on the same set: practice 0.40 → 0.475 (+3/−0), sealed test 0.20 → 0.25 (n = 20). Edit: `v_proj` rows of one KV head scaled ×0.55 in layers 16–17. Later it did not transfer to BFCL.
3. BFCL v4 `simple_python` (300 practice / 100 sealed), Claude Opus 5.5, 100 metered practice forwards. **Practice 123/300 (41%) → 185/300 (62%); sealed holdout 33/100 → 50/100 (+19/−2).** Other categories: `multiple` 0.290 → 0.395; Java, JS, parallel near the floor.

The exported edit ("current4", 32 tensors, `o_proj` and `down_proj` in layers 16–31; embeddings, LM head and final norm untouched) has three closed-form parts. Each projects out or shears along unembedding readout directions, orthogonalised against 311 protected tokens:

- remove late-MLP writes along the readouts of merged paren tokens such as `('` (≈ +44 items of 300),
- weaken copying of the system prompt's `math.factorial` example (≈ +16),
- tilt boolean flags toward `True` (≈ +8).

Random edits of the same geometry cost about −3 and never fix more than 2.

What this is and is not:

- The system prompt with the call format and a demo stays in place. The edit improves compliance under an instruction; it does not replace one. The format is a Python-style call, not JSON.
- The authors: gains are "primarily from fixing formatting bottlenecks", "expression surgery", not new knowledge.
- The edits are keyed on output-token readouts, not on a position. Two attempts at a conditional write failed: a programmed layer-0 neuron acting as a `(` detector fixed 0/9 and broke 3–7 ("early-layer conditional write does not reach the decision"); a `,`-triggered boost of `=` gave +4/−2 and was rejected.

### 3.7 nathanrs/weight-space-lp
https://github.com/nathanrs/weight-space-lp · created 2026-11-06, status box dated 2026-11-07.
Read: README and `docs/WRITEUP.md` **[FULL]** (local copy); `docs/PLAN.md` status box only.

- Shows the blog's 50-pass edit is exactly rank-1: `ΔW = u vᵀ`, with `v_j = −6 uᵀw_j` on 64 columns. At token t it adds `(v·h_t) u`.
- Given u, picks v by a linear program: minimise ‖v‖₁ subject to `v·h_t ≤ −δ` on looping tokens and `|v·h_t| ≤ ε` on productive tokens. Solved with PDHG in about a minute.
- With the agent's direction, 8 neurons per layer match the agent edit (dev 62.5 vs 59.4; base 10.9). With a heuristic direction from detected loops (no LLM; layers still the agent's) about 93% of the way.
- **Selectivity did not help.** "Fewer neurons did better (8 > 32 > 128), and selective edits did no better than broad ones." The chosen neurons fire on 89% of tokens. "A linear probe can spot a loop once it is under way (AUC 0.93) but not at its start (0.73), so an edit that waits for a loop acts too late."
- A constant steering vector equal to the edit's mean push scores 50.8 on MATH-500 (agent edit 50.0, LP edit 53.2, same seed). Half the dose is too weak, double breaks reasoning.
- "Teacher-forced metrics mislead."
- Not done: budget-matched pipeline, choosing layers without the agent, sampling-penalty and L2 baselines, onset-versus-during fits, any other behaviour or model.
- Relevance: this is the closest existing instance of "fit v so the rank-1 edit fires on one set of positions and not on another". It uses an LP where the project uses least squares, and for its behaviour the conditional part turned out not to matter.

## 4. Entries by theme

### 4a. What the editor is shown

| Work | What varied or what was shown | Finding | Read |
|---|---|---|---|
| Original post | raw weights as text (Exp 1, 2, 4, 5); Python on weights (3); + forward passes with activations (6, 8); + attention tool (7) | weights-only failed on Exp 7; succeeded with attention patterns | FULL |
| Honnibal, "The 'Astra can read and write weights to improve Qwen' post is misleading", 2026-11-03, https://honnibal.dev/blog/astra-read-write-misleading | — | raw-weight reading without tools appears only in Exp 1, 2, 4 (17K–172K weights); every LM experiment used Python, and Exp 8 also ran the model | FULL |
| WSPBench (§3.1) | behaviour probes vs probes + weights, synthetic controllers | 14/32 vs 30/32 | README |
| IWIBench (§3.3) | fixed: 4 tools, no Python | top editors wrote only to the tied embedding/output matrix | FULL |
| luizmlo/agentic-weight-editing, pre-paper dated 2026-11-13, https://github.com/luizmlo/agentic-weight-editing (DOI 10.5281/zenodo.24208535) | Claude Opus 5.5 agents on Qwen3.5-0.8B, with or without an "atlas" ("an extra map of the model's internal activity (which neurons fire during loops)") | "In all three pairs of agents with identical instructions, the one without the atlas found the better edit" (38% vs 4%, 67% vs 35%, 65% vs 40%). The authors had predicted the opposite and say three pairs are too few. | README most; paper draft two sections |
| csinva/weight-space-programming-autoresearch, https://github.com/csinva/weight-space-programming-autoresearch | Claude Opus 5.5 predicts the effect of 420 edits to Qwen3.5-0.8B with and without the weight numbers | "only slightly better than without the numbers (+0.06 correlation)" | LIST for this number; the README part I read covers a different paper (iWSP) |
| suayptalha/wsr ("Weight-Space Reading", Kocabay 2026), https://github.com/suayptalha/wsr | reading agent predicts the effect of an edit without running the model; config has conditions `[weights, text]` | paper results are not in the repo; the alphaXiv paper was not fetched | README first part |
| EditLens, https://xxlya.github.io/editlens/ · https://github.com/assassin808/edit-lens (Zhang, Shang, Li; UBC / Vector) | post-hoc analysis with ten tools, including BatchTopK SAEs on Qwen and Gemma Scope 2 on Gemma | Replacing u by one selected SAE feature direction per site gives about 40% GSM8K, against 15% unedited and 57% for the full edit. On Gemma: "Interpretable neuron descriptions and matching SAE features did not translate into a clear accuracy improvement." Steering after layer 15 reproduces the edit's output-score changes with correlation 0.98. | README most; article grepped |
| KidIkaros/wsp-reproduction, https://github.com/KidIkaros/wsp-reproduction | scripts only, no LLM | the loop is semantic, so "token-level repetition SAE features (Neuronpedia's layer-11 set for this model) don't cover it" | README first 70 lines |
| WSP Toolkit (§3.2); oumi-ai/weights-lab, https://github.com/oumi-ai/weights-lab | offer SAE tools; weights-lab's `saes` extra "adds sparse autoencoders and transcoders (`lab.load_sae`)" | no with/without measurement in what I read | README (weights-lab: lines 1–30, 60–80) |
| Weight Whisperer (§4d) | Qwen3.8 as editor, with hints pointing at "the moment reasoning ends and the answer begins" | matched Astra's score | README FULL |
| GPT-AGI/Weight-Space-Programing (WSP-Bench), https://github.com/GPT-AGI/Weight-Space-Programing | reviews the blog logs; plants defects in Qwen3.5-0.8B (LM target) | "The bug was found by running the model and comparing activations, not by staring at weights." Leaderboard not seen. | README first 5 KB |
| arXiv 2611.08070 (Ahmed), 2026-11-05, https://arxiv.org/abs/2611.08070 | Claude Opus 5.5 reads ~125K-weight CNNs as text, no tools | "local arithmetic plus dataset priors"; with CIFAR-10 labels permuted every interpreted class pair is wrong | ABS |
| arXiv 2611.05798 (Cha et al.), 2026-11-04, https://arxiv.org/abs/2611.05798 | weight readers under neuron permutations and rescalings | agreement after canonicalisation does not establish recognition | ABS |
| atharva-naik/auto-weight-space-programming, https://github.com/atharva-naik/auto-weight-space-programming | trains Qwen3.5-4B (LoRA) to apply the logit lens to 128-number slices of its own MLP neurons shown as text | uses training; a reading task | README first part |
| AlexBodner/wsp-scaling; lemonadesaltbagel/how-small-wsp; harlyn1/weight-space-advising-logs | Python vs no tools (induction heads 10/10 vs 0/10); editor size; five editors | — | LIST |

Conclusion for (a): no entry gives SAE or transcoder features to the editor and measures the effect on edit quality. Comparisons of editor inputs on a language-model target now exist (luizmlo's atlas, csinva's numbers vs none, wsr's weights vs text), all small.

### 4b. Formats, structured output, instruction following

| Work | What was done | Numbers | Instruction still in prompt? | Read |
|---|---|---|---|---|
| IWIBench (§3.3) | instruction-free JSON default on Gemma 3 4B | 0/192, all editors | no | FULL |
| SmolLM2 BFCL (§3.6) | readout-projection edits for call formatting | holdout 33% → 50% | yes | README, log |
| Kevin Kuusela, "Hot-swapped weight files let a 4B model beat a 35B MoE with 43% of the memory", 2026-11-05, https://kevin.stream/blog/swap-the-weights | Claude Opus 5.5 writes per-family "weight files" for Qwen3.5-0.8B/2B/4B, hot-swapped by a router | see below | yes | PART: definitions, "What's new", "The edits", "A 5 KB thinking gate"; rest grepped |
| mrtpk/wsp-qwen-tools, 2026-11-12, https://github.com/mrtpk/wsp-qwen-tools | Qwen3.5-0.8B tool use: rank-1 edit of layer-23 `down_proj` (Cursor); two layer-23 neurons, 6,144 weights (Claude) | BFCL-V4 overall, thinking: 20.84 → 21.49 / 22.54; non-thinking 17.68 → 20.29 / 19.55; seed spread 0.5–1.5 | yes | README first part |
| Exploring WSP (§3.4) | IFEval as the target of the search | 0.8B +36.72, 9B +1.85; both loop/truncation fixes | yes | PART |
| Original post; Rethinking (§3.5) | IFEval as a side effect | 21.0 → 62.3; only in the looping regime | yes | FULL / PART |
| luizmlo (§4a) | agents target IFEval with thinking off (base about 55%) | every method looked +5 to +12 in practice and lost on fresh prompts: agent −3.3 ± 4.1, second agent −6.0 ± 3.7, system prompt −6.6 ± 4.2, fine-tuning −7.2 ± 4.3 | yes | paper draft sections |
| desmondzee/wsp, https://github.com/desmondzee/wsp | harness; one line: IFEval on Qwen2.5-0.5B-Instruct held-out 0.351 → 0.406 | no detail in README | yes | README FULL |
| arXiv 2611.11894 (§4c) | an instruction rewording distilled into VLM MLP weights of a robot policy | 27/128 → 106/128 | original instruction stays; the rewording is removed | PART |
| trehansalil (§4c) | task supervision moved into `down_proj` at the answer position | 0.456 → 0.548 | task prompt stays | PART |
| Gen-Verse/WeightCode, manuscript, https://github.com/Gen-Verse/WeightCode | an LLM writes an executable "weight program" that installs a capability | SCAN 0% → 99.8% with 257 of 803,975 weights; BabyAI 0% → 98.4%; ImageNet ResNet-18 noise robustness | not LM formats | README first 6.5 KB; PDF not read |
| GPT-AGI WSP-Bench (§4a) | plants "a rank-1 edit that pushes a language, format or repetition behavior" as a defect to repair | the dev example is "answers in French" | — | README part |
| JonnesLin/weight-space-tax, https://github.com/JonnesLin/weight-space-tax | side effects of the blog edit by dose c = 0…6 | long-code copying 100 → 66% and 98 → 64%; template continuation 37 → 9% and 42 → 10%; **copying JSON 100% at every dose**; GSM8K-train 10 → 55 (65 at c = 5) | — | README first 3.5 KB |
| sudorian/weight-space-implants; sfc-gh-tosmith/wsp-banking77; HomoHabilis (facts: 86% vs LoRA 53%); Say43 (sycophancy 76.8% → 32.3%, system prompt 71.2%); MaceTenth; fderrigo (unauthorised tool calls 1/40 → 13/40); ianbruton (27B tool-reflex, negative); Egzakta-AILab; khordoo/dusty-wsp | skills, facts or behaviours written into LMs; none is an output format | as in the list | — | LIST, or README head for HomoHabilis, fderrigo, ianbruton |

Kuusela's post in detail, because it overlaps most:

- **JSON.** "Asked for JSON with every value in quotes, the plain 0.8B kept writing bare false and null… Opus 5.5 set the output rows of 16 such tokens to zero: 16,384 weights in a 33 KB file. Valid JSON went from 42% to 95% of answers, and the score from 21.5 to 53.8. A trained LoRA adapter reached 92.8". This is an LM-head edit, with the instruction in the prompt.
- **"Add an instruction without the words."** "Run a prompt with and without an instruction, take the difference in the residual stream, add that vector back at a few layers, and the model behaves as if it had read the instruction." The 4B math file carries "be concise, plain-text arithmetic, no LaTeX, no headings, don't verify the result" as steering vectors after layers 7, 11, 15, 19, 23, plus 38 renamed output rows ("Verify" → "Final"): GSM8K 86.5 → 90.5 (CI +1.0 to +7.0). The vector is constant over positions and is installed by a wrapper on the module output, so it is a steering vector shipped as a file, not an edit of an existing matrix.
- **Extraction.** Rank-128 edits on the MLP down-projection of layers 29–31, "in closed form, with one ridge regression and one SVD per layer on the model's own activations, and no gradients"; +25.5 points on CoNLL.
- **Labels.** 4–6 weights correcting label biases: 59.0 → 65.5, where LoRA reaches 94.0. "Six weights can correct a bias. They can't teach a task."
- Summary line: "The files help where the problem is how the model answers: the format, the structure, when to stop."
- Three families (labels, extraction, reading) got a trained file in the final system "because training teaches a task that writing couldn't". Written files helped an unseen task of the same family in 1 of 7 families on 0.8B and 2 of 7 on 2B.
- Several of these edits touch the LM head or add a bias, which the blog's Experiment 8 rules forbade.

### 4c. Conditional or position-specific edits

**Knowing Is Not Committing** (Dave Uhlemann, "NeoCortex"). Report v0.4, not peer-reviewed. Repo created 2026-11-08, pushed 2026-11-13.
https://github.com/NeoCortex0815/knowing-is-not-committing · DOI 10.5281/zenodo.24058693
Read: README first 5.5 KB; report v0.4 TL;DR items 1–10, §3.3, §3.6, §3.8 **[PART]**.

- Writes a conditional reflex into 16 rarely active SwiGLU neurons of layer 15 of Qwen3.5-0.8B: "reflex = effector × (condition ∧ context)". `gate` holds a truth probe ("this value is true"), `up` holds a context probe ("a result just ended"), `down` holds the effector, the mean gradient ∂ log p(`</think>`)/∂h at wanted positions. The SwiGLU product gives the AND with no runtime hook.
- The context is a *position defined by context* (the end of a candidate answer inside the reasoning), not a word.
- The direction comes from gradients and the two conditions from fitted linear probes, so it is training-free only in the sense of no weight training.
- Offline: at correct candidate ends p(`</think>`) goes from about 10⁻¹² to about 1.0 in 6/8; unchanged at wrong candidate ends and ordinary tokens (0/8 each).
- Their mandatory offline checks include "last prompt token (first generated token): stay at base level". The first build leaked there: 9–10 of 30 runs closed thinking at once. The fix was a sign rule for the threshold (SiLU is negative between about −4 and 0).
- Results. Development set (n = 60): 33 vs 26 correct against the same model without the trigger, 64% fewer tokens on problems both solve. Fresh MATH-500 problems 161–300: closes right after the correct value in 65% of opportunities (MIT edit 36%, p = 0.0001), but accuracy 63 vs 82 of 140 for the MIT edit (−13.6 points, CI −21.4 to −6.4).
- Why it falls short, in their words: "A forced close is not an inner close". After a trigger close the model reopens reasoning in the answer part (a second `</think>`) in 77% of closes on fresh problems. "the effector must fit the phase, and it must leave a state, not just a token."
- An *unconditional* closing push costs instruction following: IFEval 61/120 against 74 without the push (p = 0.041) and 87 for the MIT edit.
- Also: a layer-13 truth probe has AUC 0.96 on practice data; the model keeps thinking a median of about 2,000 tokens (T = 0.6) to 6,000 tokens (T = 1.0) after first writing the correct answer.
- Not done: any model other than Qwen3.5-0.8B ("transfer to other models untested"); any behaviour other than closing the thinking block; format instructions.

**Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy's Misreading.** Ali-Adeeb Abbas, Esmaeil Seraj. arXiv 2611.11894 v1, 2026-11-09.
https://arxiv.org/abs/2611.11894 · https://github.com/adeeb10abbas/weightsmith
Read: arXiv HTML §1–§6, Appendix A and the start of B **[PART]**; appendices C–H not read.

- Target: π0.5 vision-language-action policy (18-layer PaliGemma VLM + action expert), zero-shot, RoboLab simulation. Not a text-generating LM.
- Failure: told "put the cube then the banana in the bowl", it drops the cube early. A rewording ("Put the cube in the bowl then put the banana in the bowl") works with unchanged weights (22/32 vs 3/32).
- Method ("closed-form instruction distillation"): at each VLM layer 0–16, 72 keys = mean input to the MLP down-projection under the *original* instruction at 72 fixed positions (40 shared prompt positions: BOS, "Task:", robot-state tokens; 32 image cells), averaged over 6 recorded failure decisions. Values = the layer's residual increment under the rewording minus under the original. Update `ΔW = Kᵀ(KKᵀ + λI)⁻¹V`, rank ≤ 72 per layer. They call it the MEMIT closed form "aimed at a rewording"; "Context distillation trains a model to act as if a prompt were present, whereas Eq. (1) needs no training."
- Results: held-out 27/128 → 106/128 (81 fixed, 2 broken, p = 7.2×10⁻²²). A random update of the same layers, rank and singular values: 9/64. Explicit wording with no edit: 57/64 against the edit's 50/64. A variant with image-cell keys only in layers 8–16 (rank ≤ 32, "0.16% of the parameter count") scores 102/128. Transfers to two other tasks.
- Which positions matter was tested by ablation: "Copying the mean shift at the state tokens flipped none, and per-token shifts at the 40 prompt and state positions left 8 to 15. Adding the 32 image cells removed all 21. Image cells alone suffice".
- Where to edit matters: VLM 55/64; action expert 20/64; action head least-norm 0/64; a blind editor's gated rank-1 head edit (1,024 values, gated to premature-release states) 30/64.
- Seven blind Claude Opus editors, one task each: one passed held-out scoring.
- Stated limits: needs a known-good instruction as input; "low-rank but not sparse. It changes 13.4% of the policy's stored values, and the 0.16% variant still changes 7.4% once applied"; one checkpoint, simulation only; one task for the tool.
- Not done: language-model outputs, format instructions, autoregressive answer positions (VLM attention is bidirectional and no text is generated), rank-1 edits, removing the instruction altogether, a count of how many position-specific edits are needed.

**trehansalil/weight-space-programming.** https://github.com/trehansalil/weight-space-programming · listed 2026-11-15.
Read: README **[FULL]**; `results/seed2/RESULTS.md` first 60 lines; docstring of `wse/edit.py` **[PART]**.

- Replicates "Nandakishor M's 'Weight space programming will replace fine tuning' post (Qwen3.5-0.8B, edits to `mlp.down_proj` at layers 19, 21, 23)". **I could not find that original post** (two web searches).
- MEMIT-style: "K keys: the input of down_proj at the answer position of each training prompt". One linear solve per layer, `dW = Rᵀ K (KᵀK + λ n C)⁻¹`, with a covariance term from WikiText and protected tasks. The residual targets are found with Adam steps, so only the solve is closed form ("That is the part the post's 'no gradient descent' wording does not cover").
- Seed 2, eight aimed-at classification tasks, first-token letter read-out: plain 0.456 → edit 0.548 (+0.092, paired CI +0.055 to +0.130); LoRA 0.443; 8-shot ICL 0.637; 64-shot ICL 0.666. Unseen tasks 0.554 → 0.467. 524,288 weights changed (0.070%).
- Not done: instructions or formats; position analysis; rank-1.

Other conditional or keyed edits:

| Work | Condition | Result | Read |
|---|---|---|---|
| Kuusela "thinking gate" (§4d) | ridge regression on the final hidden state, "answer is settled" vs "still working" | one `</think>` output row + bias | PART |
| Weight Whisperer (§4d) | MLP activation pattern v that marks a loop | rank-1 toward `</think>` | README FULL |
| weight-space-lp (§3.7) | v fitted by LP to fire on looping tokens only | selective no better than broad | FULL |
| Exploring WSP reruns (§3.4) | reader v from MLP activations just before a late "Wait" onset; writer u orthogonal to protected token readouts | +5.00 and +27.30 | PART |
| SmolLM2 (§3.6) | `(` detector neuron in layer 0; `,`-readout trigger | failed; rejected | log |
| tmestery/circuit-injection, https://github.com/tmestery/circuit-injection | new trigger–response skill by a closed-form edit to a late MLP projection of GPT-2 small | held-out trigger accuracy 0% → 90%, false-trigger rate 10% | README abstract |
| IWIBench (§3.3); arXiv 2611.09376 (Zhang, Wang, Lin; 65 token-embedding rows of π0.5, held-out paraphrases 86.3% → 92.0%), https://arxiv.org/abs/2611.09376 | the token itself (embedding or tied output rows) | — | FULL; ABS |
| Original post Exp 4 | a small added path that "switches off" once the wheel passes the strip | 0% → 100% | FULL |
| Admiral (gated "steer home" circuit in a MAPPO policy), https://github.com/faresboudelaa/admiral | state gate in an RL policy | +0.36 score against the hard opponent | LIST |
| W-ISA, arXiv 2611.13232, 2026-11-09, https://arxiv.org/abs/2611.13232 | "Activation-based match-action rules connect measured model states to intervention selection" at run time | a runtime for rank-1 programs; no behaviour result | ABS |
| "The Mathematics of Weight-Space Programming" (Noguer Alonso), Zenodo 10.5281/zenodo.23933590 | conditions under which a selective edit to the final feed-forward module is possible; certificates of infeasibility | editors are algorithms, not LLMs; GPT-2 CounterFact | LIST |

### 4d. Stopping or shortening the thinking phase

| Work | Edit | Numbers | Read |
|---|---|---|---|
| Original post | remove the loop direction (§1) | share of MATH-500 answers cut at the limit 93.0% → 33.1% | FULL |
| Weight Whisperer, 2026-11-11, https://github.com/Prayas-Agrawal/weight-whisperer | open-weight Qwen3.8 as editor. `W ← W + α·e(</think>) vᵀ` on the layer-23 MLP output of Qwen3.5-0.8B; v contrasts MLP activations on looping and productive reasoning. "over 99% of each published edit is this one component". No forward-pass budget. | MATH-500 7.70% → 50.40% (run 1), 51.55% (run 2); 4 samples × 500 | README FULL |
| luizmlo (§4a) | at most four rank-1 edits `w_j ← w_j − α(uᵀw_j)u`, neurons found by reading weights "through the vocabulary". "make 'Wait' less likely, and make `</think>` more likely" | locked GSM8K (200 each, 12,288-token budget, llama.cpp): base 15.5%; Astra's edit 53.5% (65.0% with 20k tokens); thinking off 58.5% / 58.0%; agent edits 61.5–68.0%. Tokens per answer 11,142 → about 4,500–4,800 (Astra's edit 8,877). 148 MATH-500 problems: 47% vs 21% for Astra's edit at the same budget. LoRA 29%, scripted search 27% on practice. | README most |
| NeoCortex (§4c) | truth-gated `</think>` reflex in 16 neurons; "ERGORE" = own loop direction + conditioned closing push | ERGORE 79 vs 77 of 160 against the MIT edit with 13% fewer tokens; fresh problems 66 vs 82 of 140 | PART |
| Kuusela (§4b) | **Thinking gate** on Qwen3.5-4B: "Opus 5.5 ran the plain 4B on 64 training problems and marked the point in each trace where it already had its final answer. A ridge regression on the model's final hidden state then learned to tell 'the answer is settled' from 'still working', and that regression became one output row and one bias for `</think>`: 2,561 weights". About +14 on the `</think>` logit when settled. | 3,072-token budget. GSM8K 75.0 → 94.0, tokens per answer 1,841 → 1,020, out of budget 46 → 1; right-when-answered 97.4% → 94.5%. SVAMP (unseen) 83.5 → 93.5. "The gain comes from finishing, not from better reasoning." Blunt version on 0.8B (zero six "Wait" rows, `</think>` row × 1.6): math 49 → 55, general knowledge 52 → 23. | PART |
| conceptcountry, "Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B", working paper v1, 2026-11-08, https://concept.country/research/weight-space-programming · https://github.com/conceptcountry/weight-space-programming | directions read off residual activations on a MATH train split, written as a rank-1 additive term into the down-projections of nine alternating layers | GPQA Diamond +4.3 (CI +1.8 to +7.1) with 30.0% fewer thinking tokens; LiveCodeBench v6 +13.0 with 29.2% fewer; IFBench +3.3 with 54.1% fewer (control clause borderline). A norm-matched random edit does nothing. | README FULL; web abstract; PDF not read |
| Exploring WSP (§3.4) | "doubt minus progress" readout direction subtracted from one MLP column | +29.45 on MATH-500 | PART |
| Tree-Ising, arXiv 2611.14210, 2026-11-10, https://arxiv.org/abs/2611.14210 · https://github.com/TheQuantumGOAT/Tree-Ising | no LLM: per-row QUBO over BF16 sign flips and exponent shifts; objective "weakens a loop direction, strengthens a concluding direction" (from the last 64 thinking tokens before `</think>`) | MATH-500 54.6% vs 51.7% for the agent edit; 4 forward-pass requests; Qwen3.5-2B greedy GSM8K 21.6% → 69.4% | ABS + grep |
| weight-space-lp (§3.7) | LP edit; constant steering vector equivalent | — | FULL |
| Decomposing the Gain (§4e) | no-edit baseline: cut the loop and force an answer | MATH 34.0%, GSM8K 57.0% | README, blog |
| Xananthium "Calmer Qwen", https://github.com/Xananthium/Qwen3.8-Weight-Space-Programming | sparse corrections on Qwen3.8-27B 4-bit to reduce repetitive checking | five-question panel: 23,831 → 18,481 tokens; exploratory, uses a learned correction | README first part |
| ajaygrover (stacking a second edit: 52.2% on MATH-500, 31% fewer tokens than Astra's edit); jazzpher/qwen35-ramble-edit; Exorust/wsp-sql (loop rate 90.7% → 26.0%); clavis-systems; Egzakta-AILab | loop edits and reruns | as in the list | LIST |

The project plan's experiment 9 (read "the answer is already there" from the hidden state, then a rank-1 `down_proj` edit toward closing the thinking block) is closest to NeoCortex's reflex and Kuusela's gate. Both found the same two things: the gain comes from finishing, and accuracy on answered problems drops.

### 4e. Critiques and audits

| Work | Main point | Read |
|---|---|---|
| Honnibal (§4a) | reading raw weights and repairing with code are different techniques; the LM result is ordinary ML engineering done by an agent | FULL |
| Rethinking (§3.5) | weak baselines; thinking off gives 44.2%; difference of means gives 79% of the gain | README, blog part |
| Decomposing the Gain (Giraldo Márquez, Gómez), https://github.com/JKUL-png/decomposing-the-gain · https://jkul-png.github.io/decomposing-the-gain/ | Replicates (7.8% → 51.2% on 100 problems). Unedited, thinking off, 32k tokens: 51.5%. Placebo edit of the same size 11.0%. At temperature 1.0 (the model card's value) the gain falls from +43.5 to +11.0 points and IFEval reverses (unedited 7.0 above edited). With thinking off the edit adds nothing. | README FULL; blog most |
| GPT-AGI review (§4a) | finish rate 7% → 65% while accuracy among finished answers fell 93% → 71% | README part |
| EditLens (§4a) | "Few changed weights do not imply few changed computations": downstream attention effects, states pushed past the normal range, new generation failures | README, article grep |
| JonnesLin tax (§4b) | every dose that improves math hurts copying | README part |
| arXiv 2611.08070; arXiv 2611.05798 (§4a) | weight reading leans on priors and coordinates | ABS |
| NeoCortex (§4c) | "Sampling protocol decides half of the effect": the MIT edit scores 55% with vLLM-style sampling and 25% with llama.cpp defaults on the same 40 problems | PART |
| luizmlo (§4a) | proxy scores given to agents were gamed; thinking off solves 58% at a third of the tokens | README, paper part |
| hamley241/wsp-harness-public; hamley241/weight-space-horizon; Flecart/weight-programming; goddoe/weight-space-programming; OpenWSP (its own one-shot Astra rerun of the car repair: 0/1,024); R&D World news; clavis-systems | bounded edits lose to projected gradient descent; episode caps inflate scores; a Python script matches the Breakout network; 1 of 4 fresh car answers works | LIST |
| LessWrong, shawnghu, 2026-11-10, https://www.lesswrong.com/posts/wg3K6xFFTdM6RmRgn/claude-fable-5-2-can-do-weight-space-programming-about-as | Claude Fable 5.2 reproduces the qualitative results; with 10 passes it matches or beats Astra's 50 on the math benchmarks | first part |

### 4f. WSP-era entries with no bearing on (a)–(d)

Read as **[LIST]** only unless marked above: all 14 robotics entries except the two arXiv papers discussed; the 26 "other domains" entries (YOLO, EuroSAT, EEG/ECG, MedMNIST, SDXL and SD 1.5, genomics, recommender systems, ASR, Boltz-2, benefit formulas, SpliceAI, DNABERT-2, WeightLoop arXiv 2611.07319 [ABS], and others); 7 games entries; 2 quantum entries; the remaining tools and agent skills; replications. Boom5426's list adds W-ISA (above), NCP arXiv 2611.13236, Physics2Weight arXiv 2611.12954, a diffusion conjecture paper arXiv 2611.04268, "Adapters as Data" arXiv 2611.07775 and TensorMutation arXiv 2611.09650; none was opened except W-ISA's abstract.

### 4g. URLs and dates of entries cited above only from list summaries

Dates are the list's. All are GitHub repositories unless noted; none was opened beyond what the tables above say.

| Entry | URL | Date |
|---|---|---|
| ajaygrover, "Independently Discovered Weight Edits Compose" | https://github.com/ajaygrover/weight-space-programming-stacking | 2026-11-07 |
| jazzpher/qwen35-ramble-edit | https://github.com/jazzpher/qwen35-ramble-edit | 2026-11-10 |
| Exorust, "Weight-space programming, but for SQL" | https://github.com/Exorust/wsp-sql | 2026-11-07 |
| clavis-systems, GSM8K rerun at 16K | https://github.com/clavis-systems/astra-gsm8k-transformers-16k | 2026-11-07 |
| Egzakta-AILab/wsp-lab | https://github.com/Egzakta-AILab/wsp-lab | 2026-11-08 |
| sudorian/weight-space-implants | https://github.com/sudorian/weight-space-implants | 2026-11-12 |
| sfc-gh-tosmith, Banking77 | https://github.com/sfc-gh-tosmith/wsp-banking77 | 2026-11-07 |
| HomoHabilis, local replication kit | https://github.com/HomoHabilis/weight-space-programming | 2026-11-07 |
| Say43, sycophancy repair | https://github.com/Say43/Weight-Space-Programming | 2026-11-04 |
| MaceTenth, replication and GPT-2 fact insertion | https://github.com/MaceTenth/weight-space-programming-replication | 2026-11-06 |
| fderrigo, Pinocchio and tool-call patches | https://github.com/fderrigo/weight-space-programming-experiments | 2026-11-06 |
| ianbruton, follow-up in three parts | https://github.com/ianbruton/weight-space-programming-followup | 2026-11-11 |
| khordoo/dusty-wsp | https://github.com/khordoo/dusty-wsp | 2026-11-09 |
| AlexBodner/wsp-scaling | https://github.com/AlexBodner/wsp-scaling | 2026-11-05 |
| lemonadesaltbagel/how-small-wsp | https://github.com/lemonadesaltbagel/how-small-wsp | 2026-11-15 |
| harlyn1, college advising | https://github.com/harlyn1/weight-space-advising-logs | 2026-11-05 |
| hamley241, bounded repair | https://github.com/hamley241/wsp-harness-public | 2026-11-07 |
| hamley241, episode caps | https://github.com/hamley241/weight-space-horizon | 2026-11-04 |
| Flecart, stress tests | https://github.com/Flecart/weight-programming | 2026-11-03 |
| goddoe, independent reproduction | https://github.com/goddoe/weight-space-programming | 2026-11-07 |
| OpenWSP | https://github.com/openwsp/openwsp | 2026-11-03 |
| mfaseehuddin/wsp | https://github.com/mfaseehuddin/wsp | 2026-11-06 |
| Xananthium, Calmer Qwen | https://github.com/Xananthium/Qwen3.8-Weight-Space-Programming | 2026-11-05 (repo creation) |

## 5. Checks against the project plan (`docs/plan_related_work.md`, `PLAN.md`)

| Plan statement | What I found |
|---|---|
| SmolLM2: JSON tool-call accuracy stayed 20% → 20%; no successful edit | Out of date. That was run 1 (n = 20). Later: 0.20 → 0.25 on the same set, then BFCL holdout 33% → 50% with three closed-form readout edits (§3.6). Still with the format prompt in place. |
| IWIBench: all 273 of Astra's successful weight writes were on the embedding matrix; such edits fire when a particular word comes in | The paper says "tied embedding/output matrix" and describes edits to *output* rows with measured residual directions. Astra's JSON attempt edited the output side, worked on practice prompts and failed on reworded prompts. "273 successful write requests" are executed write calls. The plan's explanation of the failure needs this correction. |
| IWIBench tool set includes viewing the hidden state at a position | Partly. `inspect_target` is "teacher-forced numerical inspection" limited to 4 calls × 4 sequences, at most four layers and bounded positions. There is no Python. Astra ran at low reasoning effort with one run per episode. |
| WSP Toolkit: about 50 tools with an SAE feature index | "34 tool entries and 47 named methods". It loads SAE dictionaries and Circuit Tracer graphs; no index-on/off comparison (as the plan says). |
| WSPBench: small circuits, 14/32 vs 30/32 | Confirmed (GPT-6.1 Sol; scripted solver 5/32). "ReLU" not verified. |
| Exploring WSP: 16 runs, 6 kept the original weights | Confirmed. |
| No list entry gives SAEs or transcoders to the editor and measures the effect | Still true. But tools now ship SAE and transcoder access, EditLens rebuilt part of the edit from SAE feature directions (about 40% vs 57%), and luizmlo's "atlas" comparison found more interpretability data hurt. |
| Comparing what the editor is shown has only been done on small networks | No longer exact: luizmlo (atlas), csinva (numbers vs none), wsr (weights vs text) use LM targets. |
| No result yet makes a model keep an output format at instruction level by weight edits alone | Still true for the instruction-free case. With the instruction present, Kuusela reports valid JSON 42% → 95% and SmolLM2 reports 33% → 50%. Kuusela also carries an instruction as a constant steering vector. |
| Experiment 9 (stop thinking once the answer is written) as the largest possible benefit | The plan cites no prior work. Kuusela, Weight Whisperer, NeoCortex and luizmlo already did versions of it (§4d). |
| The closest work is "Transmuting prompts into weights" | Inside the WSP ecosystem the closest are arXiv 2611.11894 and trehansalil (§4c), which the plan does not mention. |
| awesome-wsp had 56 entries on Nov 5 | Consistent with the commit history; 122 WSP-era entries on Nov 15. |

## 6. What I could not find or did not read

- The post by "Nandakishor M" that trehansalil replicates. Not found.
- The alphaXiv papers "Weight-Space Reading" (Kocabay) and "Interpretable weight-space programming" (iWSP); the WeightCode PDF; the concept.country PDF; Zenodo preprints; X threads.
- The WSP-Scope repository README (fetch failed), the OpenWSP README (125 KB, not read), the WSPBench four-model report, IWIBench's per-episode traces on Hugging Face.
- "The Coupling Tax", cited by Kuusela, was not identified with certainty (possibly JonnesLin's "Tax" post).
- Which repository the plan means by "wsp" (hand-built circuits inserted into an LLM): mfaseehuddin/wsp and tmestery/circuit-injection both fit; not resolved.
- About 60 of the 122 WSP-era entries were read only as list summaries.

## Gaps relevant to our project

What nobody in this ecosystem has shown, as far as the material above goes:

1. An output-format instruction (JSON-only or similar) moved into weights with **no instruction left in the prompt**, at compliance near the instructed model, on held-out prompt wordings. IWIBench is 0/192; the two successes keep the instruction.
2. A weight edit **conditioned on the answer-start position** of a text-generating LM. Existing conditions are: a token (IWIBench, robot embedding rows), an output-token readout (SmolLM2), a loop state (weight-space-lp, Weight Whisperer, Exploring WSP reruns), a candidate-answer end inside the reasoning (NeoCortex), "answer settled" at the LM head (Kuusela), fixed input positions of a VLM prompt (2611.11894), and the last prompt token used as a MEMIT key for classification (trehansalil).
3. A **per-position measurement** (KL between instructed and uninstructed runs at each answer position) used to count how many conditional edits an instruction needs.
4. A **predictor of which instructions transfer**. IWIBench shows uneven task difficulty and says success in one family "correlated poorly" with others, but offers no predictor. Kuusela's "format, structure, when to stop" versus "knowledge" is the nearest informal statement.
5. An **explanation of the IWIBench structured-response failure** tested by experiment. The paper only describes it ("too brittle to withstand the change").
6. A controlled comparison of **editor inputs with SAE or transcoder features**, or with a "where do the two models diverge" tool, on an LM target.
7. MLP rank-1 format edits on **Gemma 3 4B**: IWIBench's top editors never left the tied embedding/output matrix.
8. Comparison of a position-conditional edit against the same vector applied at all positions for a *format* instruction. For loops, weight-space-lp found the conditional part unnecessary; for formats this is untested.

## Things that overlap with or threaten our claims

1. **Kuusela (2026-11-05).** Already has "instruction difference in the residual stream → added back without the words" (as a constant steering vector in a hot-swapped file), a JSON validity fix by weight edit, a closed-form low-rank extraction edit, and a hidden-state-gated `</think>` edit. A reader will ask what a position-specific rank-1 edit adds over his steering-vector file. His edits use the LM head and added biases; ours would need to state that it stays inside existing MLP matrices.
2. **arXiv 2611.11894.** Uses the title phrase "Distilling Instructions into Weights", is training-free and closed-form, uses keys at specific positions, includes a position ablation and a location comparison, and shows a random-direction control. The differences (robot policy, bidirectional prompt positions, rewording rather than removal of the instruction, rank ≤ 72 × 17 layers, 13.4% of values changed) must be stated explicitly.
3. **NeoCortex.** A conditional reflex "effector × (condition ∧ context)" written into MLP neurons, with the context defined by position-in-context rather than a word. This weakens "the new part is using a place in the answer as the condition instead of a word". It also reports the failure mode we should expect in experiment 9 (forced close gets reopened; unconditional push costs IFEval) and the answer-start leak.
4. **Weight Whisperer and luizmlo.** Rank-1 MLP edits toward `</think>` already match or beat the published edit on accuracy and tokens. Experiment 9's bar ("accuracy similar to the published edit, tokens halved") has roughly been met by others: luizmlo reports about 4,500 tokens per answer against 8,877 for Astra's edit.
5. **weight-space-lp.** Same algebra as ours (`ΔW = u vᵀ`, v fitted to separate two token sets). Its negative result on selectivity is a prior against the conditional part mattering; we need the all-positions baseline.
6. **trehansalil / the Nandakishor post.** MEMIT keys at the answer position of the prompt on Qwen3.5-0.8B `down_proj`, compared with ICL and LoRA on equal supervision. The edit lost to 8-shot ICL and hurt unseen tasks. If the original post is as described, "answer-position keyed MLP edit that replaces a prompt" has been claimed publicly.
7. **SmolLM2 BFCL and IWIBench together** change the framing of our third claim. Editors do find format edits when they have Python and many forward passes (SmolLM2), and IWIBench's failure happened under four tools, 16 inspected sequences and low reasoning effort, on a task that also needs a seeded key name. "Editors fail at format because embedding edits cannot target the answer start" is not supported as stated: Astra's attempt was on the output side and failed on transfer across wordings.
8. **Critiques of the regime.** Thinking off recovers most of the blog's gain (Rethinking, Decomposing the Gain, luizmlo), the effect depends on temperature and sampler (Decomposing, NeoCortex), and the edit has measurable side effects (JonnesLin, EditLens, Rethinking's IFEval −4.7). Any result of ours on Qwen3.5-0.8B with thinking on will be read against these baselines, and for a format instruction the obvious baseline is the instruction itself, as the plan already notes.
9. **luizmlo's atlas result and EditLens's SAE result** are early evidence against "showing the editor interpreted features helps". The plan expects the same; these can be cited as prior indications, not as our finding.
10. **Speed of the field.** 122 WSP-era entries in 13 days, several per day touching format, stopping and conditional edits. The gaps above were open on 2026-11-15 only as far as this search reached.
