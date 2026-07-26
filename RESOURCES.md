# Where else to practise this

Everywhere else worth your time on the same material, in one list. Grouped by whether
it checks your work, because that is the part that is scarce — of the resources below,
**19 give you an automatic verdict** and the rest do not.

Every link was fetched when this was written and then HTTP-checked separately. Dates are
the most recent activity that could be verified; where a project is archived or dormant it
says so, because a dead project is still worth reading and worth knowing is dead.

This is the practical cut. [`LANDSCAPE.md`](LANDSCAPE.md) covers the same ground organised
by this bank's own areas, with a verdict on where this bank does and does not add anything.

Last checked **2026-07-26**.

| | count |
|---|---:|
| Graded automatically | 19 |
| Ships tests you run yourself | 21 |
| Reference code to read | 32 |
| Reading, tools and reference | 63 |
| **Total** | **135** |

116 of 135 are free. Paid and freemium entries say so on the line.

## Graded automatically

You submit, a machine gives you a verdict. The scarcest category by far.

### [CppQuiz.org](https://cppquiz.org/)
`free` · `puzzle set`  
190 questions · last activity unknown (actively used; no abandonment signal, but no single pinned update date)  
Deep C++

You're shown a short real C++ snippet and must predict its exact output, or flag it as a compile error, unspecified behaviour, or UB; submitting scores you immediately (1 point correct, penalties for hints/wrong attempts). Written by Anders Schau Knatten with input from Olve Maudal and other ACCU members.

### [Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)
`freemium`  
6 confirmed relevant problems (self-attention, multi-head attention, masked self-attention, layer norm for sequences, positional encoding, GQA) out of 100+ site-wide · last activity unknown  
LLM internals

LeetCode-style browser platform with in-browser test-case grading. Some content sits behind an unconfirmed-price Premium tier.

### [deep-ml.com](https://www.deep-ml.com/problems)
`freemium`  
100+ problems (site's claim) · last activity unknown (live product, not a repo)  
Algorithms from scratch

Browser-based LeetCode-style ML site: write Python in an in-browser editor, get pass/fail against hidden tests instantly. Catalog (confirmed via secondary sources, not directly scrapeable since JS-rendered) includes linear regression via normal equation/gradient descent, k-NN, decision trees, k-means, PCA, SVD alongside DL/NLP/CV problems.

### [Exercism — C++ track](https://exercism.org/tracks/cpp)
`free` · `exercise repo`  
100 exercises across 19 concepts · last activity 2026-07  
Deep C++

Exercism's general-purpose C++ track: free, automated test-suite grading per exercise, plus optional human mentor review afterward.

### [Exercism — Python track](https://exercism.org/tracks/python)
`free`  
146 exercises / 17 concepts; only ~4-5 (Descriptors, Iterators, Context Manager Customization, Class Customization, a Generators concept exercise) fall inside this area · last activity 2026-07  
Deep Python

General-purpose Python practice track: write a solution, an automated test suite runs on submit, optional human mentor review afterward.

### [GPU MODE / KernelBot](https://www.gpumode.com/)
`free`  
8 problem series/competitions (PMPP practice set plus sponsored contests: AMD $100K, AMD $1.1M, NVIDIA Blackwell NVFP4, BioML, Helion hackathon, linear algebra) · last activity 2026-07  
GPU / CUDA

Competitive-kernel wing of the GPU MODE community (formerly CUDA MODE). Submit via Discord bot or the popcorn-cli, run on real sponsored/donated GPUs, ranked on a public leaderboard. Companion repo gpu-mode/reference-kernels holds the problem sets.

### [guessthedis](https://github.com/cmyui/guessthedis)
`free` · `puzzle set`  
3 GitHub stars; 60+ built-in functions across difficulty tiers, needs Python 3.10+ · last activity 2026-04 (dependency-bump commit; small but not abandoned)  
Deep Python

A terminal game: you're shown a Python function and must type out its bytecode instructions line-by-line from memory; it checks your answer against the real dis output.

### [HackerRank — C++ domain](https://www.hackerrank.com/domains/cpp)
`freemium`  
small — e.g. 5 problems in the Inheritance subdomain alone; other subdomains (Classes, STL, Debugging, Other Concepts) are comparably small · last activity unknown (live commercial platform, not a repo; problems look untouched for years)  
Deep C++

Introductory OOP/C++ problems: classes, single/multi-level inheritance, virtual functions and abstract classes, basic function/class templates.

### [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
`free` · `benchmark or leaderboard`  
250 tasks (100 Level-1 single-op, 100 Level-2 fused-op, 50 Level-3 full-architecture), 1,157 GitHub stars · last activity 2026-03  
GPU / CUDA

Stanford Scaling Intelligence Lab benchmark built to answer 'can an LLM write a fast CUDA/Triton kernel' — scores correctness plus a fast_p speedup ratio against a PyTorch reference on real GPU wall-clock time. A human can run it against their own Level-1 kernels, but the harness and surrounding papers target automated kernel generation, not a learner's curriculum.

### [LeetGPU](https://leetgpu.com/)
`freemium`  
70+ challenges (counted on the live challenges page: ~19 Easy, ~46 Medium, ~13 Hard) · last activity unknown  
GPU / CUDA

Browser CUDA/Triton/PyTorch/Mojo/CuTe-DSL/JAX judge. Verified via live browser navigation: homepage now says 'Execute high-performance GPU programs instantly on real hardware in your browser' — a shift from its 2025 launch, which ran purely on a CPU emulator with 'functional' and 'cycle accurate' (architecture-modeling) modes per its Show HN post. Has a CLI, global leaderboard, open problem-contribution repo, and a visible 'Pro' tier in-nav.

### [LeetGPU — challenge set](https://leetgpu.com/challenges)
`freemium`  
~90 challenges total; 14 directly on attention/RoPE/KV-cache (rotary-positional-embedding, sliding-window-self-attention, casual-attention, multi-head-attention, grouped-query-attention, linear-attention, decaying-causal-attention, attn-w-linear-bias, softmax-attention, int8-kv-cache-attention, gpt2-block, llama-transformer-block, speculative-decoding-verification, top-p-sampling) · last activity 2026-07-24  
Attention and KV cache

Browser IDE where you implement a kernel (CUDA/Triton/PyTorch/Mojo) against a fixed signature; the site runs it against hidden tests and a timing score, all on CPU-emulated GPU execution for free (real-hardware tier is paid).

### [Machine Learning Specialization (Andrew Ng / DeepLearning.AI + Stanford)](https://www.coursera.org/specializations/machine-learning-introduction)
`paid` · `course with labs`  
3 courses; named practice labs for decision trees, anomaly detection, k-means, PCA confirmed · last activity 2022 revision, still the live served version in 2026  
Algorithms from scratch

Current Python/NumPy successor to Andrew Ng's original Octave ML course. Programming labs implement linear/logistic regression and basic neural nets from scratch, plus hands-on labs for decision trees/ensembles, k-means, PCA, and anomaly detection, graded by Coursera's own grader for a certificate.

### [NVIDIA DLI: Optimization and Deployment of TensorFlow Models with TensorRT (+ Coursera guided-project version)](https://www.coursera.org/projects/tensorflow-tensorrt)
`paid` · `course with labs`  
one course (DLI, ~8h, ~$90) / one guided project (Coursera, 1.5h) · last activity unknown, no visible revision date; underlying TF-TRT/TensorFlow SavedModel stack reads as dated  
Compilation and export

Hands-on cloud-GPU lab: convert a TensorFlow SavedModel to TF-TRT at FP32/FP16/INT8 on InceptionV3, benchmark throughput, observe the accuracy/speed trade-off. The DLI 8-hour version ends in a graded assessment for a certificate; the Coursera guided-project twin is the same content, completion-graded only.

### [perf-ninja](https://github.com/dendibakh/perf-ninja)
`free` · `exercise repo`  
20+ labs (9 Core Bound, 9 Memory Bound, 4 Bad Speculation, 4 Misc); 3,787 stars, 388 forks · last activity 2026-07-16  
CPU performance

C++ course (Rust/Zig ports exist) of small realistic kernels — false sharing, loop tiling/interchange, prefetching, huge pages, alignment, vectorization, branch prediction, lookup tables. You optimize a lab, PR it, and CI checks correctness plus a wall-clock speedup threshold (Google Benchmark) on real hardware (Alderlake/Zen3/M1). Verified via README fetch, GitHub API metadata, and direct read of the false-sharing lab + its CI mechanism.

### [PyBites Platform](https://pybitesplatform.com/bites/regular/)
`freemium`  
435 'Bites'; free tier gives 30, lifetime access is $300 · last activity 2026 (live commercial platform; org repos show ongoing 2026 activity, no single commit date applies)  
Deep Python

Gamified bite-sized coding challenges; each submission is checked by an automated test suite plus a linter, with belts/leaderboard progress tracking.

### [Stanford CS106L assignments](https://github.com/cs106l/cs106l-assignments)
`free` · `course with labs`  
7 assignments · last activity 2026-07  
Deep C++

Public starter code and local autograders for Stanford's 1-unit Standard C++ Programming lab course. Assignment 6 is move semantics; assignment 7 has you implement your own unique_ptr (RAII, ownership transfer, operator overloading) and the autograder checks it.

### [SW Online Judge (formerly CUDA Online Judge / cudaforces)](https://swforces.com/)
`free`  
problems split Easy/Medium/Hard, exact count not exposed without an account; 43 GitHub stars on the judge-engine repo (SungHwanYun/cudaforces) · last activity 2026-01  
GPU / CUDA

Transpiles submitted CUDA-C to OpenMP C++ and runs it on CPU ('CUDA Code → Transpiler + Validate → C++ Code (OpenMP) → CPU Execute & Judge'). The project states plainly: 'Performance benchmarking is not available — the platform is for correctness verification only.' cudaforces.com now 301-redirects here; the product has broadened to general algorithms and Linux systems too.

### [Tensara — scaled-dot-attention problem](https://tensara.org/)
`free`  
1 of ~90 problems in the repo touches attention · last activity 2026-04-23  
Attention and KV cache, GPU / CUDA

A single 'hard' problem asking for plain softmax(QK^T/√E)V over (B,H,S,E) tensors, scored on real-GPU wall-clock speed; no tiling, no online softmax, no cache.

### [Triton-Puzzles (gpu-mode / Sasha Rush)](https://github.com/gpu-mode/Triton-Puzzles)
`free` · `puzzle set`  
12 puzzles total; 2 belong to this area (#8 Long Softmax, #9 Simple FlashAttention) · last activity 2026-04-01  
Attention and KV cache

Colab notebook that builds from trivial Triton pointer kernels up to a single-tile flash-attention kernel with online softmax, each puzzle auto-checked against a reference via a Triton interpreter — no GPU needed.


## Ships tests you run yourself

No submission, but the tests are there and they are real — you find out.

### [ARENA 3.0 — Chapter 1: Transformer Interpretability](https://github.com/callummcdougall/ARENA_3.0)
`free` · `course with labs`  
chapter 1 alone: 2 compulsory exercise sets + several optional extensions (SAEs, steering vectors, IOI circuits) · last activity 2026-07-24  
LLM internals

Public materials for a mechanistic-interpretability training program. The first exercise set has you build a GPT-2-architecture transformer from scratch in raw PyTorch (attention, positional encoding, layer norm) mirroring TransformerLens internals, then sample from it, with solutions and test functions to check intermediate tensors.

### [Build a Large Language Model (From Scratch) — rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
`freemium` · `book with exercises`  
7 main chapters + bonus/appendix folders (KV-cache, MLA, LoRA, etc.); ~30 quiz questions per chapter in a separate free PDF · last activity 2026-07-11  
LLM internals

Companion repo to Sebastian Raschka's Manning book; builds embeddings, causal/multi-head attention, LayerNorm, feed-forwards, and full GPT-2-style pretraining in plain PyTorch, chapter by chapter, with end-of-chapter exercises (solutions in Appendix C) and a confirmed dedicated ch04/03_kv-cache folder implementing KV-cache inference.

### [CMU 10-414/714 — Deep Learning Systems (Needle)](https://dlsyscourse.org/assignments/)
`free` · `course with labs`  
5 homeworks + final project · last activity 2026 (Fall 2025 due dates visible, currently taught)  
LLM systems

Students build 'Needle,' a PyTorch-like autodiff framework, from CPU/GPU backends through CNNs/RNNs/Transformers, finishing with lectures on training large models and deployment. Real autograder ('mugrade') is CMU-enrollment-only; public repos ship the tests to self-check against.

### [Computer Enhance: Performance-Aware Programming](https://www.computerenhance.com/p/table-of-contents)
`paid` · `course with labs`  
5 parts + bonus material, weekly homework · last activity 2023-01 (series launch); could not confirm if still being extended  
CPU performance

Casey Muratori's language-agnostic video course covering cache-size/bandwidth testing, cache indexing, branch prediction, and SSE intrinsics with weekly homework. Homework is self-checked against community solutions on GitHub, no autograder. Confirmed paywalled (a subscriber comment on the page states they paid specifically for this course; no exact price found).

### [CS231n Assignment 1 (Stanford)](https://cs231n.github.io/assignments2026/assignment1/)
`free` · `course with labs`  
1 assignment, 5 parts (kNN / SVM / Softmax / two-layer net / image features) · last activity 2026 (current live edition)  
Algorithms from scratch

Stanford computer-vision course, actively run with a 2026 edition. Implement k-NN, SVM loss, a softmax classifier, and a two-layer neural network with backprop derived and coded by hand in raw NumPy, with gradient-check and expected-loss sanity-check cells built into the Colab notebooks for self-verification.

### [CS:APP Cache Lab (CMU 15-213)](https://csapp.cs.cmu.edu/3e/labs.html)
`free` · `course with labs`  
1 lab, 2 parts (cache simulator + matrix-transpose optimization), part of an 11-lab course · last activity 2014 (self-study handout date); lab unchanged since CS:APP3e (~2015)  
CPU performance

Students write a cache simulator (csim.c) and optimize a matrix-transpose kernel (trans.c), scored by a driver script on exact cache-miss counts against a Valgrind-traced reference — a fixed deterministic metric. A self-study handout (cachelab-handout.tar) lets non-CMU learners run it independently. Verified by fetching the labs page and the cachelab.pdf handout directly.

### [dataflowr — Flash-Attention in Triton](https://github.com/dataflowr/gpu_llm_flash-attention)
`free` · `course with labs`  
3 homework parts: softmax-matmul kernel, FA forward/backward in PyTorch, FA ported to Triton with benchmarking · last activity 2026-02-09  
Attention and KV cache

A university course module handing you an empty notebook and a PDF spec; you fill in TODOs to build online softmax and tiled attention step by step, checked against an included tests/ folder.

### [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)
`free` · `puzzle set`  
14 puzzles, 12,341 GitHub stars · last activity 2024-09  
GPU / CUDA

The famous one. Python via NUMBA's CUDA JIT (not CUDA-C), each puzzle self-checked in-notebook against a NumPy reference. Designed to run on a real GPU (recommended: Colab GPU runtime), not a simulator. Quiet for ~22 months but still the most-cited resource in this space.

### [karpathy/minbpe](https://github.com/karpathy/minbpe)
`free` · `exercise repo`  
1 focused exercise, 3 tokenizer classes (Basic/Regex/GPT4) · last activity 2024-07-01  
LLM internals

Karpathy's minimal from-scratch BPE implementation with a companion exercise.md that lays out a step-by-step build order to your own GPT-4-style tokenizer, checked against the shipped reference gpt4.py and the repo's own pytest tests.

### [LearnCpp.com](https://www.learncpp.com/)
`free` · `book with exercises`  
unknown exact count; dozens of chapters, roughly 200 lessons, each ending in a self-check quiz · last activity 2025-01 (per the smart-pointers/move-semantics chapter's own revision date; the site is updated lesson-by-lesson, not in dated batches)  
Deep C++

A complete, free, widely used C++ tutorial book. Each chapter ends in a short quiz where the answer is hidden until you click to reveal it and compare against your own. Chapter 22 covers move semantics and smart pointers; sibling chapters cover virtual functions, templates and object relationships.

### [llm-inference-engine (achi9629)](https://github.com/achi9629/llm-inference-engine)
`free` · `exercise repo`  
1 star, 122 pytest tests · last activity 2026-05  
Batching and serving

A solo project that builds an inference engine in explicit incremental stages: plain transformer forward pass, KV cache, static batching, continuous batching, paged KV cache, then an async FastAPI serving layer, with 122 pytest tests and isolated before/after benchmarks at each stage. Structurally the closest GitHub match to 'implement the mechanic, then check yourself' for this area, but a very young, one-star, single-author repo, not an established or vetted resource.

### [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles)
`free` · `puzzle set`  
8 puzzles · last activity 2024-01  
LLM systems

Eight Colab puzzles simulating a multi-GPU cluster (not real hardware) where you implement data parallelism, pipeline parallelism, and ZeRO-style sharding against a memory budget; an in-notebook Model.check() asserts the weights were correctly sharded and updated, printing 'Correct!'.

### [MIT 6.5940 / EfficientML.ai — TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/course)
`free` · `course with labs`  
5 labs total (pruning lab + part of quantization lab overlap this area); offered again Fall 2026 · last activity ongoing (Fall 2026 offering listed)  
Sparsity, pruning, distillation

Song Han's MIT graduate course on efficient AI computing covering pruning, quantization, distillation, NAS, and on-device LLM deployment, each backed by a hands-on Colab lab; third-party mirrors of student solutions confirm the official labs ship in-notebook tests a learner runs to self-check.

### [MIT 6.5940 — TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
`free` · `course with labs`  
5 labs + final project · last activity 2024-09 (Fall 2024); not offered Fall 2025, next offering unconfirmed  
LLM systems

Song Han's course on efficient deep learning: pruning, quantization, neural architecture search, and a lab deploying Llama-2-7B locally, with lecture coverage of distributed training and gradient/model compression. Labs are Colab notebooks with built-in pass/fail sanity checks; public without enrollment, but not currently a running/live course.

### [MLC: Machine Learning Compilation (mlc.ai)](https://mlc.ai/courses.html)
`free` · `course with labs`  
~8 chapters/notebooks · last activity 2022-07 (course content not visibly revised since; mlc-ai org itself active but has shifted to shipping mlc-llm)  
Compilation and export

CMU-taught open course on ML compilation built around Apache TVM/TensorIR; each chapter pairs a lecture with a notebook where you write TVMScript and compare output against a NumPy reference.

### [Oak Ridge OLCF CUDA Training Series](https://github.com/olcf/cuda-training-series)
`free` · `course with labs`  
13 sessions/homeworks (hw1–hw11+), 1,022 GitHub stars · last activity 2024-08  
GPU / CUDA, Memory and offload

National-lab training series (slides at olcf.ornl.gov/cuda-training-series, confirmed live): one deck + one hands-on exercise per topic (CUDA C++ basics, shared memory, fundamental optimization parts 1-2, atomics/reductions/warp shuffle, streams, cooperative groups). Live sessions ran Jan 2020–Oct 2021 and have not resumed; the repo is a finished, static artifact now, not a running course. You compile and self-check output against README-stated expected values.

### [Python Morsels](https://www.pythonmorsels.com/exercises/paths/)
`paid`  
170+ exercises across 15 learning paths, including named descriptors / metaclasses / context-managers / generators-and-iterators paths · last activity 2026 (paywalled article and exercise pages confirmed live this session)  
Deep Python

Trey Hunner's weekly-exercise subscription: an explanatory article on a protocol (e.g. descriptors: __get__/__set__/__set_name__, data vs non-data) followed by several small coding exercises, each shipping its own test file you run locally. $14-29/month or $120-240/year; 3 free preview exercises.

### [Stanford CS336 — Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
`free` · `course with labs`  
1 multi-part assignment, ~10 gradable components (tokenizer, RMSNorm, RoPE, SwiGLU, MHA, transformer block/LM, cross-entropy, AdamW, LR schedule, checkpointing) · last activity 2026-04-07  
LLM internals

Public student repo for Stanford's Language Modeling From Scratch course. tests/adapters.py requires implementing run_rmsnorm, run_rope, run_swiglu, run_scaled_dot_product_attention, run_multihead_self_attention_with_rope and run_transformer_block/run_transformer_lm plus a BPE tokenizer, all checked by pytest starting from NotImplementedError.

### [Stanford CS336 — Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)
`free` · `course with labs`  
1 assignment (of 5), 27 commits · last activity 2026-05  
Attention and KV cache, LLM systems

The systems assignment of Stanford's 'build an LLM from scratch' course: profiling, mixed-precision training, a hand-written FlashAttention-2 Triton kernel, and FSDP-style distributed data parallel training on top of your own Assignment-1 model. Ships real tests and a submission script; needs real multi-GPU compute to run.

### [Tensor Puzzles (srush)](https://github.com/srush/Tensor-Puzzles)
`free` · `puzzle set`  
21 puzzles · last activity 2024-03  
Numerics and tensors

Reimplement NumPy/PyTorch primitives (ones, sum, outer, diag, cumsum, scatter_add, bincount, etc.) in one line using only broadcasting, arithmetic, comparison, @ and indexing - no library shortcuts. Each puzzle ships a Hypothesis-based run_test() checker you run locally that gives pass/fail plus a broadcast-shape diagram on failure.

### [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)
`free` · `puzzle set`  
2,539 GitHub stars (part of a 7-puzzle-series family by the same author) · last activity 2026-04  
GPU / CUDA

Sister project to GPU-Puzzles for Triton, a Python DSL that compiles to GPU code. Explicitly does not need a real GPU — runs on a Triton interpreter — and is not CUDA-C.


## Reference code to read

Implementations to study. Nothing checks you.

### [AviSoori1x/makeMoE](https://github.com/AviSoori1x/makeMoE)
`free` · `reference implementation`  
811 stars · last activity 2024-10  
Sparsity, pruning, distillation

From-scratch, single-file (plus notebooks) sparse Mixture-of-Experts language model in the style of Karpathy's makemore/nanoGPT: top-k and noisy top-k gating, plus a follow-on notebook adding expert-capacity limits.

### [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
`free` · `reference implementation`  
8,340 stars · last activity within the last day — very active, Hugging Face-backed  
Applied quantization, Memory and offload

Production library implementing LLM.int8() vector-wise outlier-aware quantization, blockwise 8-bit optimizers, and NF4 4-bit quantization (the QLoRA dtype), with real dequant kernels behind a drop-in nn.Linear. Use-as-a-library; no exercises.

### [C++ Templates (2nd ed.) study notes — downdemo/Cpp-Templates-2ed](https://github.com/downdemo/Cpp-Templates-2ed)
`free` · `reference implementation`  
~1,600 GitHub stars, 15 chapters, mostly Chinese-language notes with runnable code · last activity 2025-01  
Deep C++

A distilled, code-verified companion to Vandevoorde/Josuttis's "C++ Templates: The Complete Guide," covering function/class templates, variadics, SFINAE, traits, CRTP and C++20 concepts with runnable examples for each idea.

### [CPython Internals: Your Guide to the Python 3 Interpreter (book + cpython-book-samples)](https://realpython.com/products/cpython-internals-book/)
`paid` · `book with exercises`  
first edition, CPython 3.9-era; companion repo has 285 stars and sample code for 9 chapters · last activity companion repo last commit 2020-12-12 — over 5 years stale; book not revised for post-3.9 interpreter changes · *host blocks automated fetchers; the resource is live*  
Deep Python

Anthony Shaw's book walking a reader through the real CPython C source: compiling your own interpreter, then modifying core object types, generators, and memory management. The product page 403'd this fetcher; authenticity/content verified instead by fetching and reading the publicly-hosted sample-chapters PDF directly, and by fetching the companion code repo github.com/tonybaloney/cpython-book-samples (both succeeded and matched).

### [ddbourgin/numpy-ml](https://github.com/ddbourgin/numpy-ml)
`free` · `reference implementation`  
16.3k stars · last activity 2022-01 (dormant ~4.5 years) · **dormant or archived**  
Algorithms from scratch

Documented, more rigorous reference implementations: CART decision trees, bagging/random forests/GBTs, GMM trained with actual EM, HMMs, Bayesian linear regression, Gaussian processes, plus SGD/AdaGrad/RMSProp/Adam optimizers. Ships its own internal test suite for the maintainer's code correctness, not for grading a learner's solutions.

### [EleutherAI Cookbook](https://github.com/EleutherAI/cookbook)
`free` · `reference implementation`  
845 stars, ~54 commits · last activity 2026-03  
LLM systems

Runnable calc/ scripts for FLOPs, memory, and parameter-count estimation, plus communication and GEMM benchmarks and a curated reading list, framed by its authors as 'deep learning for dummies' — the practical utilities around real model training.

### [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
`free` · `reference implementation`  
32.4k stars, 374 commits · last activity 2019-10 (dead, no commits in ~6.75 years) · **dormant or archived**  
Algorithms from scratch

The most-starred 'ML from scratch' repo in existence; bare-bones NumPy code for nearly every classic algorithm (regression variants, trees, ensembles, k-NN, k-means, GMM, PCA, DBSCAN, SVM, plus autoencoder/GAN/RBM). Read-only, no exercises, no grading, and unmaintained since 2019.

### [Fluent Python, 2nd ed. (book + example-code-2e)](https://github.com/fluentpython/example-code-2e)
`paid` · `reading list`  
4.1k GitHub stars, 360 commits, code for 24 chapters · last activity 2025-06  
Deep Python

Luciano Ramalho's book, organized explicitly around 'the Python data model' — dunder protocols, descriptors, __slots__, iterators/generators, coroutines, context managers, metaclasses — paired with this companion repo of runnable example code per chapter.

### [ggml-org/llama.cpp — ggml-quants.c (GGUF k-quants)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
`free` · `reference implementation`  
121,618 stars (whole repo) · last activity within the last day — extremely active  
Applied quantization

The actual bit-packing source for Q2_K through Q8_0 (and legacy Q4_0/Q5_0): super-block scale quantization and per-block zero-points, the real byte layout our GGUF k-quant tasks model. C, not Python, production inference code with no learner-facing exercises.

### [google-research/lottery-ticket-hypothesis](https://github.com/google-research/lottery-ticket-hypothesis)
`free` · `reference implementation`  
731 stars · last activity 2020-07  
Sparsity, pruning, distillation

Frankle & Carbin's own reimplementation: iterative magnitude pruning on MNIST fully-connected nets with weight-rewind-to-init between rounds, in TensorFlow 1.x. Archived/read-only by the owner.

### [Guru of the Week (GotW)](https://herbsutter.com/gotw/)
`free` · `puzzle set`  
94+ numbered issues (original 88 plus ongoing additions) · last activity unknown (ongoing; site shows revision/addition activity through 2026 but no single dated latest issue was visible)  
Deep C++

Herb Sutter's long-running column of C++ engineering puzzles: a scenario and a question (is this exception-safe, when does this template become ambiguous, what's wrong with this class), followed by a full worked solution. Basis for his "Exceptional C++" books.

### [Hugging Face Transformers docs — Knowledge Distillation (image classification)](https://huggingface.co/docs/transformers/tasks/knowledge_distillation_for_image_classification)
`free` · `reference implementation`  
one tutorial page · last activity current (live official docs)  
Sparsity, pruning, distillation

Distills a fine-tuned ViT teacher into a randomly-initialized MobileNetV2 student on the beans dataset via a custom Trainer subclass with KL-divergence soft-target loss plus true-label loss; reports 72% vs 63% test accuracy.

### [huggingface/nn_pruning](https://github.com/huggingface/nn_pruning)
`free` · `reference implementation`  
409 stars · last activity 2022-06  
Sparsity, pruning, distillation

Movement pruning extended to semi-structured/block-structured variants so masks align to hardware-friendly tiles; demonstrated on BERT/SQuAD and GLUE. Archived Jul 2025.

### [insdout/ML-Algorithms-From-Scratch](https://github.com/insdout/ML-Algorithms-From-Scratch)
`free` · `reference implementation`  
2 stars, 165 commits · last activity unknown  
Algorithms from scratch

Personal study repo, notable for being one of the very few things found anywhere that explicitly implements QR decomposition and eigendecomposition/SVD alongside the usual k-means/k-NN/decision-tree/GMM-EM/PCA/random-forest set. Extremely low visibility - a learner would have to already know to search for it.

### [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
`free` · `reference implementation`  
2,338 stars · last activity 2024-03-27 (stale, 2+ years)  
Applied quantization

Original ICLR 2023 GPTQ paper code: Hessian-based layer-wise post-training quantization of OPT/BLOOM to 2/3/4 bits, with CUDA kernels and perplexity evaluation scripts. No exercises, no starter/reference split — read-only ground truth for the algorithm, largely superseded in practice by maintained forks (e.g. ModelCloud/GPTQModel).

### [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
`free` · `reference implementation`  
200+ CUDA kernels, GPL-3.0, 11,632 GitHub stars · last activity 2026-07  
GPU / CUDA

Despite the name, not a judge — a progressively-harder reference-kernel library (easy to hard++) with PyTorch bindings and benchmark tables against cuBLAS/cuDNN, covering GEMM/GEMV, FlashAttention variants, and Tensor-Core paths. Nothing to submit; you read and adapt the code.

### [llm.c](https://github.com/karpathy/llm.c)
`free` · `reference implementation`  
1536+ commits, 30.6k stars · last activity 2025-06  
LLM systems

GPT-2/GPT-3 training in raw C/CUDA with no PyTorch dependency: mixed-precision training, gradient accumulation, a tokenized dataloader, and multi-GPU/multi-node training via MPI+NCCL, with unit tests cross-checking against a PyTorch reference.

### [locuslab/wanda](https://github.com/locuslab/wanda)
`free` · `reference implementation`  
868 stars · last activity 2024-08  
Sparsity, pruning, distillation

Official code for the Wanda pruning paper (ICLR 2024): prunes by |weight| x input-activation-norm per output, no retraining needed; also ships magnitude and SparseGPT baselines and supports 2:4/4:8 N:M structured patterns.

### [Microsoft NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)
`free` · `reference implementation`  
14,367 stars — largest project in this list · last activity 2024-09 (archived; last real release 2022-05) · **dormant or archived**  
Sparsity, pruning, distillation

General AutoML toolkit whose compression module implemented many pruners (L1/L2 Norm, FPGM, Taylor-FO, Movement, AGP, AutoCompress), quantizers, and a basic distillation component, each with quick-start tutorials. Archived Sep 2024, read-only.

### [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
`free` · `reference implementation`  
3,596 stars · last activity 2025-07-17  
Applied quantization

Original AWQ paper code (MLSys 2024 Best Paper): activation-aware salient-channel search, INT3/4 weight-only quantization, real CUDA kernels, precomputed model zoo, TinyChat inference engine. Reference code only, no exercises.

### [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
`free` · `reference implementation`  
1,672 stars · last activity 2024-07-12 (about 2 years stale)  
Applied quantization

Original ICML 2023 SmoothQuant paper code: migrates quantization difficulty from activations to weights via a per-channel smoothing factor to enable W8A8 with near-fp16 accuracy. OPT/Llama demo notebooks, TensorRT-LLM/ONNX export examples. Reference-only.

### [OscarSavolainen/Quantization-Tutorials](https://github.com/OscarSavolainen/Quantization-Tutorials)
`free` · `reference implementation`  
31 stars (small) · last activity 2024-05-21 (slowing, single contributor)  
Applied quantization

Companion code to a YouTube series: PyTorch eager-mode static/dynamic PTQ, FX-graph-mode PTQ, FX QAT, and cross-layer equalization, all on ResNet. No built-in correctness checking — read the matching folder alongside the video.

### [picoGPT](https://github.com/jaymody/picoGPT)
`free` · `reference implementation`  
~60-120 lines depending on the file · last activity 2023-04-24  
LLM internals

GPT-2 forward pass in plain NumPy including its own BPE encoder, loads real GPT-2 weights and produces real output; deliberately has no training code, batching, or KV-cache. Nothing to submit; you read and modify it yourself.

### [picotron](https://github.com/huggingface/picotron)
`free` · `reference implementation`  
~180 commits, 2.3k stars · last activity 2025-08  
LLM systems

Minimalist reference implementation of 4D parallelism (data/tensor/pipeline/context) for pretraining LLaMA-style models, deliberately kept to single files under ~300 lines each, built explicitly for education alongside companion video tutorials.

### [Programming Massively Parallel Processors (PMPP), 4th ed. + community solutions](https://github.com/tugot17/pmpp)
`paid` · `book with exercises`  
20 chapters of end-of-chapter exercises; this solutions repo has 817 GitHub stars and covers all of them in CUDA C and Python · last activity 2025-06  
GPU / CUDA

The field's standard textbook (Kirk, Hwu & Hajj). No official autograder for its exercises; most self-learners cross-check against community solution repos like this one (several similar ones exist: nvixnu, guanrenyang, Isalia20). Book itself is paid; the solutions repo is free and MIT-licensed.

### [PyTorch tutorial — Accelerating BERT with 2:4 sparsity (torchao)](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html)
`free` · `reference implementation`  
one tutorial; backed by pytorch/ao (2,917 stars) · last activity 2026-07 (torchao pushed 2026-07-26, very active)  
Sparsity, pruning, distillation

End-to-end walkthrough: train BERT on SQuAD dense, apply magnitude-based 2:4 pruning via torch.ao.pruning, fine-tune to recover F1, then use SparseSemiStructuredTensor for real inference speedup (~1.3x, up to 2x with torch.compile).

### [PyTorch tutorial — Pruning (torch.nn.utils.prune)](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
`free` · `reference implementation`  
one tutorial page; parent repo pytorch/tutorials has 9,255 stars · last activity 2026-07 (parent repo actively maintained; page is current live docs)  
Sparsity, pruning, distillation

Official walkthrough of random/l1_unstructured, ln_structured, global_unstructured pruning APIs and writing a custom BasePruningMethod, with before/after sparsity printouts.

### [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)
`free` · `reference implementation`  
11.2k stars, 151 commits · last activity 2026-05  
Algorithms from scratch

Minimal, clean NumPy implementations of linear/logistic regression, k-NN, k-means, GMM, Naive Bayes, PCA, SVM, random forests, gradient boosting, factorization machines, t-SNE. No tests, no grader, read-only reference code.

### [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)
`free` · `reference implementation`  
1 notebook covering the full original Transformer paper · last activity 2024-04-07  
LLM internals

Line-by-line PyTorch implementation of Attention Is All You Need interleaved with the paper's own text as a runnable notebook. Nothing to fill in and nothing checks your work; it's read-and-run, not exercise-and-grade.

### [tiny-vllm](https://github.com/jmaczan/tiny-vllm)
`free` · `exercise repo`  
952 stars · last activity 2026-07  
Batching and serving

A guided 'build vLLM yourself' repo in C++/CUDA: starts from bf16 arithmetic and safetensors loading, works through hand-written kernels (RMSNorm, RoPE, attention, softmax), then covers prefill vs decode, static batching, continuous batching, online softmax and PagedAttention. Implements the batching/scheduling mechanics for real, but through low-level CUDA kernels rather than a Python request-path layer. No automated grading; you implement against the author's reference and compare.

### [Transformer Math 101 (EleutherAI blog)](https://blog.eleuther.ai/transformer-math/)
`free` · `reading list`  
1 blog post · last activity 2023-04  
LLM systems, Memory and offload

The canonical reference derivation of training compute (C≈6PD), full memory accounting across weights/gradients/optimizer-states/activations under different precisions and ZeRO stages, and achievable per-GPU throughput ranges.

### [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
`free` · `reference implementation`  
3,582 stars, 3,133+ commits · last activity within the last day — very active  
Applied quantization

One library that runs GPTQ, AWQ, SmoothQuant, AutoRound and rotation-based methods (SpinQuant/QuIP) end to end for vLLM deployment, targeting W8A8/W4A16/FP8/NVFP4/MXFP4. Ships example scripts per recipe, not exercises; GGUF is out of scope. Sibling tools worth knowing: ModelCloud/GPTQModel (active, multi-hardware GPTQ/AWQ export) and the now-archived casper-hansen/AutoAWQ.


## Reading, tools and reference

Books, papers, docs and interactive explainers. Nothing checks you.

### [100 NumPy exercises (rougier/numpy-100)](https://github.com/rougier/numpy-100)
`free` · `exercise repo`  
100 exercises · last activity 2026-07  
Numerics and tensors

The canonical NumPy exercise list pulled from the mailing list, Stack Overflow and the docs, shipped as no-solution / hints / full-solutions markdown and notebook versions. You self-check by diffing against the solutions file; there is no automated grader.

### [Advanced Python Mastery (David Beazley)](https://github.com/dabeaz-course/python-mastery)
`free` · `exercise repo`  
12.8k GitHub stars; 9 sections, exercise files ex1_1.md–ex9_4.md · last activity 2025-12  
Deep Python

David Beazley's free public release of his multi-day corporate Python training: slides, exercises, and full checked-in solutions you compare your own attempt against manually — no test runner or automated grading.

### [Agner Fog's optimization manuals](https://www.agner.org/optimize/)
`free` · `reading list`  
5 manuals (C++ optimization, asm optimization, microarchitecture, instruction tables, calling conventions) · last activity 2026-07 (most recently updated manual, confirmed via direct fetch)  
CPU performance

The reference of record for x86/x86-64 instruction latencies/throughputs and microarchitectural detail (out-of-order execution, branch prediction internals). Actively maintained. Pure reference, no exercises.

### [AIPerf](https://github.com/ai-dynamo/aiperf)
`free` · `benchmark or leaderboard`  
469 stars · last activity 2026-07  
Batching and serving

NVIDIA's current LLM-serving load-testing CLI (successor to the now-deprecated genai-perf). Points at any OpenAI-compatible/TGI endpoint you already have running and reports TTFT, inter-token latency, throughput and per-user token rate under configurable concurrency and workload shapes. Its predecessor ray-project/llmperf was archived Dec 2025 after its last commit in Dec 2024.

### [Algorithms for Modern Hardware (Algorithmica)](https://en.algorithmica.org/hpc/)
`free` · `reading list`  
~9 chapters in the Performance Engineering part · last activity 2024-08 (underlying repo last push); author called it ~75% complete in 2022, appears stalled there · **dormant or archived**  
CPU performance

Free online book by Sergey Slotin: dense exposition plus C++ snippets and case studies (e.g. faster matrix multiplication, faster binary search) covering cache lines, associativity, prefetching, SIMD intrinsics, branch prediction. No exercises, no grading. Verified via direct fetch and GitHub API on algorithmica-org/algorithmica.

### [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
`free` · `reference implementation`  
7,261 stars · last activity 2026-06  
Sparsity, pruning, distillation

CLI toolkit for merging pretrained LLM checkpoints (linear, SLERP, TIES, DARE, task-arithmetic) out-of-core on CPU/low VRAM; mergekit-extract-lora decomposes a fine-tune's weight delta into a PEFT-compatible LoRA adapter.

### [array-api-tests (data-apis)](https://github.com/data-apis/array-api-tests)
`free` · `reference implementation`  
full Array API spec surface · last activity 2026 (active)  
Numerics and tensors

A conformance test suite for the Python Array API standard, not a learning resource per se. Encodes NumPy-style dtype-promotion rules as executable assertions rather than prose; a learner can point ARRAY_API_TESTS_MODULE at NumPy and run/read the promotion tests to see the real casting table exercised against real inputs.

### [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
`free` · `reading list`  
5.4k stars, 100+ tracked papers/repos · last activity 2026-07  
Batching and serving

A curated paper+code index (same maintainer family as LeetCUDA) with dedicated sections for continuous/in-flight batching, prefill/decode disaggregation (DistServe, Mooncake), scheduling papers, and serving frameworks (vLLM, SGLang, TensorRT-LLM, LMDeploy). Nearly every entry links both a paper and a real implementation. Good for going deep on one subtopic, not a starting point for a beginner.

### [awesomeMLSys (GPU MODE)](https://github.com/gpu-mode/awesomeMLSys)
`free` · `reading list`  
~1.1k stars, 17 commits, 8 topic categories · last activity 2026-02  
LLM systems

A curated bibliography of papers/videos/repos for ML-systems onboarding — attention, inference optimization, quantization, long context, distributed training, speculative decoding — explicitly framed as study material, not a curriculum with checkpoints.

### [Colossal-AI (Gemini heterogeneous memory manager)](https://github.com/hpcaitech/ColossalAI)
`free` · `reference implementation`  
41,426 stars · last activity 2026-07-13 (pushed_at) — active  
Memory and offload

Large distributed-training system whose Gemini memory manager (built on earlier PatrickStar work) tracks tensor liveness and dynamically places model states across GPU/CPU/NVMe under a runtime memory budget.

### [csc-training/CUDA — exercises/unified-memory-streams](https://github.com/csc-training/CUDA/tree/master/exercises/unified-memory-streams)
`free` · `exercise repo`  
130 stars on parent repo; 1 exercise among a handful · last activity 2017-05-19 (pushed_at) — dead ~9 years, no commits since · **dormant or archived**  
Memory and offload

A CSC training exercise: allocate CUDA managed (unified) memory and split addition-kernel launches across multiple streams, filling TODOs in streams.cu against a provided solution/ folder to diff against. No automated checker.

### [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
`free` · `reference implementation`  
24,539 stars; production FlashAttention-2/3 library (causal, MQA/GQA, varlen, paged-KV support) · last activity 2026-07-25  
Attention and KV cache

The canonical production implementation the whole field benchmarks against; CUDA/CUTLASS kernels to read, no tests or puzzles attached.

### [Data Science from Scratch (Joel Grus) - book + code](https://github.com/joelgrus/data-science-from-scratch)
`freemium` · `book with exercises`  
27 chapters · last activity 2020-09  
Algorithms from scratch

Builds k-NN, k-means, hierarchical clustering, decision trees, gradient descent, and a simple neural net using plain Python lists with no NumPy at all, specifically to force understanding of every step. Code-along narrative with 'for further exploration' prompts rather than checked exercises; the book text is paid, the code repo is free/MIT.

### [Data-Oriented Design (Richard Fabian)](https://www.dataorienteddesign.com/dodbook/)
`freemium` · `reading list`  
1 book, ~9 chapters · last activity 2018  
CPU performance

Free reduced online version (paid paperback for full version) treating SoA vs AoS and data-oriented layout as a first-class subject, mostly through a game-dev lens (components, hierarchical LOD). No exercises.

### [DeepLearning.AI — Quantization Fundamentals with Hugging Face](https://learn.deeplearning.ai/courses/quantization-fundamentals)
`freemium` · `course with labs`  
short course, ~9 video-with-code lessons · last activity unknown  
Applied quantization

Video-with-code short course by Hugging Face engineers Younes Belkada and Marc Sun: int/float dtype basics, loading models in different precisions, linear quantization via Hugging Face's quanto library, and bf16 downcasting. Free tier is watch-and-run-the-cell; graded notebooks/quizzes are a paid Plus feature.

### [DeepLearning.AI — Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)
`freemium` · `course with labs`  
13 code examples · last activity unknown  
Applied quantization

Follow-on course, same instructors: you build a general linear quantizer in PyTorch from scratch, choosing symmetric vs. asymmetric mode and per-tensor/per-channel/per-group granularity, then implement weight packing down to 2-bit storage. Free tier is code-along plus a quiz; the one graded assignment is paid-tier only.

### [DeepSpeed — ZeRO / ZeRO-Offload tutorials](https://www.deepspeed.ai/tutorials/zero-offload/)
`free` · `reference implementation`  
42,806 stars on the library; 2 tutorial pages plus real example configs · last activity tutorial footer 2026-07-24; repo pushed 2026-07-26 — very active  
Memory and offload

Official walkthrough for ZeRO stages 1-3 (partition optimizer state/gradients/params) and ZeRO-Offload/-Infinity (push optimizer state and compute to CPU, or CPU+NVMe). Config-JSON only, no code changes — you edit settings and watch memory/throughput move.

### [depyf](https://github.com/thuml/depyf)
`free` · `reference implementation`  
815 stars · last activity 2025-10  
Compilation and export

A tool (with a JMLR paper behind it) that decompiles the bytecode torch.compile/Dynamo generates back into readable Python, so you can see exactly how and where your function was split at each graph break.

### [Efficiently Serving LLMs (DeepLearning.AI / Predibase)](https://www.deeplearning.ai/courses/efficiently-serving-llms/)
`freemium` · `course with labs`  
9 video lessons / 7 notebooks · last activity unknown  
Batching and serving, LLM systems

2h40m video course by Predibase's CTO: KV caching, batching, continuous batching, quantization, LoRA and multi-LoRA serving, ending in a look at Predibase's LoRAX server. Seven run-along notebooks (unofficial mirror at github.com/ksm26/Efficiently-Serving-LLMs). Free tier is watch-and-run-the-cell; one graded assignment exists but is gated behind a paid DeepLearning.AI/Coursera tier.

### [fastai Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra)
`free` · `course with labs`  
8 lecture notebooks · last activity 2017-07  
Numerics and tensors

Rachel Thomas's USF course notebooks (with YouTube lectures). Lecture 3 covers stability of LU with/without pivoting, Lecture 6 has a section explicitly on 'Conditioning & Stability' (why matrix inversion is unstable), Lecture 8 compares classical vs modified Gram-Schmidt stability. No autograder; you run the notebooks and compare to shown output.

### [FlexGen / FlexLLMGen](https://github.com/FMInference/FlexLLMGen)
`free` · `reference implementation`  
9,363 stars · last activity pushed 2024-10-28, archived 2024-12-01 — dead, read-only · **dormant or archived**  
Memory and offload

Stanford/Berkeley/CMU system for high-throughput batch LLM generation on one consumer GPU via a linear-program schedule that jointly places weights, activations, and KV cache across GPU/CPU/disk, plus 4-bit weight/cache compression. Project renamed from FlexGen; GitHub repo is archived.

### [FLHonker/Awesome-Knowledge-Distillation](https://github.com/FLHonker/Awesome-Knowledge-Distillation)
`free` · `reading list`  
2,679 stars; 658 papers · last activity 2023-05  
Sparsity, pruning, distillation

Curated bibliography of knowledge distillation papers (2014-2021) organized by KD mechanism (logits, feature, KD+GAN, data-free) and application domain, with links to original code.

### [Float Exposed](https://float.exposed/)
`free` · `interactive explainer`  
single-page tool · last activity unknown  
Numerics and tensors

Type a decimal or flip individual bits of a half/bfloat16/float/double and see the exact base-10 value, base-2 breakdown, and the exact delta to the next/previous representable value (local ULP/epsilon). Pure visualization, no exercises.

### [fp-conv](https://sw23.github.io/fp-conv/)
`free` · `interactive explainer`  
single-page tool, ~15 formats · last activity 2025  
Numerics and tensors

Same click-a-bit/type-a-value interaction as float.exposed but covering the full modern ML dtype zoo explicitly: fp64/fp32/fp16, bf16, tf32, and the OCP microscaling fp8 (e4m3, e5m2), fp6 (e3m2, e2m3) and fp4 (e2m1) formats, plus custom user-defined bit layouts. A near-duplicate exists at kuterdinel.com/float-gallery.html covering the same fp8/fp6/fp4/bf16/tf32 comparison in one static gallery.

### [From Python to NumPy (rougier)](https://github.com/rougier/from-python-to-numpy)
`free` · `book with exercises`  
~9 chapters · last activity 2025-05  
Numerics and tensors

Free open-access book; Chapter 2 'Anatomy of an array' is a careful, code-backed treatment of strides, memory layout, views vs copies and reshaping cost. Later chapters are vectorization exercises you check by comparing your own timing/output, not an automated grader.

### [Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)
`free` · `reference implementation`  
7 worked examples · last activity 2010  
CPU performance

Single classic blog post by Igor Ostrovsky with runnable C# snippets isolating cache effects: stride vs line size, L1/L2 capacity cliffs, associativity conflicts, false-dependency ILP, and a 15x false-sharing slowdown demo. Confirmed live via direct fetch. Article to read/re-run, not an exercise with a verdict.

### [GPU MODE lectures](https://github.com/gpu-mode/lectures)
`free` · `course with labs`  
6,358 GitHub stars; ongoing series of recorded lecture notebooks · last activity 2026-06  
GPU / CUDA

Jupyter-notebook lecture materials (paired with recorded video) covering kernel fundamentals through Triton, CUTLASS, and quantization. No submission or checking system — you read/run the notebooks yourself.

### [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
`free` · `reading list`  
2,497 stars; ~300+ papers · last activity 2024-04  
Sparsity, pruning, distillation

Curated bibliography of neural network pruning papers (2015-2023) organized by year and pruning type (filter/weight/other), with code links to the original authors' repos where available.

### [hkproj/triton-flash-attention (Umar Jamil)](https://github.com/hkproj/triton-flash-attention)
`free` · `exercise repo`  
257 stars; one complete FlashAttention-2 Triton kernel plus a long-form video walkthrough · last activity 2025-01-02  
Attention and KV cache

A 'code along with the video' path into flash-attention tiling; names two follow-on exercises (autotune the backward pass, skip masked blocks in causal attention) but ships no checker for them.

### [How To Scale Your Model (jax-ml Scaling Book)](https://jax-ml.github.io/scaling-book/)
`free` · `book with exercises`  
~14 chapters · last activity 2026-07  
LLM systems

Google DeepMind's ongoing blog-style textbook on scaling LLMs: roofline analysis, TPU/GPU hardware, FLOPs/memory/communication math, parallelism strategy selection, worked LLaMA-3 case studies. Has embedded 'problems to work for yourself' but no automatic answer-checking.

### [Hugging Face Accelerate — Big Model Inference guide](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
`free` · `reference implementation`  
unknown (single guide in an actively maintained library) · last activity living docs page, continuously updated with huggingface/accelerate  
Memory and offload

Hands-on guide to init_empty_weights + load_checkpoint_and_dispatch(device_map="auto"), which places each layer on the fastest device with room and streams weights layer-by-layer to CPU/disk when nothing fits.

### [Hugging Face Transformers — KV cache strategies doc](https://huggingface.co/docs/transformers/en/kv_cache)
`free` · `reference implementation`  
unknown (single guide in an actively maintained library) · last activity living docs page, continuously updated with huggingface/transformers  
Memory and offload

Runnable comparison of DynamicCache/StaticCache/QuantizedCache (hqq/quanto backends, int2-int8) and cache offloading (cache_implementation="offloaded", keeps only current layer's KV on GPU), including a worked OOM-retry example.

### [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)
`free` · `reference implementation`  
421 stars; 2-bit asymmetric per-channel/per-token KV-cache quantization reference · last activity 2025-11-20  
Attention and KV cache

Reference implementation of the KIVI paper's tuning-free 2-bit KV-cache quantization scheme, actively maintained relative to the other reference repos here.

### [kipp.ly — Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
`free` · `reading list`  
unknown (single long-form post) · last activity published 2022-03-30, static since  
Memory and offload

KV-cache byte-size formulas per token, a worked example of tokens-of-KV-cache that fit on an A100 for a 52B model, and when recompute beats caching.

### [Maxime Labonne — Introduction to Weight Quantization](https://maximelabonne.substack.com/p/introduction-to-weight-quantization-2494701b9c0c)
`free` · `interactive explainer`  
1 article + 1 companion Colab notebook · last activity 2023-07  
Applied quantization

Long-form article implementing absmax (symmetric) and zero-point (asymmetric) INT8 quantization from scratch in plain PyTorch, then LLM.int8() mixed-precision outlier handling, comparing GPT-2 perplexity before/after. Free, runnable, no test harness — you read the output and judge it yourself. Predates GPTQ/AWQ/GGUF, which it does not cover.

### [Maxime Labonne's LLM Course — quantization notebooks](https://github.com/mlabonne/llm-course)
`free` · `course with labs`  
81k+ GitHub stars; quantization section = 3 Colabs (GPTQ, GGUF/llama.cpp, ExLlamaV2) + a GPTQ/AWQ reading-list module · last activity 2026-02 (repo actively maintained; the quantization Colabs themselves are from the 2023 GPTQ/GGUF wave)  
Applied quantization

The de-facto roadmap most self-taught practitioners land on for 'how do I actually quantize a model today'. Runnable Colabs that apply existing libraries (auto-gptq, llama.cpp) to a real model and let you inspect resulting file size/perplexity — not from-scratch algorithm implementation.

### [microsoft/Tutel](https://github.com/microsoft/Tutel)
`free` · `reference implementation`  
1,001 stars · last activity 2026-07  
Sparsity, pruning, distillation

Production Mixture-of-Experts library: adaptive parallelism, dynamic capacity/routing switching, and the real dispatch/combine kernels behind fast sparse-MoE training and inference at scale.

### [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
`free` · `reference implementation`  
7,249 stars; Llama-2/MPT/Falcon/Pythia attention-sink + sliding-window implementations · last activity 2024-07-11  
Attention and KV cache

Original code for the StreamingLLM/attention-sinks paper; no commits in about two years, so treat it as a historical reference rather than a maintained tool against current library versions.

### [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
`free` · `reference implementation`  
14.6k GitHub stars, ~1,200 LOC · last activity 2026-04  
Batching and serving

A from-scratch vLLM reimplementation in ~1,200 lines of readable Python, with a real scheduler.py and block_manager.py implementing continuous batching and paged KV-cache management (not just the prefix-caching the README headline mentions). Claims throughput comparable to real vLLM; you read the code and run bench.py, there is no test suite or grading.

### [Neural Networks and Deep Learning (Michael Nielsen)](https://github.com/mnielsen/neural-networks-and-deep-learning)
`free` · `book with exercises`  
6 chapters + appendices · last activity unknown exact date; written for Python 2.6-2.7, author states no further updates planned - finished/frozen, not abandoned mid-way · **dormant or archived**  
Algorithms from scratch

The classic free derivation of backprop by hand: chapter 2 works through the four backpropagation equations from first principles, chapter 1 hand-codes an MNIST classifier in raw Python with only NumPy for matrix ops. End-of-section 'problems' are unanswered prose exercises, not auto-checked.

### [NVIDIA Blog — How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
`free` · `interactive explainer`  
unknown (single post, links a full runnable async.cu example on GitHub) · last activity published 2012-12-13 — old but API it documents (cudaMemcpyAsync, cudaMallocHost, non-default streams) unchanged  
Memory and offload

Lays out the three conditions for overlapping a kernel with a transfer (device support, non-default streams, pinned host memory) with timed measurements across GPU generations and a complete runnable sample.

### [NVIDIA CUDA Graphs documentation & PyTorch integration guide](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/introduction.html)
`free` · `reading list`  
one multi-page official guide plus the original PyTorch blog post · last activity 2025 (copyright-dated, current)  
Compilation and export

Official NVIDIA reference documentation on torch.cuda.CUDAGraph/torch.cuda.graph()/make_graphed_callables(): capture semantics, static-input constraints, common correctness pitfalls. Prose and snippets, no lab.

### [ONNX Backend Test suite (operator/opset conformance tests)](https://onnx.ai/onnx/repo-docs/OnnxBackendTest.html)
`free` · `reference implementation`  
hundreds of per-operator Node/Model test files across opset versions · last activity unknown (part of the actively maintained onnx/onnx repo)  
Compilation and export

The Node/Model test suite ONNX itself uses to certify that a runtime correctly implements each operator across opset versions, one Python/NumPy reference file per operator. Built for backend implementers, not learners.

### [ONNX Tutorials](https://github.com/onnx/tutorials)
`free` · `exercise repo`  
3.7k stars · last activity 2026-06  
Compilation and export

The official ONNX org's notebook collection covering exporting models from PyTorch/TensorFlow/scikit-learn and running them with various runtimes. Run-and-read notebooks with no correctness checker.

### [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
`free` · `course with labs`  
3.2k stars, 1k forks, 3000+ commits · last activity 2026 (main branch tracks OpenVINO 2026.2, current release)  
Compilation and export

Intel's official Jupyter notebook catalog for OpenVINO: model conversion to OpenVINO IR, quantization, and dozens of 'run this model' demos launchable in Colab/Binder.

### [Outlines](https://github.com/dottxt-ai/outlines)
`free` · `reference implementation`  
15.3k stars · last activity 2026-07  
Batching and serving

The most widely used structured/constrained-generation library: turns a JSON schema, regex or context-free grammar into a token-level mask via a finite-state-machine index over the vocabulary, so a model can only emit valid tokens. It is the library this area's structured-generation tasks model the mechanics of; you read the source or depend on it, there is nothing to be graded on.

### [Python behind the scenes (tenthousandmeters.com)](https://tenthousandmeters.com/blog/python-behind-the-scenes-13-the-gil-and-its-effects-on-python-multithreading/)
`free` · `reading list`  
numbered essay series, at least 13 posts by the visible numbering · last activity reported complete by ~2022 in secondary sources; could not confirm directly · *did not respond when last checked*  
Deep Python

Victor Skvortsov's deep-dive essay series reverse-engineering individual CPython mechanisms (dict implementation, generators, the GIL, async/await) straight from interpreter source. Could not be fetched live this session (connect ECONNREFUSED on every URL tried); included because search-result snippets show real, matching titles and content (post #13 = 'the GIL and its effects on Python multithreading', post #12 = 'how async/await works in Python'). Verify the domain loads before relying on it.

### [PyTorch Blog — Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
`free` · `interactive explainer`  
unknown (single post with runnable snippets) · last activity published 2025-03-05  
Memory and offload

Covers torch.utils.checkpoint, torch.compile's min-cut partitioner, Selective Activation Checkpointing (policy-driven save-vs-recompute), and the Memory Budget API that auto-tunes recompute fraction.

### [PyTorch Blog — Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)
`free` · `interactive explainer`  
unknown (single post, 2 runnable appendix scripts) · last activity published 2023-12-14, updated 2024-11-14  
Memory and offload

Narrated debugging case study: a ResNet50 loop missing optimizer.zero_grad(), diagnosed live with the Memory Snapshot/Profiler tools, with before/after code and full runnable appendices.

### [PyTorch official tutorials: torch.compile, torch.export, troubleshooting & Dynamo deep-dive](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
`free` · `interactive explainer`  
~4 long-form tutorial/guide pages plus API docs · last activity 2026 (docs main/current release branch, continuously updated)  
Compilation and export

Official PyTorch team tutorials with runnable code cells measuring torch.compile speedup and demonstrating graph breaks, guards, recompilation, and torch.export dynamic shapes/control-flow via torch.cond. The canonical written explanation of this area's core mechanics.

### [PyTorch — Understanding CUDA Memory Usage (+ memory_viz)](https://docs.pytorch.org/docs/main/torch_cuda_memory.html)
`free` · `interactive explainer`  
unknown (single docs page plus a standalone interactive tool) · last activity living page, part of actively-developed pytorch/pytorch docs  
Memory and offload

Official guide to torch.cuda.memory._record_memory_history()/_snapshot(), paired with a genuinely interactive local browser tool (pytorch.org/memory_viz) showing an Active Memory Timeline and Allocator State History with stack traces.

### [SGLang (RadixAttention / prefix caching)](https://github.com/sgl-project/sglang)
`free` · `reference implementation`  
30,751 stars; full serving engine, RadixAttention is one subsystem · last activity 2026-07-26  
Attention and KV cache

Production serving engine whose RadixAttention subsystem implements radix-tree-based prefix/prompt cache reuse across requests.

### [The Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/)
`free` · `interactive explainer`  
1 long-form book, ~4000 backing experiments, several embedded calculators · last activity 2025 (published, still served live, no dated recent edits found)  
LLM systems

Hugging Face/Nanotron's interactive web book on training LLMs across GPU clusters, grounded in ~4,000 real scaling experiments on up to 512 GPUs: DP/TP/sequence/context parallelism, pipeline schedules, ZeRO 1-3, activation recomputation, mixed precision, with embedded memory calculators and real profiler traces.

### [tomaarsen/attention_sinks](https://github.com/tomaarsen/attention_sinks)
`free` · `reference implementation`  
735 stars; drop-in transformers-API wrapper adding attention sinks · last activity 2024-04-10  
Attention and KV cache

Friendlier-to-install alternative to streaming-llm implementing the same sink/window idea as a transformers drop-in.

### [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning)
`free` · `interactive explainer`  
23.9k stars · last activity 2025-11  
Algorithms from scratch

Each algorithm pairs from-scratch Python code with an interactive Jupyter notebook explaining the underlying math. Narrower scope than the big repos: only linear regression, logistic regression, k-means, Gaussian anomaly detection, and an MLP - no PCA, k-NN, decision trees, or SVD. No tests or grading.

### [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
`free` · `reference implementation`  
1,173 stars; ~100 lines of raw CUDA, forward pass only · last activity 2024-12-30  
Attention and KV cache

The most-cited minimal raw-CUDA (not Triton) implementation of tiled attention with online softmax — short enough to read in one sitting.

### [tspeterkim/paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)
`free` · `reference implementation`  
149 stars; minimal block-table KV-cache manager on top of a Llama-3 forward pass · last activity 2024-08-26  
Attention and KV cache

A small, readable cache manager that reuses FlashAttention's PagedAttention kernel to show how block-table allocation and lookup actually work end to end.

### [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
`free` · `reference implementation`  
3,328 stars · last activity 2025-09  
Sparsity, pruning, distillation

DepGraph-based structural pruning library (CVPR 2023): automatically identifies dependency groups so pruning one layer correctly removes every coupled parameter across CNNs, ViTs, LLMs, and diffusion models.

### [Vidur](https://github.com/microsoft/vidur)
`free` · `reference implementation`  
646 stars · last activity 2025-07  
Batching and serving

Microsoft Research's LLM-inference-system simulator (MLSys 2024): configure a model, hardware, parallelism strategy and scheduling policy (including chunked prefill and speculative decoding), run a workload trace, and get TTFT/TPOT/batch-size numbers back without a GPU. A genuine systems-research tool for exploring scheduling tradeoffs; no notion of a correct answer to check against.

### [vLLM — PagedAttention design doc and kernel](https://docs.vllm.ai/en/latest/design/paged_attention/)
`free` · `reference implementation`  
repo has 87,188 stars; doc is a single walkthrough page, kernel is csrc/attention/attention_kernels.cu · last activity 2026-07-26  
Attention and KV cache

vLLM's own walkthrough of its PagedAttention CUDA kernel — block-structured KV cache, block tables, per-thread-group key access, softmax and write-out — explicitly labelled a historical explainer that defers to current source.

### [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
`free` · `reading list`  
1 paper, ~114 pages · last activity 2007  
CPU performance

Ulrich Drepper's canonical paper on DRAM, cache hierarchy, TLB, NUMA and prefetching. Static since 2007 but the hardware model is still accurate. No exercises.

### [wtfpython](https://github.com/satwikkansal/wtfpython)
`free` · `puzzle set`  
37k GitHub stars; roughly 60 examples across sections like 'Strain your brain!' and 'Slippery Slopes' · last activity 2025-05  
Deep Python

A curated collection of surprising, real Python snippets in 'guess the output, then read the explanation' format, hitting string interning, small-integer caching, is vs ==, mutable default arguments, refcounting, and GIL/threading effects.

### [zpoint/CPython-Internals](https://github.com/zpoint/CPython-Internals)
`free` · `reading list`  
5.1k GitHub stars, 478 forks · last activity 2026-02 (docs pinned to CPython 3.8.0a0 internals; recent commits look like wording/translation fixes, not a rebase onto 3.11+ internals)  
Deep Python

A diagram-heavy, structured walkthrough of CPython's C source: dict/int/str/generator internals, the GIL, gc, descriptors, exceptions, imports, threading/frames, C-extension writing. Pure explanatory notes, no exercises.


## How this was built

One research pass per area, each required to fetch every URL before reporting it and
explicitly forbidden from padding a short list with tangential material — an area with
three real resources returns three. Every URL was then checked independently of the agent
that found it.

Two limits worth knowing. It is a snapshot: several of these are dormant and any of them
can change. And absence of evidence is weak evidence — something that exists but is not
findable through GitHub search, Google, or the bibliographies of the standard books in an
area would not have been found. A missing or mischaracterised entry is a bug; open an issue.

Regenerate with `python3 tools/gen_resources.py` after re-surveying.