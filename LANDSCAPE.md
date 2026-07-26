# What else exists: the landscape, area by area

Every area of this bank already has other resources aimed at it. This page lists the real
ones, says plainly what each is, and marks whether it actually checks your work or just
shows you code. Where this bank is not the best option for something, that is written down
too — a page that claims to win everywhere would be useless to you.

**141 resources across 14 areas.** Every URL below was fetched and then
independently HTTP-checked (139 confirmed by the survey, all but two returning 200 on
recheck — the two exceptions are labelled inline). Last checked **2026-07-26**.

## How to read the labels

| Label | Meaning |
|---|---|
| **auto-graded** | a machine gives you a pass/fail verdict |
| **ships tests** | it includes tests you run yourself to check your own answer |
| **read only** | code or prose to study; nothing checks you |

The distinction matters more than it looks. Most of what exists in the applied areas is
read-only: excellent papers, production libraries and documentation, with nothing you can
attempt and be told you got wrong.

## Summary

| Area | Tasks here | What exists elsewhere | Auto-graded | Ships tests | Read only |
|---|---:|---|---:|---:|---:|
| [Deep Python](#deep-python) | 172 | **Some overlap** | 3 | 1 | 6 |
| [Deep C++](#deep-c) | 153 | **Some overlap** | 4 | 1 | 2 |
| [CPU performance](#cpu-performance) | 188 | **Some overlap** | 1 | 2 | 5 |
| [GPU / CUDA](#gpu--cuda) | 160 | **Crowded** | 5 | 3 | 3 |
| [Numerics and tensors](#numerics-and-tensors) | 194 | **Some overlap** | — | 1 | 6 |
| [Algorithms from scratch](#algorithms-from-scratch) | 73 | **Crowded** | 2 | 1 | 7 |
| [LLM internals](#llm-internals) | 192 | **Crowded** | 1 | 4 | 2 |
| [LLM systems](#llm-systems) | 200 | **Some overlap** | — | 4 | 8 |
| [Applied quantization](#applied-quantization) | 116 | **Adjacent only** | — | — | 11 |
| [Attention and KV cache](#attention-and-kv-cache) | 124 | **Some overlap** | 3 | 2 | 9 |
| [Compilation and export](#compilation-and-export) | 115 | **Adjacent only** | 1 | 1 | 6 |
| [Batching and serving](#batching-and-serving) | 128 | **Some overlap** | — | 1 | 7 |
| [Memory and offload](#memory-and-offload) | 112 | **Adjacent only** | — | — | 14 |
| [Sparsity, pruning, distillation](#sparsity-pruning-distillation) | 125 | **Adjacent only** | — | 1 | 13 |

Read down the three right-hand columns rather than the verdict: **6 of 14 areas have no
auto-graded resource at all**, and the four marked *Adjacent only* have nothing you can
practise against in any form.

## Deep Python

*172 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [Exercism](https://exercism.org/tracks/python), [PyBites Platform](https://pybitesplatform.com/bites/regular/), [Python Morsels](https://www.pythonmorsels.com/exercises/paths/).

### [Exercism — Python track](https://exercism.org/tracks/python)
`auto-graded` · `free` · `graded platform`  
Size: 146 exercises / 17 concepts; only ~4-5 (Descriptors, Iterators, Context Manager Customization, Class Customization, a Generators concept exercise) fall inside this area  
Last activity: 2026-07  

General-purpose Python practice track: write a solution, an automated test suite runs on submit, optional human mentor review afterward.

*Relation to this area:* Real auto-grading exists for a handful of this area's OOP-side topics, but it's a small corner of a mostly general-Python track — no coverage found for metaclasses, GIL, refcounting/gc, bytecode/dis, import machinery, weakrefs, interning, or asyncio internals.

### [PyBites Platform](https://pybitesplatform.com/bites/regular/)
`auto-graded` · `freemium` · `graded platform`  
Size: 435 'Bites'; free tier gives 30, lifetime access is $300  
Last activity: 2026 (live commercial platform; org repos show ongoing 2026 activity, no single commit date applies)  

Gamified bite-sized coding challenges; each submission is checked by an automated test suite plus a linter, with belts/leaderboard progress tracking.

*Relation to this area:* Covers decorators, generators, dunder methods, and context managers at a practical/idiomatic level, but no dedicated descriptors, metaclasses, GIL, gc, bytecode, or import-machinery content — overlaps only the shallow end of this area.

### [Python Morsels](https://www.pythonmorsels.com/exercises/paths/)
`ships tests` · `paid` · `graded platform`  
Size: 170+ exercises across 15 learning paths, including named descriptors / metaclasses / context-managers / generators-and-iterators paths  
Last activity: 2026 (paywalled article and exercise pages confirmed live this session)  

Trey Hunner's weekly-exercise subscription: an explanatory article on a protocol (e.g. descriptors: __get__/__set__/__set_name__, data vs non-data) followed by several small coding exercises, each shipping its own test file you run locally. $14-29/month or $120-240/year; 3 free preview exercises.

*Relation to this area:* The single closest topical match on this list for this area's data-model/descriptor/metaclass slice, at the Python-semantics level rather than the C level — but paid, and nothing here touches the GIL, refcounting, gc, bytecode, or import machinery.

### [guessthedis](https://github.com/cmyui/guessthedis)
`auto-graded` · `free` · `puzzle set`  
Size: 3 GitHub stars; 60+ built-in functions across difficulty tiers, needs Python 3.10+  
Last activity: 2026-04 (dependency-bump commit; small but not abandoned)  

A terminal game: you're shown a Python function and must type out its bytecode instructions line-by-line from memory; it checks your answer against the real dis output.

*Relation to this area:* Small and obscure, but the one resource found anywhere that is both auto-graded and specifically about bytecode/dis — this area's import-machinery, weakref, interning, and refcounting-arithmetic subtopics have nothing comparable even to this.

### [Advanced Python Mastery (David Beazley)](https://github.com/dabeaz-course/python-mastery)
`read only` · `free` · `exercise repo`  
Size: 12.8k GitHub stars; 9 sections, exercise files ex1_1.md–ex9_4.md  
Last activity: 2025-12  

David Beazley's free public release of his multi-day corporate Python training: slides, exercises, and full checked-in solutions you compare your own attempt against manually — no test runner or automated grading.

*Relation to this area:* Sections 'Inside Python Objects' (slots/descriptors), 'Metaprogramming' (metaclasses/decorators), and 'Iterators, Generators, and Coroutines' map almost one-to-one onto this area — closest topical match on this list, but zero automated grading.

### [CPython Internals: Your Guide to the Python 3 Interpreter (book + cpython-book-samples)](https://realpython.com/products/cpython-internals-book/)
`read only` · `paid` · `book with exercises`  
Size: first edition, CPython 3.9-era; companion repo has 285 stars and sample code for 9 chapters  
Last activity: companion repo last commit 2020-12-12 — over 5 years stale; book not revised for post-3.9 interpreter changes  
> **Link check 403 on 2026-07-26.** The host blocks automated fetchers; the resource itself is live.


Anthony Shaw's book walking a reader through the real CPython C source: compiling your own interpreter, then modifying core object types, generators, and memory management. The product page 403'd this fetcher; authenticity/content verified instead by fetching and reading the publicly-hosted sample-chapters PDF directly, and by fetching the companion code repo github.com/tonybaloney/cpython-book-samples (both succeeded and matched).

*Relation to this area:* The only resource found here covering compiling CPython from source at all — but dated and dormant, with grading limited to comparing your code against the author's own samples.

### [Fluent Python, 2nd ed. (book + example-code-2e)](https://github.com/fluentpython/example-code-2e)
`read only` · `paid` · `reading list`  
Size: 4.1k GitHub stars, 360 commits, code for 24 chapters  
Last activity: 2025-06  

Luciano Ramalho's book, organized explicitly around 'the Python data model' — dunder protocols, descriptors, __slots__, iterators/generators, coroutines, context managers, metaclasses — paired with this companion repo of runnable example code per chapter.

*Relation to this area:* Arguably the single most-recommended book for this area's data-model/dunder-protocol core, and broad across most of the OOP half besides — but prose-and-examples, nothing to submit or be graded on.

### [Python behind the scenes (tenthousandmeters.com)](https://tenthousandmeters.com/blog/python-behind-the-scenes-13-the-gil-and-its-effects-on-python-multithreading/)
`read only` · `free` · `reading list`  
Size: numbered essay series, at least 13 posts by the visible numbering  
Last activity: reported complete by ~2022 in secondary sources; could not confirm directly  
> **Link check 000 on 2026-07-26.** The host did not answer. DNS resolves and the Wayback Machine has a 200 snapshot from 2026-06-06, so the content existed recently — try it, and fall back to the archive if it is still down.


Victor Skvortsov's deep-dive essay series reverse-engineering individual CPython mechanisms (dict implementation, generators, the GIL, async/await) straight from interpreter source. Could not be fetched live this session (connect ECONNREFUSED on every URL tried); included because search-result snippets show real, matching titles and content (post #13 = 'the GIL and its effects on Python multithreading', post #12 = 'how async/await works in Python'). Verify the domain loads before relying on it.

*Relation to this area:* Alongside zpoint's repo, some of the best free writing specifically on this area's C-runtime half (GIL, async/await internals) — read-only, nothing to do.

### [wtfpython](https://github.com/satwikkansal/wtfpython)
`read only` · `free` · `puzzle set`  
Size: 37k GitHub stars; roughly 60 examples across sections like 'Strain your brain!' and 'Slippery Slopes'  
Last activity: 2025-05  

A curated collection of surprising, real Python snippets in 'guess the output, then read the explanation' format, hitting string interning, small-integer caching, is vs ==, mutable default arguments, refcounting, and GIL/threading effects.

*Relation to this area:* Best free treatment of this area's 'gotcha' subtopics (interning, refcounting, id/is) in puzzle form, but it's read-then-reveal — no code of your own is produced or graded.

### [zpoint/CPython-Internals](https://github.com/zpoint/CPython-Internals)
`read only` · `free` · `reading list`  
Size: 5.1k GitHub stars, 478 forks  
Last activity: 2026-02 (docs pinned to CPython 3.8.0a0 internals; recent commits look like wording/translation fixes, not a rebase onto 3.11+ internals)  

A diagram-heavy, structured walkthrough of CPython's C source: dict/int/str/generator internals, the GIL, gc, descriptors, exceptions, imports, threading/frames, C-extension writing. Pure explanatory notes, no exercises.

*Relation to this area:* The most topically complete single reference for this entire area on this list — but reading only, and several major CPython versions behind current interpreter internals.

**What none of these do.** Splits in two. The OOP/data-model half (dunders, descriptors, __slots__, metaclasses, generators, context managers) is well covered by real competitors, especially Python Morsels (paid, dedicated descriptor/metaclass/context-manager exercise paths) and dabeaz's python-mastery. The C-runtime half of this area — refcounting arithmetic, gc as a mechanism, bytecode/dis, import machinery, weakrefs, interning, asyncio's actual event loop — has almost no gradable practice anywhere: only a 3-star hobby CLI (guessthedis) for bytecode, everything else is read-only reference (zpoint's repo, tenthousandmeters, two books). No resource found combines auto-grading with this area's full breadth; that combination is where this bank is actually differentiated.

<details><summary>Survey notes for this area</summary>

Considered and excluded gregmalcolm/python_koans: real, self-checked-via-tests TDD exercises, but its "about_generators/about_decorating/about_with_statement" content is intro-level and it has no descriptors, metaclasses, GIL, gc, bytecode, import-machinery, weakref, interning, or asyncio content at all; also stale (last commit 2023-04). Left out to avoid padding. tenthousandmeters.com (the "Python behind the scenes" series) could not be fetched live this session — every request got connect ECONNREFUSED from this environment on the blog post, its resource-list page, and its homepage — so it is flagged verified:false in the report despite being a well-known, frequently-cited resource; content/title were corroborated only via search-result snippets. realpython.com/products/cpython-internals-book/ returned HTTP 403; I instead verified the book's authenticity/content directly by fetching and reading its publicly-hosted sample-chapters PDF, and verified its companion code repo (tonybaloney/cpython-book-samples, last commit 2020-12-12 — dormant). GitHub's REST API (api.github.com) was rate-limited from this environment for anonymous requests, so star counts/commit dates were obtained via WebFetch on the human-facing github.com pages instead (methodology consistent across all repos listed). PyBites is reachable at both codechalleng.es and pybitesplatform.com; I fetched and cite the pybitesplatform.com URLs since those are the ones that loaded content for me this session.

</details>

## Deep C++

*153 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [CppQuiz.org](https://cppquiz.org/), [Stanford CS106L assignments](https://github.com/cs106l/cs106l-assignments), [LearnCpp.com](https://www.learncpp.com/).

### [CppQuiz.org](https://cppquiz.org/)
`auto-graded` · `free` · `puzzle set`  
Size: 190 questions  
Last activity: unknown (actively used; no abandonment signal, but no single pinned update date)  

You're shown a short real C++ snippet and must predict its exact output, or flag it as a compile error, unspecified behaviour, or UB; submitting scores you immediately (1 point correct, penalties for hints/wrong attempts). Written by Anders Schau Knatten with input from Olve Maudal and other ACCU members.

*Relation to this area:* Overlaps heavily with our UB / ADL / overload-resolution slice in spirit (real compiled semantics, the bug is invisible) but tests recognition by reading, not by writing and compiling your own code.

### [Exercism — C++ track](https://exercism.org/tracks/cpp)
`auto-graded` · `free` · `exercise repo`  
Size: 100 exercises across 19 concepts  
Last activity: 2026-07  

Exercism's general-purpose C++ track: free, automated test-suite grading per exercise, plus optional human mentor review afterward.

*Relation to this area:* Broad idiomatic-C++ practice that incidentally touches classes and OOP, but isn't built to specifically corner a learner into a dangling reference, slicing bug, or SFINAE failure the way this area's tasks do.

### [HackerRank — C++ domain](https://www.hackerrank.com/domains/cpp)
`auto-graded` · `freemium` · `graded platform`  
Size: small — e.g. 5 problems in the Inheritance subdomain alone; other subdomains (Classes, STL, Debugging, Other Concepts) are comparably small  
Last activity: unknown (live commercial platform, not a repo; problems look untouched for years)  

Introductory OOP/C++ problems: classes, single/multi-level inheritance, virtual functions and abstract classes, basic function/class templates.

*Relation to this area:* Shallow overlap only — touches virtual functions and class design at a beginner level with no coverage of move semantics, UB, ADL, SFINAE/concepts, alignment, or exception safety.

### [LearnCpp.com](https://www.learncpp.com/)
`ships tests` · `free` · `book with exercises`  
Size: unknown exact count; dozens of chapters, roughly 200 lessons, each ending in a self-check quiz  
Last activity: 2025-01 (per the smart-pointers/move-semantics chapter's own revision date; the site is updated lesson-by-lesson, not in dated batches)  

A complete, free, widely used C++ tutorial book. Each chapter ends in a short quiz where the answer is hidden until you click to reveal it and compare against your own. Chapter 22 covers move semantics and smart pointers; sibling chapters cover virtual functions, templates and object relationships.

*Relation to this area:* The default place most people first learn every concept this bank tests (RAII, move semantics, rule of five, virtual dispatch, templates); teaches the why with no compiler in the loop and no measured score.

### [Stanford CS106L assignments](https://github.com/cs106l/cs106l-assignments)
`auto-graded` · `free` · `course with labs`  
Size: 7 assignments  
Last activity: 2026-07  

Public starter code and local autograders for Stanford's 1-unit Standard C++ Programming lab course. Assignment 6 is move semantics; assignment 7 has you implement your own unique_ptr (RAII, ownership transfer, operator overloading) and the autograder checks it.

*Relation to this area:* The one resource here where a learner actually builds a piece of this area's subject matter (a move-enabled smart pointer) and gets a real automated verdict back, though far narrower than this bank and silent on vtables, ADL, SFINAE, alignment and placement new.

### [C++ Templates (2nd ed.) study notes — downdemo/Cpp-Templates-2ed](https://github.com/downdemo/Cpp-Templates-2ed)
`read only` · `free` · `reference implementation`  
Size: ~1,600 GitHub stars, 15 chapters, mostly Chinese-language notes with runnable code  
Last activity: 2025-01  

A distilled, code-verified companion to Vandevoorde/Josuttis's "C++ Templates: The Complete Guide," covering function/class templates, variadics, SFINAE, traits, CRTP and C++20 concepts with runnable examples for each idea.

*Relation to this area:* Fills the one sub-topic (templates/SFINAE/concepts) that none of the graded or quiz resources above cover in depth, but it's reading-and-run-the-snippet material, not something you submit answers to.

### [Guru of the Week (GotW)](https://herbsutter.com/gotw/)
`read only` · `free` · `puzzle set`  
Size: 94+ numbered issues (original 88 plus ongoing additions)  
Last activity: unknown (ongoing; site shows revision/addition activity through 2026 but no single dated latest issue was visible)  

Herb Sutter's long-running column of C++ engineering puzzles: a scenario and a question (is this exception-safe, when does this template become ambiguous, what's wrong with this class), followed by a full worked solution. Basis for his "Exceptional C++" books.

*Relation to this area:* Canonical reading for exception safety, virtual dispatch, class mechanics and RAII in this area, written by one of the people who defined the idioms, but it's a worked-example archive with no automated check at all.

**What none of these do.** Nothing surveyed compiles a learner's own code with a real clang++ and scores it on a deterministic, tamper-proof number the way this bank does across the full breadth of this area. GotW and LearnCpp are read-then-self-check; CppQuiz is read-then-predict rather than write-then-compile; Exercism and HackerRank do compile and run submitted code but are general-purpose or shallow-OOP, not aimed at this area's specific traps. CS106L's autograder is the one genuine sibling — a learner builds a move-enabled unique_ptr and gets a real pass/fail — but it is 7 assignments, not 153, and never reaches vtables, ADL, alignment, placement new, or exception-safety guarantees. So the honest differentiator is breadth-with-one-deterministic-grader across the whole area, not an unclaimed topic: GotW and LearnCpp got to nearly every individual concept first and remain excellent for the conceptual "why."

<details><summary>Survey notes for this area</summary>

gotw.ca (the canonical archive domain) failed to load via WebFetch with a TLS handshake error every time it was tried; herbsutter.com/gotw/ mirrors the same content and was used instead — treat GotW as verified via that mirror, not the .ca domain. HackerRank problem counts are per-subdomain samples (Inheritance = 5), not a confirmed total across all C++ subdomains — exact site-wide count wasn't obtainable without an account. LearnCpp's own homepage footer rendered a stale "©2024" copyright in one fetch even though an individual chapter carries a 2025-01 revision date and the site is known to be continuously live-edited; I used the chapter-level date as more trustworthy. cppquiz.org also sells a companion book ("C++ Brain Teasers") by the same author — the quiz itself is free, flagged in case that reads as a paywall signal. No GitHub API calls succeeded (rate-limited every attempt from this IP); all repo activity dates were confirmed instead via the GitHub web commit-history pages through WebFetch.

</details>

## CPU performance

*188 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [perf-ninja](https://github.com/dendibakh/perf-ninja), [CS:APP Cache Lab](https://csapp.cs.cmu.edu/3e/labs.html), [Computer Enhance: Performance-Aware Programming](https://www.computerenhance.com/p/table-of-contents).

### [CS:APP Cache Lab (CMU 15-213)](https://csapp.cs.cmu.edu/3e/labs.html)
`ships tests` · `free` · `course with labs`  
Size: 1 lab, 2 parts (cache simulator + matrix-transpose optimization), part of an 11-lab course  
Last activity: 2014 (self-study handout date); lab unchanged since CS:APP3e (~2015)  

Students write a cache simulator (csim.c) and optimize a matrix-transpose kernel (trans.c), scored by a driver script on exact cache-miss counts against a Valgrind-traced reference — a fixed deterministic metric. A self-study handout (cachelab-handout.tar) lets non-CMU learners run it independently. Verified by fetching the labs page and the cachelab.pdf handout directly.

*Relation to this area:* Same grading philosophy as this bank (deterministic miss-count, not wall clock) but covers only cache blocking/associativity, one of fourteen subtopics here, and hasn't been updated in over a decade.

### [Computer Enhance: Performance-Aware Programming](https://www.computerenhance.com/p/table-of-contents)
`ships tests` · `paid` · `course with labs`  
Size: 5 parts + bonus material, weekly homework  
Last activity: 2023-01 (series launch); could not confirm if still being extended  

Casey Muratori's language-agnostic video course covering cache-size/bandwidth testing, cache indexing, branch prediction, and SSE intrinsics with weekly homework. Homework is self-checked against community solutions on GitHub, no autograder. Confirmed paywalled (a subscriber comment on the page states they paid specifically for this course; no exact price found).

*Relation to this area:* Overlaps on cache/SIMD/branch-prediction theory and hands-on homework, but ungraded and paywalled.

### [perf-ninja](https://github.com/dendibakh/perf-ninja)
`auto-graded` · `free` · `exercise repo`  
Size: 20+ labs (9 Core Bound, 9 Memory Bound, 4 Bad Speculation, 4 Misc); 3,787 stars, 388 forks  
Last activity: 2026-07-16  

C++ course (Rust/Zig ports exist) of small realistic kernels — false sharing, loop tiling/interchange, prefetching, huge pages, alignment, vectorization, branch prediction, lookup tables. You optimize a lab, PR it, and CI checks correctness plus a wall-clock speedup threshold (Google Benchmark) on real hardware (Alderlake/Zen3/M1). Verified via README fetch, GitHub API metadata, and direct read of the false-sharing lab + its CI mechanism.

*Relation to this area:* Overlaps heavily — nearly a 1:1 topic match with this area's syllabus, but grades via real-hardware wall-clock speedup rather than a deterministic simulated metric.

### [Agner Fog's optimization manuals](https://www.agner.org/optimize/)
`read only` · `free` · `reading list`  
Size: 5 manuals (C++ optimization, asm optimization, microarchitecture, instruction tables, calling conventions)  
Last activity: 2026-07 (most recently updated manual, confirmed via direct fetch)  

The reference of record for x86/x86-64 instruction latencies/throughputs and microarchitectural detail (out-of-order execution, branch prediction internals). Actively maintained. Pure reference, no exercises.

*Relation to this area:* Background reference a learner would consult to understand why optimizations in this area's tasks work; not a competing practice resource.

### [Algorithms for Modern Hardware (Algorithmica)](https://en.algorithmica.org/hpc/)
`read only` · `free` · `reading list`  
Size: ~9 chapters in the Performance Engineering part  
Last activity: 2024-08 (underlying repo last push); author called it ~75% complete in 2022, appears stalled there  

Free online book by Sergey Slotin: dense exposition plus C++ snippets and case studies (e.g. faster matrix multiplication, faster binary search) covering cache lines, associativity, prefetching, SIMD intrinsics, branch prediction. No exercises, no grading. Verified via direct fetch and GitHub API on algorithmica-org/algorithmica.

*Relation to this area:* Covers nearly the same conceptual ground as this area's theory, but as reading only — nothing to submit or get graded.

### [Data-Oriented Design (Richard Fabian)](https://www.dataorienteddesign.com/dodbook/)
`read only` · `freemium` · `reading list`  
Size: 1 book, ~9 chapters  
Last activity: 2018  

Free reduced online version (paid paperback for full version) treating SoA vs AoS and data-oriented layout as a first-class subject, mostly through a game-dev lens (components, hierarchical LOD). No exercises.

*Relation to this area:* The one resource that centers this area's SoA/AoS and data-oriented-layout topic rather than treating it as an aside.

### [Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)
`read only` · `free` · `reference implementation`  
Size: 7 worked examples  
Last activity: 2010  

Single classic blog post by Igor Ostrovsky with runnable C# snippets isolating cache effects: stride vs line size, L1/L2 capacity cliffs, associativity conflicts, false-dependency ILP, and a 15x false-sharing slowdown demo. Confirmed live via direct fetch. Article to read/re-run, not an exercise with a verdict.

*Relation to this area:* Canonical intuition-building reading for cache-line/false-sharing behavior that this area grades directly.

### [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
`read only` · `free` · `reading list`  
Size: 1 paper, ~114 pages  
Last activity: 2007  

Ulrich Drepper's canonical paper on DRAM, cache hierarchy, TLB, NUMA and prefetching. Static since 2007 but the hardware model is still accurate. No exercises.

*Relation to this area:* Foundational background reading for the memory-hierarchy topics this area grades on (TLB, NUMA, prefetch, bandwidth).

**What none of these do.** Nobody grades this breadth of topic (cache blocking, false sharing, branch prediction, SIMD, alignment, prefetch, TLB, NUMA, SoA/AoS) with a single deterministic, non-wall-clock metric end-to-end. perf-ninja has the closest topic match but its pass/fail is a real-hardware wall-clock benchmark threshold — machine-dependent by construction, exactly what this bank's simulator avoids. CS:APP's Cache Lab shares our deterministic-metric philosophy but covers one topic out of fourteen and hasn't been touched in over a decade. Nothing spans both C++ and Python, and nothing presents the material as a large resumable task bank (188 tasks) rather than one lab, one video series, or one book. Where we are honestly not ahead: perf-ninja is a better teacher for several of these topics, with narrated walkthroughs and real perf-counter analysis this bank doesn't provide.

<details><summary>Survey notes for this area</summary>

"Algorithmica" name collision: en.algorithmica.org / GitHub org algorithmica-org is Sergey Slotin's HPC book, unrelated to the Springer journal of the same name — worth a disambiguation line if this gets published. CS:APP Cache Lab is old (handout dated 2014, lab content from CS:APP3e ~2015) but still functional since it's a self-run deterministic check, not a live service. Algorithmica's repo has had no commits since 2024-08 (~2 years) — book appears stalled at ~75% per author's own 2022 status note. Computer Enhance is paywalled and I could not find an exact price or confirm whether it's still actively adding parts (launched 2023-01). GitHub API access via the sandboxed proxy IP was rate-limited (had to use authenticated `gh api` instead, which worked). Deliberately left out: Mojo GPU Puzzles (GPU-kernel context, not CPU SIMD, would be padding), CoffeeBeforeArch false-sharing blog and various tiny GitHub false-sharing microbenchmarks (reference snippets, not exercises, redundant with the Ostrovsky gallery + perf-ninja), MIT 6.172 OpenCourseWare (real and free, covers caching/ILP, but I could not confirm from the page whether homework packages ship tests/autograders, and a direct fetch of a homework resource page hit a redirect loop — leaving it out rather than reporting unverified grading behavior).

</details>

## GPU / CUDA

*160 tasks in this bank · **Crowded** — several resources let you write code here and get a verdict back*

**Start here if you want to be graded:** [LeetGPU](https://leetgpu.com/), [Tensara](https://tensara.org/), [GPU MODE / KernelBot](https://www.gpumode.com/).

### [GPU MODE / KernelBot](https://www.gpumode.com/)
`auto-graded` · `free` · `graded platform`  
Size: 8 problem series/competitions (PMPP practice set plus sponsored contests: AMD $100K, AMD $1.1M, NVIDIA Blackwell NVFP4, BioML, Helion hackathon, linear algebra)  
Last activity: 2026-07  

Competitive-kernel wing of the GPU MODE community (formerly CUDA MODE). Submit via Discord bot or the popcorn-cli, run on real sponsored/donated GPUs, ranked on a public leaderboard. Companion repo gpu-mode/reference-kernels holds the problem sets.

*Relation to this area:* Same real-hardware-speed model as LeetGPU/Tensara; oriented toward competition rather than a fixed curriculum.

### [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)
`ships tests` · `free` · `puzzle set`  
Size: 14 puzzles, 12,341 GitHub stars  
Last activity: 2024-09  

The famous one. Python via NUMBA's CUDA JIT (not CUDA-C), each puzzle self-checked in-notebook against a NumPy reference. Designed to run on a real GPU (recommended: Colab GPU runtime), not a simulator. Quiet for ~22 months but still the most-cited resource in this space.

*Relation to this area:* Builds the same thread/block/shared-memory intuition at a much lower level of formality; does not measure coalescing, bank conflicts, or divergence.

### [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
`auto-graded` · `free` · `benchmark or leaderboard`  
Size: 250 tasks (100 Level-1 single-op, 100 Level-2 fused-op, 50 Level-3 full-architecture), 1,157 GitHub stars  
Last activity: 2026-03  

Stanford Scaling Intelligence Lab benchmark built to answer 'can an LLM write a fast CUDA/Triton kernel' — scores correctness plus a fast_p speedup ratio against a PyTorch reference on real GPU wall-clock time. A human can run it against their own Level-1 kernels, but the harness and surrounding papers target automated kernel generation, not a learner's curriculum.

*Relation to this area:* Adjacent, not a practice resource in the normal sense — included for completeness since it's the closest thing to a formal 'benchmark' in this space.

### [LeetGPU](https://leetgpu.com/)
`auto-graded` · `freemium` · `graded platform`  
Size: 70+ challenges (counted on the live challenges page: ~19 Easy, ~46 Medium, ~13 Hard)  
Last activity: unknown  

Browser CUDA/Triton/PyTorch/Mojo/CuTe-DSL/JAX judge. Verified via live browser navigation: homepage now says 'Execute high-performance GPU programs instantly on real hardware in your browser' — a shift from its 2025 launch, which ran purely on a CPU emulator with 'functional' and 'cycle accurate' (architecture-modeling) modes per its Show HN post. Has a CLI, global leaderboard, open problem-contribution repo, and a visible 'Pro' tier in-nav.

*Relation to this area:* Overlaps heavily on problem topics (reduction, prefix sum, histogram, matrix transpose, GEMM) but grades on correctness + real-hardware relative speed, not on transaction/bank-conflict/divergence counts.

### [Oak Ridge OLCF CUDA Training Series](https://github.com/olcf/cuda-training-series)
`ships tests` · `free` · `course with labs`  
Size: 13 sessions/homeworks (hw1–hw11+), 1,022 GitHub stars  
Last activity: 2024-08  

National-lab training series (slides at olcf.ornl.gov/cuda-training-series, confirmed live): one deck + one hands-on exercise per topic (CUDA C++ basics, shared memory, fundamental optimization parts 1-2, atomics/reductions/warp shuffle, streams, cooperative groups). Live sessions ran Jan 2020–Oct 2021 and have not resumed; the repo is a finished, static artifact now, not a running course. You compile and self-check output against README-stated expected values.

*Relation to this area:* Directly overlaps shared memory, warp shuffle, and atomics topics at the 'write the kernel' level, not the 'measure the transaction count' level; dormant but stable.

### [SW Online Judge (formerly CUDA Online Judge / cudaforces)](https://swforces.com/)
`auto-graded` · `free` · `graded platform`  
Size: problems split Easy/Medium/Hard, exact count not exposed without an account; 43 GitHub stars on the judge-engine repo (SungHwanYun/cudaforces)  
Last activity: 2026-01  

Transpiles submitted CUDA-C to OpenMP C++ and runs it on CPU ('CUDA Code → Transpiler + Validate → C++ Code (OpenMP) → CPU Execute & Judge'). The project states plainly: 'Performance benchmarking is not available — the platform is for correctness verification only.' cudaforces.com now 301-redirects here; the product has broadened to general algorithms and Linux systems too.

*Relation to this area:* Closest philosophical cousin (real CUDA-C, genuinely no GPU hardware needed) but checks output correctness only, never memory-access pattern or divergence.

### [Tensara](https://tensara.org/)
`auto-graded` · `free` · `graded platform`  
Size: 84 problems (site states '84 of 84... working towards 100')  
Last activity: 2026-07  

Open-source (GPL-3.0), sponsor-funded (Modal) CUDA/Triton/Mojo judge. Submissions compile with the real NVIDIA toolchain and run sandboxed on real T4/A100/H100 hardware, ranked on a per-GPU leaderboard against cuBLAS/PyTorch/Triton baselines.

*Relation to this area:* Same category as LeetGPU: real hardware, wall-clock/relative speed as the score, correctness verified but no architectural-metric grading.

### [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)
`ships tests` · `free` · `puzzle set`  
Size: 2,539 GitHub stars (part of a 7-puzzle-series family by the same author)  
Last activity: 2026-04  

Sister project to GPU-Puzzles for Triton, a Python DSL that compiles to GPU code. Explicitly does not need a real GPU — runs on a Triton interpreter — and is not CUDA-C.

*Relation to this area:* Different abstraction level (block-level Triton ops); Triton hides warp divergence and bank conflicts entirely, so no topic overlap on those specific mechanics.

### [GPU MODE lectures](https://github.com/gpu-mode/lectures)
`read only` · `free` · `course with labs`  
Size: 6,358 GitHub stars; ongoing series of recorded lecture notebooks  
Last activity: 2026-06  

Jupyter-notebook lecture materials (paired with recorded video) covering kernel fundamentals through Triton, CUTLASS, and quantization. No submission or checking system — you read/run the notebooks yourself.

*Relation to this area:* Strong syllabus/worked-example source for the same topics (warps, shared memory, coalescing) but purely expository, not graded.

### [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
`read only` · `free` · `reference implementation`  
Size: 200+ CUDA kernels, GPL-3.0, 11,632 GitHub stars  
Last activity: 2026-07  

Despite the name, not a judge — a progressively-harder reference-kernel library (easy to hard++) with PyTorch bindings and benchmark tables against cuBLAS/cuDNN, covering GEMM/GEMV, FlashAttention variants, and Tensor-Core paths. Nothing to submit; you read and adapt the code.

*Relation to this area:* Best free source of 'what does a genuinely fast kernel look like' once past puzzle-level exercises; no grading of any kind.

### [Programming Massively Parallel Processors (PMPP), 4th ed. + community solutions](https://github.com/tugot17/pmpp)
`read only` · `paid` · `book with exercises`  
Size: 20 chapters of end-of-chapter exercises; this solutions repo has 817 GitHub stars and covers all of them in CUDA C and Python  
Last activity: 2025-06  

The field's standard textbook (Kirk, Hwu & Hajj). No official autograder for its exercises; most self-learners cross-check against community solution repos like this one (several similar ones exist: nvixnu, guanrenyang, Isalia20). Book itself is paid; the solutions repo is free and MIT-licensed.

*Relation to this area:* The reference syllabus most other resources here (OLCF series, GPU MODE's practice set) are explicitly built to accompany; covers our exact topics (coalescing, tiling, atomics, scan, histogram) as prose + exercises, not as a graded bank.

**What none of these do.** This corner of programming education is genuinely crowded: real-GPU leaderboards (LeetGPU, Tensara, GPU MODE's KernelBot), a CPU-transpiler correctness judge (SW Online Judge/cudaforces), two well-known self-checked puzzle notebooks (GPU-Puzzles, Triton-Puzzles), and a 200+-kernel reference library (LeetCUDA) all let a learner write and check real CUDA/Triton code today, largely for free. We did not find any of them grading on the hardware-model counters themselves (coalescing-transaction counts, shared-memory bank-conflict counts, warp-divergence counts) as a deterministic, machine-independent pass/fail metric — every grader we found is either correctness-only or a real-hardware wall-clock/relative-speed number on shared, non-reproducible hardware. That is a real, defensible differentiator for our 146 CUDA tasks. It is not a differentiator on raw kernel-writing breadth or performance-tuning competition, where LeetCUDA, PMPP's solution repos, and the LeetGPU/Tensara/KernelBot leaderboards are more mature and more used than we are.

<details><summary>Survey notes for this area</summary>

cudaforces.com now 301-redirects to swforces.com — same team/GitHub repo (SungHwanYun/cudaforces), rebranded to a broader 'SW Online Judge' covering CUDA + general algorithms + Linux systems, not CUDA-only anymore. LeetGPU and Tensara are both client-rendered SPAs; static curl/WebFetch returns only the JS shell, so their content (pricing, challenge lists, real-vs-emulated hardware claims) was verified via an actual browser navigation + accessibility snapshot, not plain HTTP fetch. LeetGPU's original 2025 launch was pure CPU emulation (functional + 'cycle accurate' modes, no GPU needed); as of this check its marketing copy says it now runs on real hardware, and a 'Pro' tier button is visible in its nav though no pricing page rendered without login — treat cost as freemium with unconfirmed paid-tier scope. Per the task's calibration anchors: en.algorithmica.org's free HPC book has NO shipped GPU/CUDA chapter (fetched en.algorithmica.org/hpc/complexity/gpu/, got 404) — it's a good CPU-performance-area anchor but does not apply here. deep-ml.com was also checked directly and has no GPU/CUDA category at all (pure NumPy/PyTorch ML-algorithm problems) — excluded from the list. GitHub's public REST API was rate-limited on this network; all GitHub stats were fetched instead via the authenticated `gh api` CLI in this session, which returns the same live data.

</details>

## Numerics and tensors

*194 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles).

### [Tensor Puzzles (srush)](https://github.com/srush/Tensor-Puzzles)
`ships tests` · `free` · `puzzle set`  
Size: 21 puzzles  
Last activity: 2024-03  

Reimplement NumPy/PyTorch primitives (ones, sum, outer, diag, cumsum, scatter_add, bincount, etc.) in one line using only broadcasting, arithmetic, comparison, @ and indexing - no library shortcuts. Each puzzle ships a Hypothesis-based run_test() checker you run locally that gives pass/fail plus a broadcast-shape diagram on failure.

*Relation to this area:* Overlaps heavily with the broadcasting/indexing/no-explicit-loop portion of this area; does not touch floating-point precision at all.

### [100 NumPy exercises (rougier/numpy-100)](https://github.com/rougier/numpy-100)
`read only` · `free` · `exercise repo`  
Size: 100 exercises  
Last activity: 2026-07  

The canonical NumPy exercise list pulled from the mailing list, Stack Overflow and the docs, shipped as no-solution / hints / full-solutions markdown and notebook versions. You self-check by diffing against the solutions file; there is no automated grader.

*Relation to this area:* Touches strides, views, dtype and broadcasting incidentally as general array-fu, but is organized around 'how do I do X in NumPy' rather than 'why does NumPy behave this way' - only loosely overlapping with this area's floating-point focus.

### [Float Exposed](https://float.exposed/)
`read only` · `free` · `interactive explainer`  
Size: single-page tool  
Last activity: unknown  

Type a decimal or flip individual bits of a half/bfloat16/float/double and see the exact base-10 value, base-2 breakdown, and the exact delta to the next/previous representable value (local ULP/epsilon). Pure visualization, no exercises.

*Relation to this area:* Best available hands-on tool for the epsilon/denormals/rounding portion of this area's floating-point bullet; nothing to grade against.

### [From Python to NumPy (rougier)](https://github.com/rougier/from-python-to-numpy)
`read only` · `free` · `book with exercises`  
Size: ~9 chapters  
Last activity: 2025-05  

Free open-access book; Chapter 2 'Anatomy of an array' is a careful, code-backed treatment of strides, memory layout, views vs copies and reshaping cost. Later chapters are vectorization exercises you check by comparing your own timing/output, not an automated grader.

*Relation to this area:* Directly covers this area's 'NumPy strides and views vs copies' bullet in more depth than any exercise set found.

### [array-api-tests (data-apis)](https://github.com/data-apis/array-api-tests)
`read only` · `free` · `reference implementation`  
Size: full Array API spec surface  
Last activity: 2026 (active)  

A conformance test suite for the Python Array API standard, not a learning resource per se. Encodes NumPy-style dtype-promotion rules as executable assertions rather than prose; a learner can point ARRAY_API_TESTS_MODULE at NumPy and run/read the promotion tests to see the real casting table exercised against real inputs.

*Relation to this area:* The only rigorous, runnable artifact found anywhere for this area's 'dtype promotion' bullet, though it is a spec-compliance harness for library authors, not a pedagogical exercise set.

### [fastai Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra)
`read only` · `free` · `course with labs`  
Size: 8 lecture notebooks  
Last activity: 2017-07  

Rachel Thomas's USF course notebooks (with YouTube lectures). Lecture 3 covers stability of LU with/without pivoting, Lecture 6 has a section explicitly on 'Conditioning & Stability' (why matrix inversion is unstable), Lecture 8 compares classical vs modified Gram-Schmidt stability. No autograder; you run the notebooks and compare to shown output.

*Relation to this area:* The one resource found that treats condition number as a hands-on, code-backed idea rather than a textbook definition; dead since 2017 but the numerical content hasn't aged even though library APIs may have.

### [fp-conv](https://sw23.github.io/fp-conv/)
`read only` · `free` · `interactive explainer`  
Size: single-page tool, ~15 formats  
Last activity: 2025  

Same click-a-bit/type-a-value interaction as float.exposed but covering the full modern ML dtype zoo explicitly: fp64/fp32/fp16, bf16, tf32, and the OCP microscaling fp8 (e4m3, e5m2), fp6 (e3m2, e2m3) and fp4 (e2m1) formats, plus custom user-defined bit layouts. A near-duplicate exists at kuterdinel.com/float-gallery.html covering the same fp8/fp6/fp4/bf16/tf32 comparison in one static gallery.

*Relation to this area:* Directly matches this area's 'fp16/bf16/fp32/fp8 ranges and rounding' bullet for the newer ML-specific formats that float.exposed doesn't cover.

**What none of these do.** Broadcasting/indexing/strides array-mechanics is reasonably well served (numpy-100's 100 exercises, Tensor-Puzzles' auto-checked no-loop implementations), and IEEE-754/bf16/fp8 bit-level exploration is excellent thanks to two independent interactive visualizers. But nobody grades the sharper half of this area: no exercise set anywhere checks a learner's Kahan/compensated-summation implementation against a tolerance, grades a stable-softmax/log-sum-exp implementation against an overflow-triggering input, scores detection of an in-place aliasing bug, quizzes dtype-promotion predictions against real NumPy casting rules, or tests deterministic-vs-non-deterministic reduction order. Those topics exist only as blog posts, papers, or passive visualizers - never as something you submit and get scored on. That half of the area is where this bank has no real graded competitor.

<details><summary>Survey notes for this area</summary>

Excluded from the resources list after checking, to avoid padding: HackerRank's "Python > NumPy" domain (10 auto-graded challenges, verified real and free) exists but only drills basic array-construction syntax (reshape/concatenate/eye/sum) with no connection to floating-point mechanics, aliasing, or promotion - it's a syntax quiz, not part of this area's real content. w3resource's "100 NumPy Broadcasting problems" page is a low-effort SEO content-mill duplicate of numpy-100 with no grading, also excluded. kuterdinel.com/float-gallery.html is a near-duplicate of the fp-conv tool (same fp8/fp6/fp4/bf16/tf32 bit-comparison idea); mentioned inline in the fp-conv entry rather than as its own card. GitHub REST API was rate-limited from this network (403), so repo last-activity dates were confirmed by fetching each repo's /commits page directly instead of the API.

</details>

## Algorithms from scratch

*73 tasks in this bank · **Crowded** — several resources let you write code here and get a verdict back*

**Start here if you want to be graded:** [deep-ml.com](https://www.deep-ml.com/problems), [CS231n Assignment 1](https://cs231n.github.io/assignments2026/assignment1/), [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction).

### [CS231n Assignment 1 (Stanford)](https://cs231n.github.io/assignments2026/assignment1/)
`ships tests` · `free` · `course with labs`  
Size: 1 assignment, 5 parts (kNN / SVM / Softmax / two-layer net / image features)  
Last activity: 2026 (current live edition)  

Stanford computer-vision course, actively run with a 2026 edition. Implement k-NN, SVM loss, a softmax classifier, and a two-layer neural network with backprop derived and coded by hand in raw NumPy, with gradient-check and expected-loss sanity-check cells built into the Colab notebooks for self-verification.

*Relation to this area:* Directly overlaps our 'softmax regression' and 'backprop by hand' tasks with a well-known, currently-maintained, free university assignment.

### [Machine Learning Specialization (Andrew Ng / DeepLearning.AI + Stanford)](https://www.coursera.org/specializations/machine-learning-introduction)
`auto-graded` · `paid` · `course with labs`  
Size: 3 courses; named practice labs for decision trees, anomaly detection, k-means, PCA confirmed  
Last activity: 2022 revision, still the live served version in 2026  

Current Python/NumPy successor to Andrew Ng's original Octave ML course. Programming labs implement linear/logistic regression and basic neural nets from scratch, plus hands-on labs for decision trees/ensembles, k-means, PCA, and anomaly detection, graded by Coursera's own grader for a certificate.

*Relation to this area:* Overlaps most of our supervised+unsupervised topic list in a guided, fill-in-the-blank lab format rather than an open implement-from-spec format; paid ($49/mo, financial aid available).

### [deep-ml.com](https://www.deep-ml.com/problems)
`auto-graded` · `freemium` · `graded platform`  
Size: 100+ problems (site's claim)  
Last activity: unknown (live product, not a repo)  

Browser-based LeetCode-style ML site: write Python in an in-browser editor, get pass/fail against hidden tests instantly. Catalog (confirmed via secondary sources, not directly scrapeable since JS-rendered) includes linear regression via normal equation/gradient descent, k-NN, decision trees, k-means, PCA, SVD alongside DL/NLP/CV problems.

*Relation to this area:* Closest analog to our bank in spirit for this area - it is actually graded, not just reference code.

### [Data Science from Scratch (Joel Grus) - book + code](https://github.com/joelgrus/data-science-from-scratch)
`read only` · `freemium` · `book with exercises`  
Size: 27 chapters  
Last activity: 2020-09  

Builds k-NN, k-means, hierarchical clustering, decision trees, gradient descent, and a simple neural net using plain Python lists with no NumPy at all, specifically to force understanding of every step. Code-along narrative with 'for further exploration' prompts rather than checked exercises; the book text is paid, the code repo is free/MIT.

*Relation to this area:* Directly overlaps our from-scratch topic list but with a stricter no-library philosophy (not even NumPy) and a teaching-book format instead of graded tasks.

### [Neural Networks and Deep Learning (Michael Nielsen)](https://github.com/mnielsen/neural-networks-and-deep-learning)
`read only` · `free` · `book with exercises`  
Size: 6 chapters + appendices  
Last activity: unknown exact date; written for Python 2.6-2.7, author states no further updates planned - finished/frozen, not abandoned mid-way  

The classic free derivation of backprop by hand: chapter 2 works through the four backpropagation equations from first principles, chapter 1 hand-codes an MNIST classifier in raw Python with only NumPy for matrix ops. End-of-section 'problems' are unanswered prose exercises, not auto-checked.

*Relation to this area:* Directly overlaps our 'backprop by hand' task with the single most-cited derivation of the algorithm, though ungraded and old code.

### [ddbourgin/numpy-ml](https://github.com/ddbourgin/numpy-ml)
`read only` · `free` · `reference implementation`  
Size: 16.3k stars  
Last activity: 2022-01 (dormant ~4.5 years)  

Documented, more rigorous reference implementations: CART decision trees, bagging/random forests/GBTs, GMM trained with actual EM, HMMs, Bayesian linear regression, Gaussian processes, plus SGD/AdaGrad/RMSProp/Adam optimizers. Ships its own internal test suite for the maintainer's code correctness, not for grading a learner's solutions.

*Relation to this area:* Overlaps heavily, and is the best of these repos for explicit EM-trained GMM; GPL-3.0 license, dormant but comprehensive.

### [eriklindernoren/ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
`read only` · `free` · `reference implementation`  
Size: 32.4k stars, 374 commits  
Last activity: 2019-10 (dead, no commits in ~6.75 years)  

The most-starred 'ML from scratch' repo in existence; bare-bones NumPy code for nearly every classic algorithm (regression variants, trees, ensembles, k-NN, k-means, GMM, PCA, DBSCAN, SVM, plus autoencoder/GAN/RBM). Read-only, no exercises, no grading, and unmaintained since 2019.

*Relation to this area:* Overlaps heavily with our area's topic list; the canonical reference every learner already finds first, but frozen code not a practice tool.

### [insdout/ML-Algorithms-From-Scratch](https://github.com/insdout/ML-Algorithms-From-Scratch)
`read only` · `free` · `reference implementation`  
Size: 2 stars, 165 commits  
Last activity: unknown  

Personal study repo, notable for being one of the very few things found anywhere that explicitly implements QR decomposition and eigendecomposition/SVD alongside the usual k-means/k-NN/decision-tree/GMM-EM/PCA/random-forest set. Extremely low visibility - a learner would have to already know to search for it.

*Relation to this area:* Direct topical overlap on the numerical-linear-algebra half of our area (SVD/QR/power iteration) that almost nothing else covers, but obscure and unmaintained-looking.

### [rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)
`read only` · `free` · `reference implementation`  
Size: 11.2k stars, 151 commits  
Last activity: 2026-05  

Minimal, clean NumPy implementations of linear/logistic regression, k-NN, k-means, GMM, Naive Bayes, PCA, SVM, random forests, gradient boosting, factorization machines, t-SNE. No tests, no grader, read-only reference code.

*Relation to this area:* Overlaps heavily with our supervised/unsupervised classic-ML tasks; complements by showing production-lean code style.

### [trekhleb/homemade-machine-learning](https://github.com/trekhleb/homemade-machine-learning)
`read only` · `free` · `interactive explainer`  
Size: 23.9k stars  
Last activity: 2025-11  

Each algorithm pairs from-scratch Python code with an interactive Jupyter notebook explaining the underlying math. Narrower scope than the big repos: only linear regression, logistic regression, k-means, Gaussian anomaly detection, and an MLP - no PCA, k-NN, decision trees, or SVD. No tests or grading.

*Relation to this area:* Covers a subset of our topics with a different pedagogical angle (math walkthrough vs. code dump); complements rather than duplicates the bigger reference repos.

**What none of these do.** On breadth we are not differentiated: every algorithm in this area (k-means, PCA, k-NN, gradient descent, linear/logistic regression, decision trees, GMM/EM, backprop) already has multiple free, well-known NumPy/pure-Python implementations to read, plus two graded/semi-graded routes (deep-ml.com's auto-grader, Andrew Ng's Coursera labs). Where we differ is mechanism, not topic: none of these give a starter verified to fail and a reference verified to pass with a fully deterministic score - the reference repos are read-only, the book is narrative code-along, and even the graded platforms (deep-ml.com, Coursera) are fill-in-the-blank or hidden-unit-test style rather than implement-the-whole-thing-from-a-spec. The one topic nobody serves well is the numerical-linear-algebra half of the list (SVD, QR, power iteration) - it appears almost nowhere except a 2-star personal repo.

<details><summary>Survey notes for this area</summary>

neuralnetworksanddeeplearning.com (the actual book site, not the code repo) currently fails to load via HTTPS - its TLS certificate lists only github.com/*.github.com as valid names, i.e. the domain looks misconfigured right now. Linked the code repo instead and marked the book-site URL unverified in the writeup; a reader clicking straight to the book domain may hit the same certificate error. deep-ml.com's problem catalog is rendered client-side (React/JS), so I could not scrape the literal list of problem titles via WebFetch - confirmed catalog contents (k-means, PCA, decision trees, SVD, k-NN present) via a public solutions repo and a walkthrough article instead; the platform URL itself was fetched directly and did return real content. eriklindernoren/ML-From-Scratch has had zero commits since October 2019 (dead ~6.75 years) despite being the most-starred and most-often-linked repo in this space - flagged prominently since a naive learner would otherwise assume it's current. Full report written to /Users/macbook/mlsys-lab/docs/seo/landscape/algorithms-scratch.md.

</details>

## LLM internals

*192 tasks in this bank · **Crowded** — several resources let you write code here and get a verdict back*

**Start here if you want to be graded:** [Stanford CS336](https://github.com/stanford-cs336/assignment1-basics), [karpathy/minbpe](https://github.com/karpathy/minbpe), [Build a Large Language Model](https://github.com/rasbt/LLMs-from-scratch).

### [ARENA 3.0 — Chapter 1: Transformer Interpretability](https://github.com/callummcdougall/ARENA_3.0)
`ships tests` · `free` · `course with labs`  
Size: chapter 1 alone: 2 compulsory exercise sets + several optional extensions (SAEs, steering vectors, IOI circuits)  
Last activity: 2026-07-24  

Public materials for a mechanistic-interpretability training program. The first exercise set has you build a GPT-2-architecture transformer from scratch in raw PyTorch (attention, positional encoding, layer norm) mirroring TransformerLens internals, then sample from it, with solutions and test functions to check intermediate tensors.

*Relation to this area:* Overlaps heavily on the attention/positional-encoding/architecture slice; the most actively maintained resource in this list.

### [Build a Large Language Model (From Scratch) — rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
`ships tests` · `freemium` · `book with exercises`  
Size: 7 main chapters + bonus/appendix folders (KV-cache, MLA, LoRA, etc.); ~30 quiz questions per chapter in a separate free PDF  
Last activity: 2026-07-11  

Companion repo to Sebastian Raschka's Manning book; builds embeddings, causal/multi-head attention, LayerNorm, feed-forwards, and full GPT-2-style pretraining in plain PyTorch, chapter by chapter, with end-of-chapter exercises (solutions in Appendix C) and a confirmed dedicated ch04/03_kv-cache folder implementing KV-cache inference.

*Relation to this area:* Overlaps heavily across nearly the whole area, and is the one resource here that explicitly exercises KV-cache shape/layout.

### [Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)
`auto-graded` · `freemium` · `graded platform`  
Size: 6 confirmed relevant problems (self-attention, multi-head attention, masked self-attention, layer norm for sequences, positional encoding, GQA) out of 100+ site-wide  
Last activity: unknown  

LeetCode-style browser platform; I fetched the collection page and problem #391 (Grouped Query Attention) directly and confirmed both are live with in-browser test-case grading. Some content sits behind an unconfirmed-price Premium tier.

*Relation to this area:* Some overlap — covers the attention/norm/positional-encoding family as bite-size auto-graded problems, not tokenization, sampling, or KV-cache.

### [Stanford CS336 — Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
`ships tests` · `free` · `course with labs`  
Size: 1 multi-part assignment, ~10 gradable components (tokenizer, RMSNorm, RoPE, SwiGLU, MHA, transformer block/LM, cross-entropy, AdamW, LR schedule, checkpointing)  
Last activity: 2026-04-07  

Public student repo for Stanford's Language Modeling From Scratch course. I read tests/adapters.py directly: it requires implementing run_rmsnorm, run_rope, run_swiglu, run_scaled_dot_product_attention, run_multihead_self_attention_with_rope and run_transformer_block/run_transformer_lm plus a BPE tokenizer, all checked by pytest starting from NotImplementedError.

*Relation to this area:* Overlaps heavily — near one-to-one match of this bank's LLM-internals syllabus (minus ALiBi, which it doesn't cover).

### [karpathy/minbpe](https://github.com/karpathy/minbpe)
`ships tests` · `free` · `exercise repo`  
Size: 1 focused exercise, 3 tokenizer classes (Basic/Regex/GPT4)  
Last activity: 2024-07-01  

Karpathy's minimal from-scratch BPE implementation with a companion exercise.md that lays out a step-by-step build order to your own GPT-4-style tokenizer, checked against the shipped reference gpt4.py and the repo's own pytest tests.

*Relation to this area:* Overlaps heavily with just the tokenisation/BPE slice of this area; does not touch attention, positional encodings, or norms at all.

### [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)
`read only` · `free` · `reference implementation`  
Size: 1 notebook covering the full original Transformer paper  
Last activity: 2024-04-07  

Line-by-line PyTorch implementation of Attention Is All You Need interleaved with the paper's own text as a runnable notebook. Nothing to fill in and nothing checks your work; it's read-and-run, not exercise-and-grade.

*Relation to this area:* Covers the same mechanism (attention, positional encoding) but as reading material, not something you practise against.

### [picoGPT](https://github.com/jaymody/picoGPT)
`read only` · `free` · `reference implementation`  
Size: ~60-120 lines depending on the file  
Last activity: 2023-04-24  

GPT-2 forward pass in plain NumPy including its own BPE encoder, loads real GPT-2 weights and produces real output; deliberately has no training code, batching, or KV-cache. Nothing to submit; you read and modify it yourself.

*Relation to this area:* Adjacent — same mechanism end-to-end in miniature, but reference-only and dormant for 3+ years.

**What none of these do.** On the core mechanics themselves — implementing MHA/GQA, causal masking, RoPE, RMSNorm, BPE — we are not differentiated; Stanford CS336 and ARENA both already require implementing nearly this exact list under pytest-style checks, and Raschka's book/repo covers the rest including KV-cache. Where we differ is grading depth and framing, not topic coverage: every resource found here checks final-tensor correctness against a reference (numerical match to a tolerance), none of them grade a mechanistic or structural property of the implementation (e.g. correct causal-mask behavior at the boundary token, correct KV-cache memory layout, degenerate-distribution handling in top-p sampling) the way this bank's other areas do for C++/CUDA. ALiBi is close to absent from all of the above (everyone teaches RoPE instead), and weight-init/embedding-tying are never graded in isolation anywhere — both are genuine small gaps. Being one area inside a single deterministic, offline, 14-area/2052-task bank is a structural difference from any of these standalone resources, but that is a packaging argument, not a content-uniqueness one for this specific area.

<details><summary>Survey notes for this area</summary>

Excluded after checking: karpathy/LLM101n (archived on GitHub, was always a syllabus/skeleton, no real exercise content to link to). srush/Transformer-Puzzles was considered (same author as the GPU-Puzzles anchor) but left out of the report body — it teaches transformer mechanics through a RASP-style selector language rather than raw attention/RoPE/tensor code, is a materially different framing than everything else here, and is stale (last push 2023-05, 398 stars); worth a mention to the owner but didn't make the cut under the no-padding rule. Deep-ML's exact Premium price and which problems are gated could not be confirmed (WebFetch renders it as an empty client-rendered shell); the free collection page and the GQA problem itself were both confirmed live and ungated at fetch time. All GitHub star/push-date numbers came from `gh api repos/<owner>/<repo>` (authenticated), not from WebFetch summaries, after `curl` hit anonymous GitHub API rate limits.

</details>

## LLM systems

*200 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [Stanford CS336](https://github.com/stanford-cs336/assignment2-systems), [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles), [MIT 6.5940](https://hanlab.mit.edu/courses/2024-fall-65940).

### [CMU 10-414/714 — Deep Learning Systems (Needle)](https://dlsyscourse.org/assignments/)
`ships tests` · `free` · `course with labs`  
Size: 5 homeworks + final project  
Last activity: 2026 (Fall 2025 due dates visible, currently taught)  

Students build 'Needle,' a PyTorch-like autodiff framework, from CPU/GPU backends through CNNs/RNNs/Transformers, finishing with lectures on training large models and deployment. Real autograder ('mugrade') is CMU-enrollment-only; public repos ship the tests to self-check against.

*Relation to this area:* Adjacent, not overlapping: operates one layer down (build the framework/kernels) from where our tasks operate (reason about scheduling, sharding, and memory given an existing framework); included because learners researching this area reliably land here.

### [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles)
`ships tests` · `free` · `puzzle set`  
Size: 8 puzzles  
Last activity: 2024-01  

Eight Colab puzzles simulating a multi-GPU cluster (not real hardware) where you implement data parallelism, pipeline parallelism, and ZeRO-style sharding against a memory budget; an in-notebook Model.check() asserts the weights were correctly sharded and updated, printing 'Correct!'.

*Relation to this area:* The closest direct analogue in shape (small, gradable, conceptual, simulated hardware) to our distributed-training tasks, but 8 puzzles vs. our 200, and frozen for two and a half years.

### [MIT 6.5940 — TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940)
`ships tests` · `free` · `course with labs`  
Size: 5 labs + final project  
Last activity: 2024-09 (Fall 2024); not offered Fall 2025, next offering unconfirmed  

Song Han's course on efficient deep learning: pruning, quantization, neural architecture search, and a lab deploying Llama-2-7B locally, with lecture coverage of distributed training and gradient/model compression. Labs are Colab notebooks with built-in pass/fail sanity checks; public without enrollment, but not currently a running/live course.

*Relation to this area:* Its LLM-deployment lab overlaps our inference/memory-accounting tasks, but its center of gravity is model compression, not the scheduling/sharding/dataloader mechanics that dominate our 200 tasks — and it is currently paused.

### [Stanford CS336 — Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)
`ships tests` · `free` · `course with labs`  
Size: 1 assignment (of 5), 27 commits  
Last activity: 2026-05  

The systems assignment of Stanford's 'build an LLM from scratch' course: profiling, mixed-precision training, a hand-written FlashAttention-2 Triton kernel, and FSDP-style distributed data parallel training on top of your own Assignment-1 model. Ships real tests and a submission script; needs real multi-GPU compute to run.

*Relation to this area:* Overlaps heavily on mixed precision, profiling, and sharding at much greater depth per topic, but as one multi-week hardware-bound assignment inside a full course, not a standing bank of short items.

### [Efficiently Serving LLMs / Fast & Efficient LLM Inference with vLLM (DeepLearning.AI)](https://www.deeplearning.ai/courses/efficiently-serving-llms/)
`read only` · `freemium` · `course with labs`  
Size: ~9 video lessons / ~7 code notebooks each, across 2 short courses  
Last activity: 2026-06 (vLLM course launch); earlier course dates to 2024  

Two short video courses on LLM serving: KV caching, continuous batching, quantization, LoRA serving (Predibase-taught), and a newer course on the real vLLM stack covering PagedAttention, prefix caching, and quantization with GuideLLM/lm-eval benchmarking. Free tier is video plus ungraded code-along notebooks; a graded assignment exists behind DeepLearning.AI's paid Pro tier (content not verified).

*Relation to this area:* Covers the serving/batching/latency-throughput half of our scope hands-on but as a few hours of guided notebooks, not a bank of graded drills.

### [EleutherAI Cookbook](https://github.com/EleutherAI/cookbook)
`read only` · `free` · `reference implementation`  
Size: 845 stars, ~54 commits  
Last activity: 2026-03  

Runnable calc/ scripts for FLOPs, memory, and parameter-count estimation, plus communication and GEMM benchmarks and a curated reading list, framed by its authors as 'deep learning for dummies' — the practical utilities around real model training.

*Relation to this area:* A calculator toolbox for the same memory/FLOPs math our tasks require you to derive yourself; complements rather than replaces practice.

### [How To Scale Your Model (jax-ml Scaling Book)](https://jax-ml.github.io/scaling-book/)
`read only` · `free` · `book with exercises`  
Size: ~14 chapters  
Last activity: 2026-07  

Google DeepMind's ongoing blog-style textbook on scaling LLMs: roofline analysis, TPU/GPU hardware, FLOPs/memory/communication math, parallelism strategy selection, worked LLaMA-3 case studies. Has embedded 'problems to work for yourself' but no automatic answer-checking.

*Relation to this area:* Closest conceptual match to our roofline/throughput-accounting tasks — same FLOPs-vs-bytes-vs-wall-clock reasoning — delivered as worked exposition with practice questions, not gradable tasks.

### [The Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/)
`read only` · `free` · `interactive explainer`  
Size: 1 long-form book, ~4000 backing experiments, several embedded calculators  
Last activity: 2025 (published, still served live, no dated recent edits found)  

Hugging Face/Nanotron's interactive web book on training LLMs across GPU clusters, grounded in ~4,000 real scaling experiments on up to 512 GPUs: DP/TP/sequence/context parallelism, pipeline schedules, ZeRO 1-3, activation recomputation, mixed precision, with embedded memory calculators and real profiler traces.

*Relation to this area:* Covers the same taxonomy (DP/TP/PP/ZeRO/mixed precision) as reading material and reference numbers with no exercises — explains the why, doesn't test whether you learned it.

### [Transformer Math 101 (EleutherAI blog)](https://blog.eleuther.ai/transformer-math/)
`read only` · `free` · `reading list`  
Size: 1 blog post  
Last activity: 2023-04  

The canonical reference derivation of training compute (C≈6PD), full memory accounting across weights/gradients/optimizer-states/activations under different precisions and ZeRO stages, and achievable per-GPU throughput ranges.

*Relation to this area:* The formulas underlying our memory-accounting and tokens/s tasks — a reference to check a derivation against, not something that checks it for you.

### [awesomeMLSys (GPU MODE)](https://github.com/gpu-mode/awesomeMLSys)
`read only` · `free` · `reading list`  
Size: ~1.1k stars, 17 commits, 8 topic categories  
Last activity: 2026-02  

A curated bibliography of papers/videos/repos for ML-systems onboarding — attention, inference optimization, quantization, long context, distributed training, speculative decoding — explicitly framed as study material, not a curriculum with checkpoints.

*Relation to this area:* A map of what to read around every topic our bank covers; no exercises, zero format overlap.

### [llm.c](https://github.com/karpathy/llm.c)
`read only` · `free` · `reference implementation`  
Size: 1536+ commits, 30.6k stars  
Last activity: 2025-06  

GPT-2/GPT-3 training in raw C/CUDA with no PyTorch dependency: mixed-precision training, gradient accumulation, a tokenized dataloader, and multi-GPU/multi-node training via MPI+NCCL, with unit tests cross-checking against a PyTorch reference.

*Relation to this area:* Covers mixed precision, gradient accumulation, and dataloader mechanics our tasks also probe, at the C/CUDA layer instead of the Python/conceptual layer our bank uses.

### [picotron](https://github.com/huggingface/picotron)
`read only` · `free` · `reference implementation`  
Size: ~180 commits, 2.3k stars  
Last activity: 2025-08  

Minimalist reference implementation of 4D parallelism (data/tensor/pipeline/context) for pretraining LLaMA-style models, deliberately kept to single files under ~300 lines each, built explicitly for education alongside companion video tutorials.

*Relation to this area:* A worked reference for the exact sharding techniques (DP/TP/PP) our tasks probe — good to read after struggling with a task, not a substitute for being tested on it.

**What none of these do.** We are differentiated on format, not on topic coverage: everything found either explains these concepts (books, playbooks, blog math) or lets you attempt them once inside a multi-week course/hardware-bound assignment (CS336, MIT 6.5940) or a small frozen 8-puzzle notebook (LLM-Training-Puzzles, stalled since Jan 2024). Nobody offers a large, stable, offline, deterministically-graded bank of short independent tasks covering batching/scheduling, memory accounting, DP/TP/PP/ZeRO, mixed precision, dataloaders, and roofline reasoning that runs on a laptop with no GPU and no course calendar. On raw topic depth we are not ahead of CS336's systems assignment or the Ultra-Scale Playbook; our edge is purely that ours is a bank you can practise against incrementally with a pass/fail signal, which none of the comparable resources are.

<details><summary>Survey notes for this area</summary>

MIT 6.5940 was not offered Fall 2025 (instructor sabbatical) and its next offering is unconfirmed — flagged as paused in the writeup, not dead but not currently running. karpathy/llm.c and srush/LLM-Training-Puzzles have had no commits in 13 and 30 months respectively (as of 2026-07) despite high star counts — both flagged as stale/frozen but still functional. CMU 10-414/714 (dlsyscourse.org) and MIT 6.5940 are included as adjacent/honest-negative entries (learners researching this area land on them) even though their actual center of gravity is framework-building / model-compression rather than distributed-scheduling-at-scale — this is called out explicitly rather than padding the list. DeepLearning.AI's two short courses mention a "graded assignment" gated behind their paid Pro tier that I could not access or verify the content/rigor of, so I marked free-tier graded:no and cost:freemium rather than guessing. GitHub REST API (api.github.com) returned 403 to the sandboxed WebFetch tool; all star/commit/date metadata came from `gh api` via Bash instead, cross-checked against the fetched repo pages themselves.

</details>

## Applied quantization

*116 tasks in this bank · **Adjacent only** — papers, docs and reference code — nothing to practise against*

### [DeepLearning.AI — Quantization Fundamentals with Hugging Face](https://learn.deeplearning.ai/courses/quantization-fundamentals)
`read only` · `freemium` · `course with labs`  
Size: short course, ~9 video-with-code lessons  
Last activity: unknown  

Video-with-code short course by Hugging Face engineers Younes Belkada and Marc Sun: int/float dtype basics, loading models in different precisions, linear quantization via Hugging Face's quanto library, and bf16 downcasting. Free tier is watch-and-run-the-cell; graded notebooks/quizzes are a paid Plus feature.

*Relation to this area:* overlaps our int8/per-tensor basics but as a guided library walkthrough, not exercises.

### [DeepLearning.AI — Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)
`read only` · `freemium` · `course with labs`  
Size: 13 code examples  
Last activity: unknown  

Follow-on course, same instructors: you build a general linear quantizer in PyTorch from scratch, choosing symmetric vs. asymmetric mode and per-tensor/per-channel/per-group granularity, then implement weight packing down to 2-bit storage. Free tier is code-along plus a quiz; the one graded assignment is paid-tier only.

*Relation to this area:* closest curriculum match to a real chunk of this area (symmetric/asymmetric, per-tensor/channel/group, packing) — we are not differentiated on that specific slice of content, only on volume and having actual per-task auto-grading.

### [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
`read only` · `free` · `reference implementation`  
Size: 2,338 stars  
Last activity: 2024-03-27 (stale, 2+ years)  

Original ICLR 2023 GPTQ paper code: Hessian-based layer-wise post-training quantization of OPT/BLOOM to 2/3/4 bits, with CUDA kernels and perplexity evaluation scripts. No exercises, no starter/reference split — read-only ground truth for the algorithm, largely superseded in practice by maintained forks (e.g. ModelCloud/GPTQModel).

*Relation to this area:* ground truth for the GPTQ mechanics our GPTQ tasks test understanding of.

### [Maxime Labonne — Introduction to Weight Quantization](https://maximelabonne.substack.com/p/introduction-to-weight-quantization-2494701b9c0c)
`read only` · `free` · `interactive explainer`  
Size: 1 article + 1 companion Colab notebook  
Last activity: 2023-07  

Long-form article implementing absmax (symmetric) and zero-point (asymmetric) INT8 quantization from scratch in plain PyTorch, then LLM.int8() mixed-precision outlier handling, comparing GPT-2 perplexity before/after. Free, runnable, no test harness — you read the output and judge it yourself. Predates GPTQ/AWQ/GGUF, which it does not cover.

*Relation to this area:* closest open thing to 'write the dequant math yourself and see it work', but a single walkthrough article, not a problem bank, and stops at plain INT8.

### [Maxime Labonne's LLM Course — quantization notebooks](https://github.com/mlabonne/llm-course)
`read only` · `free` · `course with labs`  
Size: 81k+ GitHub stars; quantization section = 3 Colabs (GPTQ, GGUF/llama.cpp, ExLlamaV2) + a GPTQ/AWQ reading-list module  
Last activity: 2026-02 (repo actively maintained; the quantization Colabs themselves are from the 2023 GPTQ/GGUF wave)  

The de-facto roadmap most self-taught practitioners land on for 'how do I actually quantize a model today'. Runnable Colabs that apply existing libraries (auto-gptq, llama.cpp) to a real model and let you inspect resulting file size/perplexity — not from-scratch algorithm implementation.

*Relation to this area:* curated pointer to applied GPTQ/GGUF/EXL2 tooling, not an exercise bank.

### [OscarSavolainen/Quantization-Tutorials](https://github.com/OscarSavolainen/Quantization-Tutorials)
`read only` · `free` · `reference implementation`  
Size: 31 stars (small)  
Last activity: 2024-05-21 (slowing, single contributor)  

Companion code to a YouTube series: PyTorch eager-mode static/dynamic PTQ, FX-graph-mode PTQ, FX QAT, and cross-layer equalization, all on ResNet. No built-in correctness checking — read the matching folder alongside the video.

*Relation to this area:* closer to classic CNN PTQ/QAT than LLM-era GPTQ/AWQ/GGUF, but a real from-scratch walkthrough of QAT mechanics this area also covers.

### [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
`read only` · `free` · `reference implementation`  
Size: 8,340 stars  
Last activity: within the last day — very active, Hugging Face-backed  

Production library implementing LLM.int8() vector-wise outlier-aware quantization, blockwise 8-bit optimizers, and NF4 4-bit quantization (the QLoRA dtype), with real dequant kernels behind a drop-in nn.Linear. Use-as-a-library; no exercises.

*Relation to this area:* production reference for zero-point/blockwise dequant kernels and NF4, not covered by the paper repos above.

### [ggml-org/llama.cpp — ggml-quants.c (GGUF k-quants)](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
`read only` · `free` · `reference implementation`  
Size: 121,618 stars (whole repo)  
Last activity: within the last day — extremely active  

The actual bit-packing source for Q2_K through Q8_0 (and legacy Q4_0/Q5_0): super-block scale quantization and per-block zero-points, the real byte layout our GGUF k-quant tasks model. C, not Python, production inference code with no learner-facing exercises.

*Relation to this area:* the only place to see the real k-quant layout rather than a paraphrase of it.

### [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
`read only` · `free` · `reference implementation`  
Size: 3,596 stars  
Last activity: 2025-07-17  

Original AWQ paper code (MLSys 2024 Best Paper): activation-aware salient-channel search, INT3/4 weight-only quantization, real CUDA kernels, precomputed model zoo, TinyChat inference engine. Reference code only, no exercises.

*Relation to this area:* ground truth for why AWQ protects channels by activation magnitude rather than weight magnitude.

### [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)
`read only` · `free` · `reference implementation`  
Size: 1,672 stars  
Last activity: 2024-07-12 (about 2 years stale)  

Original ICML 2023 SmoothQuant paper code: migrates quantization difficulty from activations to weights via a per-channel smoothing factor to enable W8A8 with near-fp16 accuracy. OPT/Llama demo notebooks, TensorRT-LLM/ONNX export examples. Reference-only.

*Relation to this area:* ground truth for the SmoothQuant migration-factor mechanics.

### [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
`read only` · `free` · `reference implementation`  
Size: 3,582 stars, 3,133+ commits  
Last activity: within the last day — very active  

One library that runs GPTQ, AWQ, SmoothQuant, AutoRound and rotation-based methods (SpinQuant/QuIP) end to end for vLLM deployment, targeting W8A8/W4A16/FP8/NVFP4/MXFP4. Ships example scripts per recipe, not exercises; GGUF is out of scope. Sibling tools worth knowing: ModelCloud/GPTQModel (active, multi-hardware GPTQ/AWQ export) and the now-archived casper-hansen/AutoAWQ.

*Relation to this area:* this is what 'post-training pipelines as shipped' looks like in production today — the closest real-world match to this task category's framing, but a call-the-library tool, not a learning resource.

**What none of these do.** Nothing found gives a deterministic, machine-independent pass/fail on quantization error, bits-per-weight, or packing correctness the way this bank does. The closest analogue, DeepLearning.AI's "Quantization in Depth," has learners build the same symmetric/asymmetric per-tensor/per-channel/per-group quantizer this area covers, but grading is either absent (free tier) or one paid assignment, and it tops out at 13 examples versus our 116 tasks. Every algorithm-specific repo (GPTQ, AWQ, SmoothQuant, GGUF k-quant source) is read-only reference code with no starter/reference split, and two of the three original paper repos have been quiet 2+ years. On raw curriculum content (what GPTQ/AWQ/SmoothQuant/GGUF do and why), we are not unique — that is already well covered for free; our differentiation is the grading harness, not the topic list.

<details><summary>Survey notes for this area</summary>

deep-ml.com was checked directly (its problems listing) and has zero quantization problems — confirmed absent, not just unsearched. MIT 6.5940 (TinyML)'s Lab 2 "Quantization" and Lab 4 "LLM Compression" exist and are the most on-point university lab material, but they are gated behind MIT Canvas/Colab-with-MIT-credentials; only lecture slides/recordings are public, so the lab itself is not usable by an outside learner — do not re-list it as available in a future pass. Two of the three original algorithm repos (IST-DASLab/gptq, mit-han-lab/smoothquant) have had no commits in 2+ years; mit-han-lab/llm-awq is the freshest of the three (~1 year). casper-hansen/AutoAWQ is archived (confirmed via API) — mentioned only in passing, not as a listed resource. ggml-org/llama.cpp's ggml-quants.c file was too large for full single-fetch inspection; the fetch that succeeded confirmed real k-quant reference code (quantize_row_q2_K_ref etc.) at that path, matching the description given — treated as verified on that basis.

</details>

## Attention and KV cache

*124 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [LeetGPU](https://leetgpu.com/challenges), [Tensara](https://tensara.org/), [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles).

### [LeetGPU — challenge set](https://leetgpu.com/challenges)
`auto-graded` · `freemium` · `graded platform`  
Size: ~90 challenges total; 14 directly on attention/RoPE/KV-cache (rotary-positional-embedding, sliding-window-self-attention, casual-attention, multi-head-attention, grouped-query-attention, linear-attention, decaying-causal-attention, attn-w-linear-bias, softmax-attention, int8-kv-cache-attention, gpt2-block, llama-transformer-block, speculative-decoding-verification, top-p-sampling)  
Last activity: 2026-07-24  

Browser IDE where you implement a kernel (CUDA/Triton/PyTorch/Mojo) against a fixed signature; the site runs it against hidden tests and a timing score, all on CPU-emulated GPU execution for free (real-hardware tier is paid).

*Relation to this area:* Direct competitor for the attention-math-as-a-kernel slice of this area (RoPE, sliding-window, causal, GQA, ALiBi, one int8-KV-cache exercise); has nothing on block tables, paged allocation, prefix caching, or chunked prefill.

### [Stanford CS336 — Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems)
`ships tests` · `free` · `course with labs`  
Size: one multi-part assignment: profiling/benchmarking, FlashAttention-2 forward+backward in Triton, then DDP + optimizer-state sharding  
Last activity: 2026-05-01  

Public student repo of a real current Stanford LLM-systems course; you write the FlashAttention-2 Triton forward kernel and a from-scratch two-pass backward, then benchmark against torch.compile/SDPA, checked with a provided pytest suite and an optional public leaderboard.

*Relation to this area:* Strongest direct overlap with the FlashAttention-tiling/online-softmax subtopic at production seriousness; no coverage of paged attention, prefix caching or eviction.

### [Tensara — scaled-dot-attention problem](https://tensara.org/)
`auto-graded` · `free` · `graded platform`  
Size: 1 of ~90 problems in the repo touches attention  
Last activity: 2026-04-23  

A single 'hard' problem asking for plain softmax(QK^T/√E)V over (B,H,S,E) tensors, scored on real-GPU wall-clock speed; no tiling, no online softmax, no cache.

*Relation to this area:* Barely overlaps — it's the textbook operator as a speed target, not FlashAttention's algorithm and not KV-cache management; worth naming so nobody assumes Tensara covers this area.

### [Triton-Puzzles (gpu-mode / Sasha Rush)](https://github.com/gpu-mode/Triton-Puzzles)
`auto-graded` · `free` · `puzzle set`  
Size: 12 puzzles total; 2 belong to this area (#8 Long Softmax, #9 Simple FlashAttention)  
Last activity: 2026-04-01  

Colab notebook that builds from trivial Triton pointer kernels up to a single-tile flash-attention kernel with online softmax, each puzzle auto-checked against a reference via a Triton interpreter — no GPU needed.

*Relation to this area:* Best free, no-GPU, auto-graded intro to online softmax and flash-attention tiling found; only 2 puzzles, no backward pass or causal-masking depth.

### [dataflowr — Flash-Attention in Triton](https://github.com/dataflowr/gpu_llm_flash-attention)
`ships tests` · `free` · `course with labs`  
Size: 3 homework parts: softmax-matmul kernel, FA forward/backward in PyTorch, FA ported to Triton with benchmarking  
Last activity: 2026-02-09  

A university course module handing you an empty notebook and a PDF spec; you fill in TODOs to build online softmax and tiled attention step by step, checked against an included tests/ folder.

*Relation to this area:* Smaller, gentler cousin of the CS336 assignment on the same subtopic (tiling/online softmax), without the distributed-training overhead.

### [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
`read only` · `free` · `reference implementation`  
Size: 24,539 stars; production FlashAttention-2/3 library (causal, MQA/GQA, varlen, paged-KV support)  
Last activity: 2026-07-25  

The canonical production implementation the whole field benchmarks against; CUDA/CUTLASS kernels to read, no tests or puzzles attached.

*Relation to this area:* Defines 'correct' for the tiling/online-softmax subtopic in production; read after doing graded exercises elsewhere, not instead of.

### [SGLang (RadixAttention / prefix caching)](https://github.com/sgl-project/sglang)
`read only` · `free` · `reference implementation`  
Size: 30,751 stars; full serving engine, RadixAttention is one subsystem  
Last activity: 2026-07-26  

Production serving engine whose RadixAttention subsystem implements radix-tree-based prefix/prompt cache reuse across requests.

*Relation to this area:* Production reference for prefix/prompt caching; no isolated tutorial or grader exists for RadixAttention anywhere.

### [hkproj/triton-flash-attention (Umar Jamil)](https://github.com/hkproj/triton-flash-attention)
`read only` · `free` · `exercise repo`  
Size: 257 stars; one complete FlashAttention-2 Triton kernel plus a long-form video walkthrough  
Last activity: 2025-01-02  

A 'code along with the video' path into flash-attention tiling; names two follow-on exercises (autotune the backward pass, skip masked blocks in causal attention) but ships no checker for them.

*Relation to this area:* Well-known teaching path into flash-attention tiling; no automated grading and no cache-management coverage.

### [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)
`read only` · `free` · `reference implementation`  
Size: 421 stars; 2-bit asymmetric per-channel/per-token KV-cache quantization reference  
Last activity: 2025-11-20  

Reference implementation of the KIVI paper's tuning-free 2-bit KV-cache quantization scheme, actively maintained relative to the other reference repos here.

*Relation to this area:* Most concrete code found for the quantised-KV subtopic specifically; still read-only, no grader.

### [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
`read only` · `free` · `reference implementation`  
Size: 7,249 stars; Llama-2/MPT/Falcon/Pythia attention-sink + sliding-window implementations  
Last activity: 2024-07-11  

Original code for the StreamingLLM/attention-sinks paper; no commits in about two years, so treat it as a historical reference rather than a maintained tool against current library versions.

*Relation to this area:* Origin implementation for the sink+sliding-window subtopic; dormant.

### [tomaarsen/attention_sinks](https://github.com/tomaarsen/attention_sinks)
`read only` · `free` · `reference implementation`  
Size: 735 stars; drop-in transformers-API wrapper adding attention sinks  
Last activity: 2024-04-10  

Friendlier-to-install alternative to streaming-llm implementing the same sink/window idea as a transformers drop-in.

*Relation to this area:* Same subtopic as streaming-llm, equally dormant (over 2 years with no commits).

### [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
`read only` · `free` · `reference implementation`  
Size: 1,173 stars; ~100 lines of raw CUDA, forward pass only  
Last activity: 2024-12-30  

The most-cited minimal raw-CUDA (not Triton) implementation of tiled attention with online softmax — short enough to read in one sitting.

*Relation to this area:* Good short reference for the tiling/online-softmax algorithm without a framework in the way; code to study, not an exercise.

### [tspeterkim/paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)
`read only` · `free` · `reference implementation`  
Size: 149 stars; minimal block-table KV-cache manager on top of a Llama-3 forward pass  
Last activity: 2024-08-26  

A small, readable cache manager that reuses FlashAttention's PagedAttention kernel to show how block-table allocation and lookup actually work end to end.

*Relation to this area:* Only small readable reference found for real block-table mechanics, versus reading the full vLLM codebase; dormant ~2 years but still functional as a reference.

### [vLLM — PagedAttention design doc and kernel](https://docs.vllm.ai/en/latest/design/paged_attention/)
`read only` · `free` · `reference implementation`  
Size: repo has 87,188 stars; doc is a single walkthrough page, kernel is csrc/attention/attention_kernels.cu  
Last activity: 2026-07-26  

vLLM's own walkthrough of its PagedAttention CUDA kernel — block-structured KV cache, block tables, per-thread-group key access, softmax and write-out — explicitly labelled a historical explainer that defers to current source.

*Relation to this area:* Primary reference for block-table/paged-KV-cache layout, otherwise unpracticed anywhere in this search; documentation plus source, not something you're tested against.

**What none of these do.** For the "compute attention" half of this area (FlashAttention tiling, online softmax, RoPE, sliding-window/causal masking, GQA) real auto-graded or self-checked practice already exists — LeetGPU, Triton-Puzzles, and Stanford CS336's public assignment all cover it, so our bank is not uniquely positioned there beyond breadth and volume. For the "manage the cache" half — KV-cache block-table layout, paged allocation, prefix/prompt caching, chunked prefill, cache eviction, and quantised-KV numerics — nothing found offers a graded or self-checked exercise at all; the only material is production source (vLLM, SGLang) or frozen research repos (StreamingLLM, attention_sinks, KIVI) that a learner reads rather than solves against a pass/fail gate. RoPE *scaling* specifically (NTK/YaRN/linear long-context extrapolation, as opposed to plain RoPE application) also has zero exercise-style coverage anywhere — only blog posts and papers. That cache-management and RoPE-scaling territory is where this bank's 124 tasks are genuinely filling an empty space rather than competing in a crowded one.

<details><summary>Survey notes for this area</summary>

Paradigm's "Attention Kernel Challenge" (paradigm.xyz/attention-kernel-challenge) was fetched and confirmed real but is a closed, time-boxed hackathon (submissions already ended) — deliberately left out of the resources list as not a standing practice resource, but noted here so it isn't mistaken for a dead/fabricated link if the owner runs into it. leetgpu.com's pricing page failed to render content for this session (client-side auth gate), so the "freemium" cost label for LeetGPU rests on the platform's public Show-HN description ("write and execute CUDA on the web, no GPU required, for free") plus the orchestrator's own calibration note about a paid real-GPU tier, not on a fetched pricing page. Several reference repos (streaming-llm, attention_sinks, tspeterkim/paged-attention-minimal) have had no commits in 2+ years — flagged individually in the report rather than dropped, per instructions, since they're still the best available reference for their subtopic. srush/Transformer-Puzzles was checked and excluded: it's about RASP-style "thinking like a transformer" logic puzzles using attention as a primitive for interpretability, not the systems-level tiling/KV-cache content of this area.

</details>

## Compilation and export

*115 tasks in this bank · **Adjacent only** — papers, docs and reference code — nothing to practise against*

**Start here if you want to be graded:** [NVIDIA DLI: Optimization and Deployment of TensorFlow Models with TensorRT](https://www.coursera.org/projects/tensorflow-tensorrt), [MLC: Machine Learning Compilation](https://mlc.ai/courses.html).

### [MLC: Machine Learning Compilation (mlc.ai)](https://mlc.ai/courses.html)
`ships tests` · `free` · `course with labs`  
Size: ~8 chapters/notebooks  
Last activity: 2022-07 (course content not visibly revised since; mlc-ai org itself active but has shifted to shipping mlc-llm)  

CMU-taught open course on ML compilation built around Apache TVM/TensorIR; each chapter pairs a lecture with a notebook where you write TVMScript and compare output against a NumPy reference.

*Relation to this area:* Adjacent, not overlapping: teaches compiler internals via TVM, a different stack from torch.compile/Dynamo/ONNX/TensorRT, and useful only as background on why graph compilers behave as they do.

### [NVIDIA DLI: Optimization and Deployment of TensorFlow Models with TensorRT (+ Coursera guided-project version)](https://www.coursera.org/projects/tensorflow-tensorrt)
`auto-graded` · `paid` · `course with labs`  
Size: one course (DLI, ~8h, ~$90) / one guided project (Coursera, 1.5h)  
Last activity: unknown, no visible revision date; underlying TF-TRT/TensorFlow SavedModel stack reads as dated  

Hands-on cloud-GPU lab: convert a TensorFlow SavedModel to TF-TRT at FP32/FP16/INT8 on InceptionV3, benchmark throughput, observe the accuracy/speed trade-off. The DLI 8-hour version ends in a graded assessment for a certificate; the Coursera guided-project twin is the same content, completion-graded only.

*Relation to this area:* Overlaps with TensorRT-conversion and numerical-drift-after-conversion, but via TensorFlow/TF-TRT rather than the PyTorch-centric stack this syllabus targets.

### [NVIDIA CUDA Graphs documentation & PyTorch integration guide](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/introduction.html)
`read only` · `free` · `reading list`  
Size: one multi-page official guide plus the original PyTorch blog post  
Last activity: 2025 (copyright-dated, current)  

Official NVIDIA reference documentation on torch.cuda.CUDAGraph/torch.cuda.graph()/make_graphed_callables(): capture semantics, static-input constraints, common correctness pitfalls. Prose and snippets, no lab.

*Relation to this area:* Best available explanation of CUDA-graph capture correctness, purely expository.

### [ONNX Backend Test suite (operator/opset conformance tests)](https://onnx.ai/onnx/repo-docs/OnnxBackendTest.html)
`read only` · `free` · `reference implementation`  
Size: hundreds of per-operator Node/Model test files across opset versions  
Last activity: unknown (part of the actively maintained onnx/onnx repo)  

The Node/Model test suite ONNX itself uses to certify that a runtime correctly implements each operator across opset versions, one Python/NumPy reference file per operator. Built for backend implementers, not learners.

*Relation to this area:* The most concrete existing artifact for 'operator coverage / opset issues', but a compliance suite for framework authors, only usable as a study aid by repurposing it yourself.

### [ONNX Tutorials](https://github.com/onnx/tutorials)
`read only` · `free` · `exercise repo`  
Size: 3.7k stars  
Last activity: 2026-06  

The official ONNX org's notebook collection covering exporting models from PyTorch/TensorFlow/scikit-learn and running them with various runtimes. Run-and-read notebooks with no correctness checker.

*Relation to this area:* Direct overlap with the export half of the syllabus but shallow — shows how to export, not how to diagnose opset mismatches or numerical drift.

### [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
`read only` · `free` · `course with labs`  
Size: 3.2k stars, 1k forks, 3000+ commits  
Last activity: 2026 (main branch tracks OpenVINO 2026.2, current release)  

Intel's official Jupyter notebook catalog for OpenVINO: model conversion to OpenVINO IR, quantization, and dozens of 'run this model' demos launchable in Colab/Binder.

*Relation to this area:* Covers the OpenVINO-conversion side of the syllabus but is a demo catalog to run, not a problem set with a right answer.

### [PyTorch official tutorials: torch.compile, torch.export, troubleshooting & Dynamo deep-dive](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
`read only` · `free` · `interactive explainer`  
Size: ~4 long-form tutorial/guide pages plus API docs  
Last activity: 2026 (docs main/current release branch, continuously updated)  

Official PyTorch team tutorials with runnable code cells measuring torch.compile speedup and demonstrating graph breaks, guards, recompilation, and torch.export dynamic shapes/control-flow via torch.cond. The canonical written explanation of this area's core mechanics.

*Relation to this area:* Direct curriculum overlap on graph breaks, guards/recompilation and dynamic shapes, but it is documentation to read and run, not a bank of problems with a checked answer.

### [depyf](https://github.com/thuml/depyf)
`read only` · `free` · `reference implementation`  
Size: 815 stars  
Last activity: 2025-10  

A tool (with a JMLR paper behind it) that decompiles the bytecode torch.compile/Dynamo generates back into readable Python, so you can see exactly how and where your function was split at each graph break.

*Relation to this area:* A tool a learner would use while doing our graph-break/recompilation tasks, not a teaching resource on its own.

**What none of these do.** None of the resources found grade a learner on this area's actual failure modes (graph-break counts, recompilation storms from guard failures, numerical drift after an ONNX/TensorRT/OpenVINO conversion, opset coverage gaps). Everything real is either official documentation/tutorials to read, a debugging tool (depyf), a conformance-test suite built for backend implementers rather than learners (ONNX Backend Test), or a paid hands-on lab that is completion-graded at best and built on an aging TensorFlow/TF-TRT stack rather than the PyTorch/torch.compile-centric syllabus here. Our bank's differentiation in this area is real: it appears to be the only place that turns "does your fix reduce graph breaks / stop recompilation / keep ONNX output within tolerance" into a deterministic auto-graded number.

<details><summary>Survey notes for this area</summary>

The NVIDIA DLI TensorRT course and its Coursera guided-project twin are the same underlying content (TF-TRT on InceptionV3) sold two ways; listed both since cost/format differ but flagged the overlap. MLC (mlc.ai) is TVM-centric, not torch.compile/ONNX/TensorRT-centric — included as adjacent background on compiler internals, not a direct competitor; its course content is dated July 2022 and appears stalled (the mlc-ai org has since focused on shipping mlc-llm rather than updating the course), while the org/GitHub itself is active. The ONNX Backend Test suite is a real, current, actively-maintained resource but is built for people implementing an ONNX runtime backend, not for learners studying export from PyTorch — repurposing it as a study aid is possible but not what it's for. Could not find a single "X puzzles"/"leet<X>" style graded platform specific to torch.compile, Dynamo, or model export/conversion despite searching several phrasings.

</details>

## Batching and serving

*128 tasks in this bank · **Some overlap** — one or two real practice resources, covering part of the area*

**Start here if you want to be graded:** [llm-inference-engine](https://github.com/achi9629/llm-inference-engine).

### [llm-inference-engine (achi9629)](https://github.com/achi9629/llm-inference-engine)
`ships tests` · `free` · `exercise repo`  
Size: 1 star, 122 pytest tests  
Last activity: 2026-05  

A solo project that builds an inference engine in explicit incremental stages: plain transformer forward pass, KV cache, static batching, continuous batching, paged KV cache, then an async FastAPI serving layer, with 122 pytest tests and isolated before/after benchmarks at each stage. Structurally the closest GitHub match to 'implement the mechanic, then check yourself' for this area, but a very young, one-star, single-author repo, not an established or vetted resource.

*Relation to this area:* same progression as the batching/continuous-batching/paged-cache slice of this area, executed as a personal from-scratch build with real tests rather than a maintained curriculum

### [AIPerf](https://github.com/ai-dynamo/aiperf)
`read only` · `free` · `benchmark or leaderboard`  
Size: 469 stars  
Last activity: 2026-07  

NVIDIA's current LLM-serving load-testing CLI (successor to the now-deprecated genai-perf). Points at any OpenAI-compatible/TGI endpoint you already have running and reports TTFT, inter-token latency, throughput and per-user token rate under configurable concurrency and workload shapes. Its predecessor ray-project/llmperf was archived Dec 2025 after its last commit in Dec 2024.

*Relation to this area:* directly relevant to this area's TTFT/TPOT/throughput tasks, but as a load tester for a server you've already built, not a checker for whether your own scheduling code is correct

### [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
`read only` · `free` · `reading list`  
Size: 5.4k stars, 100+ tracked papers/repos  
Last activity: 2026-07  

A curated paper+code index (same maintainer family as LeetCUDA) with dedicated sections for continuous/in-flight batching, prefill/decode disaggregation (DistServe, Mooncake), scheduling papers, and serving frameworks (vLLM, SGLang, TensorRT-LLM, LMDeploy). Nearly every entry links both a paper and a real implementation. Good for going deep on one subtopic, not a starting point for a beginner.

*Relation to this area:* reference bibliography for nearly every subtopic in this area — disaggregation, scheduling, speculative decoding — one level more academic than this bank's tasks

### [Efficiently Serving LLMs (DeepLearning.AI / Predibase)](https://www.deeplearning.ai/courses/efficiently-serving-llms/)
`read only` · `freemium` · `course with labs`  
Size: 9 video lessons / 7 notebooks  
Last activity: unknown  

2h40m video course by Predibase's CTO: KV caching, batching, continuous batching, quantization, LoRA and multi-LoRA serving, ending in a look at Predibase's LoRAX server. Seven run-along notebooks (unofficial mirror at github.com/ksm26/Efficiently-Serving-LLMs). Free tier is watch-and-run-the-cell; one graded assignment exists but is gated behind a paid DeepLearning.AI/Coursera tier.

*Relation to this area:* covers the conceptual ground of continuous batching and multi-adapter serving that overlaps part of this area, but teaches by demonstration rather than by scored implementation

### [Outlines](https://github.com/dottxt-ai/outlines)
`read only` · `free` · `reference implementation`  
Size: 15.3k stars  
Last activity: 2026-07  

The most widely used structured/constrained-generation library: turns a JSON schema, regex or context-free grammar into a token-level mask via a finite-state-machine index over the vocabulary, so a model can only emit valid tokens. It is the library this area's structured-generation tasks model the mechanics of; you read the source or depend on it, there is nothing to be graded on.

*Relation to this area:* covers exactly the structured/constrained-generation slice of this area, as a production reference implementation rather than an exercise set

### [Vidur](https://github.com/microsoft/vidur)
`read only` · `free` · `reference implementation`  
Size: 646 stars  
Last activity: 2025-07  

Microsoft Research's LLM-inference-system simulator (MLSys 2024): configure a model, hardware, parallelism strategy and scheduling policy (including chunked prefill and speculative decoding), run a workload trace, and get TTFT/TPOT/batch-size numbers back without a GPU. A genuine systems-research tool for exploring scheduling tradeoffs; no notion of a correct answer to check against.

*Relation to this area:* same subject as this area's SLO-aware scheduling and disaggregation tasks, but for research-level exploration of policy tradeoffs, not for practising implementation

### [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
`read only` · `free` · `reference implementation`  
Size: 14.6k GitHub stars, ~1,200 LOC  
Last activity: 2026-04  

A from-scratch vLLM reimplementation in ~1,200 lines of readable Python, with a real scheduler.py and block_manager.py implementing continuous batching and paged KV-cache management (not just the prefix-caching the README headline mentions). Claims throughput comparable to real vLLM; you read the code and run bench.py, there is no test suite or grading.

*Relation to this area:* overlaps heavily with the continuous-batching and paged-KV-cache mechanics this area tests — the closest thing on GitHub to seeing the whole mechanism in one file

### [tiny-vllm](https://github.com/jmaczan/tiny-vllm)
`read only` · `free` · `exercise repo`  
Size: 952 stars  
Last activity: 2026-07  

A guided 'build vLLM yourself' repo in C++/CUDA: starts from bf16 arithmetic and safetensors loading, works through hand-written kernels (RMSNorm, RoPE, attention, softmax), then covers prefill vs decode, static batching, continuous batching, online softmax and PagedAttention. Implements the batching/scheduling mechanics for real, but through low-level CUDA kernels rather than a Python request-path layer. No automated grading; you implement against the author's reference and compare.

*Relation to this area:* adjacent — covers continuous batching and prefill/decode splitting as one stop in a much lower-level CUDA-kernel curriculum, complementing rather than substituting for this area's Python-level tasks

**What none of these do.** Nothing found auto-grades a learner's own scheduler, admission-control policy, or batching loop against a deterministic gate — queue depth under load, P99 TTFT vs an SLO, correct KV-cache-budget accounting, speculative-decoding acceptance-rate math, or a correct prefill/decode split under a disaggregation policy. Best available options are: read a finished reference system (nano-vllm, tiny-vllm), watch a video course that demonstrates the effect in someone else's notebook (DeepLearning.AI), simulate scheduling policy tradeoffs for research (Vidur), or load-test a black-box server you already finished building (AIPerf). This bank's 128 auto-graded Python tasks are the only thing found that turn "did you implement continuous batching / SLO-aware scheduling / speculative decoding correctly" into a scored, measured gate rather than a read-and-vibe-check exercise. Two of this area's named subtopics, load shedding and autoscaling arithmetic, had no dedicated practice material anywhere in this search — they exist only as prose in blog posts and papers.

<details><summary>Survey notes for this area</summary>

ray-project/llmperf (the previous standard for TTFT/TPOT/throughput load testing) was archived December 2025 after its last real commit in December 2024; NVIDIA's own genai-perf is likewise deprecated in favor of AIPerf — both are noted so a reader doesn't land on a dead tool. achi9629/llm-inference-engine is the single closest structural match to this area's batching progression (KV cache -> static batch -> continuous batch -> paged cache, with pytest checks at each stage) but is a brand-new one-star personal repo, not an established resource — flagged as a good find, not a vetted one. jmaczan/tiny-vllm and Outlines both genuinely touch this area's subject matter (continuous batching/prefill-decode; structured generation) but their center of gravity is elsewhere: tiny-vllm is a CUDA-kernel curriculum that happens to pass through batching, Outlines is a production library, not a course. An interactive continuous-batching/PagedAttention visualization at engineersofai.com/playground/continuous-batching turned up in search but returned HTTP 403 on fetch (anti-bot), so it is deliberately omitted rather than reported unverified. deep-ml.com was checked and has no serving/batching problems — it is purely ML-algorithm puzzles, not relevant to this area despite being a well-known "LeetCode for ML" brand.

</details>

## Memory and offload

*112 tasks in this bank · **Adjacent only** — papers, docs and reference code — nothing to practise against*

### [Colossal-AI (Gemini heterogeneous memory manager)](https://github.com/hpcaitech/ColossalAI)
`read only` · `free` · `reference implementation`  
Size: 41,426 stars  
Last activity: 2026-07-13 (pushed_at) — active  

Large distributed-training system whose Gemini memory manager (built on earlier PatrickStar work) tracks tensor liveness and dynamically places model states across GPU/CPU/NVMe under a runtime memory budget.

*Relation to this area:* A second, differently-architected reference for CPU/NVMe offload alongside DeepSpeed; read-only, no grading.

### [DeepSpeed — ZeRO / ZeRO-Offload tutorials](https://www.deepspeed.ai/tutorials/zero-offload/)
`read only` · `free` · `reference implementation`  
Size: 42,806 stars on the library; 2 tutorial pages plus real example configs  
Last activity: tutorial footer 2026-07-24; repo pushed 2026-07-26 — very active  

Official walkthrough for ZeRO stages 1-3 (partition optimizer state/gradients/params) and ZeRO-Offload/-Infinity (push optimizer state and compute to CPU, or CPU+NVMe). Config-JSON only, no code changes — you edit settings and watch memory/throughput move.

*Relation to this area:* Canonical reference for the ZeRO-stages and CPU/NVMe-offload subtopics; nothing to be graded against, purely read-and-run.

### [EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
`read only` · `free` · `reading list`  
Size: unknown (single long-form post)  
Last activity: published 2023-04-18, static since  

Memory-planning formulas for optimizer-state memory (Adam/8-bit/SGD), activation memory with/without recomputation, gradient memory, and how ZeRO sharding changes each term. Explicitly does not cover KV cache.

*Relation to this area:* Most-cited training-side memory-arithmetic reference; has a real gap (no KV cache) worth flagging.

### [FlexGen / FlexLLMGen](https://github.com/FMInference/FlexLLMGen)
`read only` · `free` · `reference implementation`  
Size: 9,363 stars  
Last activity: pushed 2024-10-28, archived 2024-12-01 — dead, read-only  

Stanford/Berkeley/CMU system for high-throughput batch LLM generation on one consumer GPU via a linear-program schedule that jointly places weights, activations, and KV cache across GPU/CPU/disk, plus 4-bit weight/cache compression. Project renamed from FlexGen; GitHub repo is archived.

*Relation to this area:* Clearest single reference combining weight streaming with KV-cache offload under one memory budget, but archived and unmaintained since late 2024 — cite the idea, don't expect a clean install.

### [Hugging Face Accelerate — Big Model Inference guide](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
`read only` · `free` · `reference implementation`  
Size: unknown (single guide in an actively maintained library)  
Last activity: living docs page, continuously updated with huggingface/accelerate  

Hands-on guide to init_empty_weights + load_checkpoint_and_dispatch(device_map="auto"), which places each layer on the fastest device with room and streams weights layer-by-layer to CPU/disk when nothing fits.

*Relation to this area:* Standard runnable reference for weight streaming / offloaded inference in the HF ecosystem; no exercise structure or checker.

### [Hugging Face Transformers — KV cache strategies doc](https://huggingface.co/docs/transformers/en/kv_cache)
`read only` · `free` · `reference implementation`  
Size: unknown (single guide in an actively maintained library)  
Last activity: living docs page, continuously updated with huggingface/transformers  

Runnable comparison of DynamicCache/StaticCache/QuantizedCache (hqq/quanto backends, int2-int8) and cache offloading (cache_implementation="offloaded", keeps only current layer's KV on GPU), including a worked OOM-retry example.

*Relation to this area:* Most directly relevant run-it-yourself resource for KV-cache budgeting and quantised KV; documentation with example code, not scored.

### [NVIDIA Blog — How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
`read only` · `free` · `interactive explainer`  
Size: unknown (single post, links a full runnable async.cu example on GitHub)  
Last activity: published 2012-12-13 — old but API it documents (cudaMemcpyAsync, cudaMallocHost, non-default streams) unchanged  

Lays out the three conditions for overlapping a kernel with a transfer (device support, non-default streams, pinned host memory) with timed measurements across GPU generations and a complete runnable sample.

*Relation to this area:* Standard reference for pinned-memory/transfer-overlap; canonical despite age, no grading.

### [PyTorch Blog — Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
`read only` · `free` · `interactive explainer`  
Size: unknown (single post with runnable snippets)  
Last activity: published 2025-03-05  

Covers torch.utils.checkpoint, torch.compile's min-cut partitioner, Selective Activation Checkpointing (policy-driven save-vs-recompute), and the Memory Budget API that auto-tunes recompute fraction.

*Relation to this area:* Most current reading for activation-checkpointing/recompute, straight from the PyTorch team; no exercise wrapper.

### [PyTorch Blog — Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)
`read only` · `free` · `interactive explainer`  
Size: unknown (single post, 2 runnable appendix scripts)  
Last activity: published 2023-12-14, updated 2024-11-14  

Narrated debugging case study: a ResNet50 loop missing optimizer.zero_grad(), diagnosed live with the Memory Snapshot/Profiler tools, with before/after code and full runnable appendices.

*Relation to this area:* Best worked OOM-bug-in-the-visualizer walkthrough found; complements the reference doc with a real bug and fix.

### [PyTorch — Understanding CUDA Memory Usage (+ memory_viz)](https://docs.pytorch.org/docs/main/torch_cuda_memory.html)
`read only` · `free` · `interactive explainer`  
Size: unknown (single docs page plus a standalone interactive tool)  
Last activity: living page, part of actively-developed pytorch/pytorch docs  

Official guide to torch.cuda.memory._record_memory_history()/_snapshot(), paired with a genuinely interactive local browser tool (pytorch.org/memory_viz) showing an Active Memory Timeline and Allocator State History with stack traces.

*Relation to this area:* Primary tool for allocator-fragmentation and OOM-forensics subtopics; hands-on but no pass/fail.

### [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
`read only` · `free` · `reference implementation`  
Size: 8,340 stars  
Last activity: 2026-07-25 (pushed_at) — very active, nightly tests  

Library implementing blockwise 8-bit (and 4-bit/QLoRA) optimizer states, keeping Adam momentum/variance in 8 bits instead of 32. Install, swap in bnb.optim.Adam8bit, observe reduced optimizer memory.

*Relation to this area:* Direct reference implementation for the quantised-optimiser-states subtopic; no exercises or tests to pass.

### [csc-training/CUDA — exercises/unified-memory-streams](https://github.com/csc-training/CUDA/tree/master/exercises/unified-memory-streams)
`read only` · `free` · `exercise repo`  
Size: 130 stars on parent repo; 1 exercise among a handful  
Last activity: 2017-05-19 (pushed_at) — dead ~9 years, no commits since  

A CSC training exercise: allocate CUDA managed (unified) memory and split addition-kernel launches across multiple streams, filling TODOs in streams.cu against a provided solution/ folder to diff against. No automated checker.

*Relation to this area:* Only hands-on exercise found for unified/managed memory and stream-based memory separation; narrow (one file) and long dormant.

### [kipp.ly — Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
`read only` · `free` · `reading list`  
Size: unknown (single long-form post)  
Last activity: published 2022-03-30, static since  

KV-cache byte-size formulas per token, a worked example of tokens-of-KV-cache that fit on an A100 for a 52B model, and when recompute beats caching.

*Relation to this area:* Standard reference for KV-cache-budgeting math; pure reading, still the most-cited source for these formulas.

### [olcf/cuda-training-series](https://github.com/olcf/cuda-training-series)
`read only` · `free` · `course with labs`  
Size: 1,022 stars; ~9 homework modules, 1 (hw6, Managed Memory) in this area  
Last activity: 2024-08-19 (pushed_at) — ~2 years stalled  

Public homework companion to Oak Ridge National Lab's official CUDA Training Series webinars; hw6 covers cudaMallocManaged and cudaMemPrefetchAsync with self-checked reference solutions, no autograder.

*Relation to this area:* Institutionally credible unified-memory module, but one slice of a general CUDA course, and stalled for two years.

**What none of these do.** Nothing found lets a learner implement a ZeRO stage, an offload scheduler, a KV-cache eviction/quantization scheme, or a checkpointing policy and receive a pass/fail or numeric-error verdict. Every real resource is either a production library you configure and eyeball memory numbers from (DeepSpeed, Colossal-AI, bitsandbytes, Accelerate, FlexGen), an interactive debugging tool with no "correct answer" to match (PyTorch's memory-snapshot visualizer), or a reading-only blog post with formulas and no code to run against. The only two things resembling graded exercises (csc-training, olcf) cover a single narrow slice — unified memory and pinned-memory stream separation — via a solution folder to manually diff against, not an automated grader, and both are 2-9 years stale. Our 112 auto-graded tasks with deterministic byte/transaction/error verdicts across all ten subtopics are not duplicating anything found here; this is a genuine, not manufactured, gap.

<details><summary>Survey notes for this area</summary>

FlexGen was found under its old name/URL (github.com/FMInference/FlexGen) but the GitHub API confirms the repo was renamed to FlexLLMGen and archived 2024-12-01 (read-only, last push 2024-10-28) — link it as FlexLLMGen. csc-training/CUDA has not been pushed to since 2017-05-19 (9 years) despite GitHub showing a recent updated_at (star/traffic metadata only, not code). olcf/cuda-training-series last pushed 2024-08-19 (~2 years stale). EleutherAI's Transformer Math 101 explicitly states it does not cover KV cache/inference memory and points readers to the kipp.ly post instead — I list both to cover the full memory-math reading, not as redundant padding. DeepSpeed's own tutorial page footer showed a very recent date (2026-07-24), confirming it as the freshest resource in this whole set. No graded-platform or puzzle-set of any kind was found for this area after searching GitHub topics, awesome-lists, "X puzzles/exercises" naming patterns, and university course pages (CMU 10-714/Needle, CS149/CS231n) — none had public memory-and-offload-specific graded assignments.

</details>

## Sparsity, pruning, distillation

*125 tasks in this bank · **Adjacent only** — papers, docs and reference code — nothing to practise against*

**Start here if you want to be graded:** [MIT 6.5940 / EfficientML.ai](https://hanlab.mit.edu/course).

### [MIT 6.5940 / EfficientML.ai — TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/course)
`ships tests` · `free` · `course with labs`  
Size: 5 labs total (pruning lab + part of quantization lab overlap this area); offered again Fall 2026  
Last activity: ongoing (Fall 2026 offering listed)  

Song Han's MIT graduate course on efficient AI computing covering pruning, quantization, distillation, NAS, and on-device LLM deployment, each backed by a hands-on Colab lab; third-party mirrors of student solutions confirm the official labs ship in-notebook tests a learner runs to self-check.

*Relation to this area:* The only resource found where a learner's own pruning/sparsity code is checked against something resembling a pass/fail bar rather than just read and compared by eye; caveat: official starter notebooks/autograder access outside MIT's own infrastructure is unconfirmed, and public mirrors are solved solutions.

### [AviSoori1x/makeMoE](https://github.com/AviSoori1x/makeMoE)
`read only` · `free` · `reference implementation`  
Size: 811 stars  
Last activity: 2024-10  

From-scratch, single-file (plus notebooks) sparse Mixture-of-Experts language model in the style of Karpathy's makemore/nanoGPT: top-k and noisy top-k gating, plus a follow-on notebook adding expert-capacity limits.

*Relation to this area:* Best available read-and-reimplement reference for MoE routing sparsity at toy scale; a notebook you run top to bottom, not graded problems.

### [FLHonker/Awesome-Knowledge-Distillation](https://github.com/FLHonker/Awesome-Knowledge-Distillation)
`read only` · `free` · `reading list`  
Size: 2,679 stars; 658 papers  
Last activity: 2023-05  

Curated bibliography of knowledge distillation papers (2014-2021) organized by KD mechanism (logits, feature, KD+GAN, data-free) and application domain, with links to original code.

*Relation to this area:* Covers the KD-loss variants (logit/feature/relational matching) our distillation tasks are based on; a bibliography, not an exercise set.

### [Hugging Face Transformers docs — Knowledge Distillation (image classification)](https://huggingface.co/docs/transformers/tasks/knowledge_distillation_for_image_classification)
`read only` · `free` · `reference implementation`  
Size: one tutorial page  
Last activity: current (live official docs)  

Distills a fine-tuned ViT teacher into a randomly-initialized MobileNetV2 student on the beans dataset via a custom Trainer subclass with KL-divergence soft-target loss plus true-label loss; reports 72% vs 63% test accuracy.

*Relation to this area:* Clean official runnable reference for the logit-distillation-loss mechanic (temperature, lambda-weighted mix) our KD tasks cover; reproducible but no grader, you eyeball the reported number.

### [Microsoft NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)
`read only` · `free` · `reference implementation`  
Size: 14,367 stars — largest project in this list  
Last activity: 2024-09 (archived; last real release 2022-05)  

General AutoML toolkit whose compression module implemented many pruners (L1/L2 Norm, FPGM, Taylor-FO, Movement, AGP, AutoCompress), quantizers, and a basic distillation component, each with quick-start tutorials. Archived Sep 2024, read-only.

*Relation to this area:* Was the closest thing to a unified multi-algorithm pruning/distillation library with runnable tutorials; dead since 2022/2024, worth knowing but not to build a plan around.

### [PyTorch tutorial — Accelerating BERT with 2:4 sparsity (torchao)](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html)
`read only` · `free` · `reference implementation`  
Size: one tutorial; backed by pytorch/ao (2,917 stars)  
Last activity: 2026-07 (torchao pushed 2026-07-26, very active)  

End-to-end walkthrough: train BERT on SQuAD dense, apply magnitude-based 2:4 pruning via torch.ao.pruning, fine-tune to recover F1, then use SparseSemiStructuredTensor for real inference speedup (~1.3x, up to 2x with torch.compile).

*Relation to this area:* Closest public match to our 2:4 semi-structured sparsity material with real measured speedup; single prescribed path, no bank of separate problems, nothing grades a learner's own N:M masking/packing logic.

### [PyTorch tutorial — Pruning (torch.nn.utils.prune)](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
`read only` · `free` · `reference implementation`  
Size: one tutorial page; parent repo pytorch/tutorials has 9,255 stars  
Last activity: 2026-07 (parent repo actively maintained; page is current live docs)  

Official walkthrough of random/l1_unstructured, ln_structured, global_unstructured pruning APIs and writing a custom BasePruningMethod, with before/after sparsity printouts.

*Relation to this area:* Standard first stop for the PyTorch pruning API our tasks build on; purely a walkthrough, nothing checks understanding.

### [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
`read only` · `free` · `reference implementation`  
Size: 3,328 stars  
Last activity: 2025-09  

DepGraph-based structural pruning library (CVPR 2023): automatically identifies dependency groups so pruning one layer correctly removes every coupled parameter across CNNs, ViTs, LLMs, and diffusion models.

*Relation to this area:* The real-world tool for end-to-end structural pruning; has its own test suite for its own correctness but grades nothing about the learner — you call the library, not reimplement it.

### [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
`read only` · `free` · `reference implementation`  
Size: 7,261 stars  
Last activity: 2026-06  

CLI toolkit for merging pretrained LLM checkpoints (linear, SLERP, TIES, DARE, task-arithmetic) out-of-core on CPU/low VRAM; mergekit-extract-lora decomposes a fine-tune's weight delta into a PEFT-compatible LoRA adapter.

*Relation to this area:* Production tool for the LoRA-merging slice of this area; has its own CI tests but no learner-facing grading or reference answers to compare a given merge config against.

### [google-research/lottery-ticket-hypothesis](https://github.com/google-research/lottery-ticket-hypothesis)
`read only` · `free` · `reference implementation`  
Size: 731 stars  
Last activity: 2020-07  

Frankle & Carbin's own reimplementation: iterative magnitude pruning on MNIST fully-connected nets with weight-rewind-to-init between rounds, in TensorFlow 1.x. Archived/read-only by the owner.

*Relation to this area:* Ground-truth code for the exact 'reset to original init after each pruning round' mechanic our lottery-ticket tasks test. Dead project, TF1, MNIST-scale, no exercise scaffold.

### [he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
`read only` · `free` · `reading list`  
Size: 2,497 stars; ~300+ papers  
Last activity: 2024-04  

Curated bibliography of neural network pruning papers (2015-2023) organized by year and pruning type (filter/weight/other), with code links to the original authors' repos where available.

*Relation to this area:* Maps the literature our pruning task statements draw from (magnitude/movement/structured/lottery-ticket), but is pointers only — no common interface or grader across the linked repos.

### [huggingface/nn_pruning](https://github.com/huggingface/nn_pruning)
`read only` · `free` · `reference implementation`  
Size: 409 stars  
Last activity: 2022-06  

Movement pruning extended to semi-structured/block-structured variants so masks align to hardware-friendly tiles; demonstrated on BERT/SQuAD and GLUE. Archived Jul 2025.

*Relation to this area:* Ground-truth code for movement pruning specifically and for block-structured sparsity formats; archived and unmaintained.

### [locuslab/wanda](https://github.com/locuslab/wanda)
`read only` · `free` · `reference implementation`  
Size: 868 stars  
Last activity: 2024-08  

Official code for the Wanda pruning paper (ICLR 2024): prunes by |weight| x input-activation-norm per output, no retraining needed; also ships magnitude and SparseGPT baselines and supports 2:4/4:8 N:M structured patterns.

*Relation to this area:* Closest ground-truth reference for both movement/importance pruning and 2:4 structured sparsity, at LLM scale. Read-only research code, no starter/reference split.

### [microsoft/Tutel](https://github.com/microsoft/Tutel)
`read only` · `free` · `reference implementation`  
Size: 1,001 stars  
Last activity: 2026-07  

Production Mixture-of-Experts library: adaptive parallelism, dynamic capacity/routing switching, and the real dispatch/combine kernels behind fast sparse-MoE training and inference at scale.

*Relation to this area:* Real infrastructure our MoE-routing-sparsity tasks model a piece of; a systems library for large training jobs, no exercises or small-scale walkthrough of the routing math.

**What none of these do.** None of the real resources in this area auto-grade a learner's own pruning mask, distillation loss, sparse-kernel, or LoRA-merge code against a hidden reference and return a deterministic score. The awesome-lists are bibliographies; the paper repos (lottery-ticket, Wanda, nn_pruning) are frozen read-only demonstrations at one scale; the production libraries (Torch-Pruning, torchao, mergekit, Tutel) are tools you call, not exercises you complete; the two official PyTorch tutorials and the HF distillation doc are single narrated walkthroughs with no variation. MIT 6.5940 is the one partial exception (real in-notebook self-checks) but it is 5 labs covering pruning+quantization+NAS+deployment together, not ~125 separate graded problems spanning magnitude/movement/structured pruning, 2:4 sparsity, lottery tickets, CSR/CSC/BSR formats, KD losses, MoE routing, low-rank factorization, and LoRA merging, and its official infra is not openly self-serve. deep-ml.com, checked directly, has zero coverage of this area. Breadth + individual auto-grading + full offline reproducibility is where this bank is genuinely differentiated here.

<details><summary>Survey notes for this area</summary>

Checked deep-ml.com directly: it has no pruning/sparsity/distillation/LoRA/MoE problems at all (its listed categories are ML fundamentals, DNNs, CV, NLP, linear algebra) — genuinely absent, not missed. No "GPU-Puzzles"-style auto-graded puzzle set or leaderboard exists for this area anywhere I could find, despite trying "pruning puzzles", "distillation exercises", "leetprune"-style search shapes. Two entries (google-research/lottery-ticket-hypothesis, huggingface/nn_pruning, microsoft/nni) are formally archived/read-only on GitHub — flagged clearly, still useful to read but dead. The MIT 6.5940 course is real and current (Fall 2026 offering confirmed on hanlab.mit.edu/course) but I could not verify that its official lab starter notebooks + autograder are downloadable by a non-enrolled learner; what I verified is (a) the course/lecture content is public, and (b) a third-party GitHub mirror of a student's completed labs explicitly states the labs "passed all the tests in the notebook," which confirms the labs are test-driven even though I couldn't fetch a blank starter notebook myself. Sizes for GitHub repos (stars, last push) were pulled via the public GitHub API in this session, not estimated.

</details>

## How this page was built

One research pass per area, each instructed to fetch every URL before reporting it, to
record whether a project is alive, and explicitly **not** to pad a short list with
tangential material — an empty area is a finding, not a failure. Every URL was then
HTTP-checked independently of the agent that found it.

Two limits worth knowing. First, this is a snapshot: several projects here are dormant or
archived and are listed as such, but any of them can change. Second, absence of evidence is
weak evidence — a resource that exists but is not findable through GitHub search, Google, or
the bibliographies of the standard books in an area would not have been found.

Corrections are welcome. A dead link or a mischaracterised project on this page is a bug;
open an issue.
