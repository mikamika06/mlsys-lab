# How many bytes on disk do installed models actually take up

A local model registry stores weights in content-addressed blobs:
each blob has a digest key, and the same blob can be referenced by
several tags at once — a shared tokenizer, a shared license, and sometimes
shared layers between quants of the same model. The utility that reports
"how much space the installed models take up" currently just sums the size
of each tag separately — and reports a number noticeably bigger than what's
actually on disk. Because of that, an admin deletes tags and frees less
space than expected, and misjudges how much room a new `pull` needs, then
runs out of disk mid-download.

We need the real numbers: how many bytes are actually unique, how much will
actually need to be downloaded for one more quant of an already-installed
model, and which blobs nobody needs anymore.

## What you write

`blobstore/index.py` — `build_blob_index(config) -> list[blob]`. The config
is `{"tags": {tag_name: [{"digest", "size"}, ...], ...}}`. The same
`digest` can repeat across several tags — that's one and the same blob on
disk, not two different ones. A blob is `{"digest", "size", "tags"}`, where
`tags` is a sorted list of the tag names that reference it. The list of
blobs is ordered by `digest`.

`blobstore/cost.py`:

```python
unique_bytes_on_disk(config)
naive_total_bytes(config)
incremental_pull_bytes(config, candidate)
```

`unique_bytes_on_disk` — the sum of sizes over each unique `digest`, counted
exactly once. `naive_total_bytes` — the sum of sizes per tag separately,
with duplicates (what the naive summer currently computes).
`incremental_pull_bytes(config, candidate)` — how many bytes will actually
need to be downloaded if you install one more tag `candidate` (same format
as the blob list inside a tag): everything whose `digest` isn't already
among the currently installed tags in `config`, and not counted twice if
`candidate` itself references the same blob more than once.

`blobstore/gc.py` — `find_orphaned_blobs(config, disk_blobs)` and
`orphaned_bytes(config, disk_blobs)`. `disk_blobs` is `{digest: size}`,
what's physically sitting in blob storage. An orphaned blob is one that's
on disk but not referenced by any tag in `config`. `find_orphaned_blobs`
returns a sorted list of such `digest`s, `orphaned_bytes` — the sum of
their sizes.

## How it's graded

The grader computes the reference answer itself, from the same configs,
across several installs and pull scenarios. The third milestone is yours:
you write a test, and we swap in an index builder that merges two
different blobs into a single record whenever they happen to have the
same size. Your test needs to catch that.

```
mlsys project start m-compute-true-unique-bytes-on-disk-across-all-install
mlsys project grade m-compute-true-unique-bytes-on-disk-across-all-install --milestone 1
```
