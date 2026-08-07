import sys
import ref


def check(workdir):
  if workdir not in sys.path:
    sys.path.insert(0, workdir)

  from break_analyzer.tracer import count_graph_breaks, rank_snippets

  out = {"ranking_accuracy": 0.0, "break_counts_matched": 0}

  snippets = ref.SNIPPETS
  ref_counts = [ref.count_graph_breaks(s) for s in snippets]

  matched_counts = 0
  for i, s in enumerate(snippets):
    try:
      got_count = count_graph_breaks(s)
      if got_count == ref_counts[i]:
        matched_counts += 1
    except Exception:
      pass

  out["break_counts_matched"] = matched_counts

  ref_ranked = ref.rank_snippets(snippets)
  try:
    got_ranked = rank_snippets(snippets)
    ref_indices = [r["index"] for r in ref_ranked]
    got_indices = [r["index"] for r in got_ranked]
    if got_indices == ref_indices:
      out["ranking_accuracy"] = 1.0
    else:
      out["_note"] = f"Expected ranking {ref_indices}, got {got_indices}"
  except Exception as e:
    out["_note"] = f"rank_snippets failed: {type(e).__name__}: {e}"

  return out
