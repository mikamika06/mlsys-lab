import ref

def check(workdir):
  from workflow.vram_reconcile import reconcile_moe_vram
  from workflow.installer_parse import parse_installer_transcript

  out = {"vram_reconciled": 0.0, "installer_parsed": 0.0}

  want_vram = ref.reconcile_moe_vram(ref.MOE_CONFIG)
  got_vram = reconcile_moe_vram(ref.MOE_CONFIG)
  if isinstance(got_vram, dict) and "bf16_size_gb" in got_vram and "overhead_gb" in got_vram:
    err1 = abs(got_vram["bf16_size_gb"] - want_vram["bf16_size_gb"])
    err2 = abs(got_vram["overhead_gb"] - want_vram["overhead_gb"])
    if err1 < 1e-3 and err2 < 1e-3:
      out["vram_reconciled"] = 1.0
    else:
      out["_note"] = f"VRAM mismatch: got {got_vram}, expected {want_vram}"
  else:
    out["_note"] = "VRAM reconciliation returned invalid format"

  want_packages = ref.parse_installer_transcript(ref.INSTALLER_TRANSCRIPT)
  got_packages = parse_installer_transcript(ref.INSTALLER_TRANSCRIPT)
  if got_packages == want_packages:
    out["installer_parsed"] = 1.0
  else:
    out["_note"] = f"Installer parse mismatch: got {got_packages}, expected {want_packages}"

  return out
