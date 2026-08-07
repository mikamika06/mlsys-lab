import ref
import numpy as np


def check(workdir):
  from quant.hessian import compute_hessian, domain_discrepancy

  data_matched = ref.generate_data(10, 500, 16)
  data_mismatched = ref.generate_data(99, 500, 16)

  h_matched = compute_hessian(data_matched)
  h_mismatched = compute_hessian(data_mismatched)

  h_oracle = compute_hessian(data_matched)
  err = domain_discrepancy(h_oracle, h_matched)

  out = {"rel_err": float(err)}
  return out
