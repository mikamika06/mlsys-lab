import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from bench_analysis.parser import extract_tensor_bytes, parse_do_bench_trace

    records = ref.generate_test_records()
    parsed_count = 0.0

    for idx, rec in enumerate(records):
        try:
            u_res = parse_do_bench_trace(rec["unfused_trace"])
            f_res = parse_do_bench_trace(rec["fused_trace"])
            ref_u_mean, ref_u_std = ref.reference_parse_trace(
                rec["unfused_trace"]
            )
            ref_f_mean, ref_f_std = ref.reference_parse_trace(rec["fused_trace"])

            if (
                abs(u_res["mean_ms"] - ref_u_mean) < 1e-4
                and abs(u_res["std_ms"] - ref_u_std) < 1e-4
                and abs(f_res["mean_ms"] - ref_f_mean) < 1e-4
                and abs(f_res["std_ms"] - ref_f_std) < 1e-4
            ):
                parsed_count += 1.0
            else:
                return {
                    "traces_parsed": parsed_count,
                    "_note": f"Mismatch on record {idx}",
                }
        except Exception as e:
            return {
                "traces_parsed": parsed_count,
                "_note": f"Error during parsing: {e}",
            }

    try:
        b_got = extract_tensor_bytes([1024, 1024], "float32")
        if b_got != 1024 * 1024 * 4:
            return {
                "traces_parsed": 0.0,
                "_note": "extract_tensor_bytes produced incorrect byte count",
            }
    except Exception as e:
        return {
            "traces_parsed": 0.0,
            "_note": f"extract_tensor_bytes failed: {e}",
        }

    return {"traces_parsed": parsed_count}
