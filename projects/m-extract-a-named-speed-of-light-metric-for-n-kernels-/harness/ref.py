import csv
import io

def generate_sample_csv():
    headers = ["ID", "Kernel Name", "Metric Name", "Metric Value", "Section Name", "Warning"]
    rows = [
        ["1", "kernel_matmul", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "78.5", "SpeedOfLight", ""],
        ["2", "kernel_softmax", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "62.1", "SpeedOfLight", ""],
        ["3", "kernel_layer norm", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "45.0", "SpeedOfLight", ""],
    ]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)
    return out.getvalue()

def generate_basic_and_full_csv():
    headers = ["ID", "Kernel Name", "Metric Name", "Metric Value", "Section Name", "Warning"]
    basic_rows = [
        ["1", "kernel_matmul", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "78.5", "SpeedOfLight", ""],
    ]
    full_rows = [
        ["1", "kernel_matmul", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "78.5", "SpeedOfLight", ""],
        ["2", "kernel_matmul", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "50.2", "MemoryWorkloadAnalysis", ""],
        ["3", "kernel_matmul", "l1tex__t_sectors_pipe_lsu_mem_op_read.sum", "1024", "LaunchStats", ""],
    ]

    b_out = io.StringIO()
    bw = csv.writer(b_out)
    bw.writerow(headers)
    for r in basic_rows:
        bw.writerow(r)

    f_out = io.StringIO()
    fw = csv.writer(f_out)
    fw.writerow(headers)
    for r in full_rows:
        fw.writerow(r)

    return b_out.getvalue(), f_out.getvalue()

def generate_warning_csv():
    headers = ["ID", "Kernel Name", "Metric Name", "Metric Value", "Section Name", "Warning"]
    rows = [
        ["1", "kernel_unstable", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "99.0", "SpeedOfLight", "kernel replay mismatch warning"],
    ]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)
    return out.getvalue()
