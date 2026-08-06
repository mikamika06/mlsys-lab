# Detect benchmark runs with folded load duration in throughput

Our local runner automated regression suite compares performance metrics across multiple target hardware variants and model configurations. Recently, engineers noticed that certain benchmark runs report abnormally low tokens-per-second values when running short prompts or lightweight models where initialization overhead is prominent, while other runs show expected generation speeds.

Upon inspecting raw logs, we suspect that specific benchmark runner scripts inadvertently include the model initialization and weight loading time (`load_duration`) inside the denominator when computing generation throughput (`tok/s`), rather than dividing purely by active generation duration (`generation_duration`). This distorts performance comparisons and masks true decoding throughput regressions.

Your task is to build an auditing module that parses raw benchmark result dictionaries, computes expected versus reported token throughput metrics, detects whether model load time was incorrectly folded into the throughput calculation, and writes a robust regression test ensuring this issue is caught reliably.
