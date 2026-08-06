Issue Ticket: Edge deployment pipeline lacks standardized PTQ tradeoff metrics
Reporter: Mobile Runtime Optimization Team

Description:
Our edge deployment pipeline converts neural network backbones for low-power mobile devices and embedded microcontrollers using LiteRT / TFLite post-training quantization. During recent field trials, several vision models exported with default integer quantization exhibited significant accuracy degradation, while models exported with dynamic range quantization caused frame rate drops and high power consumption on budget hardware.

Engineers currently select quantization flags (such as Float16, Dynamic Range INT8, or Full Integer INT8) without visibility into how each post-training quantization mode affects memory footprint reduction, accuracy drift (MSE), and memory-bandwidth/compute-bound latency. Without a standardized evaluation harness to compare these parameters across all post-training modes prior to export, teams are forced to manually deploy multiple candidate artifacts to hardware targets, creating lengthy release delays and unpredictable performance regressions.

Target Requirement:
We need an evaluation utility in `quanteval/modes.py` and `quanteval/table.py` that simulates quantization transformations across four standard PTQ modes (FP32 baseline, FP16, Dynamic Range INT8, Full INT8), calculates layer parameter footprint in bytes, computes output mean-squared error drift against FP32 reference activations, estimates roofline execution latency for target hardware configs, and outputs a complete summary trade-off table.
