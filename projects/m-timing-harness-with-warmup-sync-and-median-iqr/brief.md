Our CI performance tracking has become completely useless over the past two weeks. The performance numbers reported for our core layers are fluctuating by as much as 40% between consecutive runs on the exact same hardware. Yesterday, we even merged a 'performance regression' alert that turned out to be an actual optimization once we profiled it manually.

A colleague recently replaced our benchmarking logic with a custom script, and I suspect there are several serious methodology issues in how it measures time. Currently, it just starts a timer, runs the model 100 times in a single loop, stops the timer, and computes the mean.

We need a robust, statistically sound benchmarking harness.
Please implement a proper `benchmark_step` function. It needs to run a specified number of warmup iterations first. Then, it should time independent repetitions, ensuring that asynchronous GPU operations are fully synchronized if CUDA is used. Finally, it must return the median and the Interquartile Range (IQR) to be resilient against outliers.

Additionally, implement `compute_required_reps`. Use the sample size formula `(Z * sample_std / margin)^2`, where margin is `tolerance * sample_mean`. Use `ddof=1` for your sample standard deviation and return the ceiling of the final computed value.
