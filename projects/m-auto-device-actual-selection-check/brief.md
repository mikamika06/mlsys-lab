# AUTO Device Selection and Compile-Success Matrix Verification

We are observing unpredictable behavior across our deployment pipeline when delegating device targeting to OpenVINO's `AUTO` execution mode. Downstream consumers report silent execution fallbacks and runtime compilation failures when handling different model input dynamicities across heterogeneous hardware topologies.

Specifically, the platform team needs a robust introspection tool that verifies which physical device `AUTO` actually selected for compiled models, along with an automated shape compatibility validator. Without these checks, static-shape models optimized for specific accelerators crash during payload inference, or quietly run on low-throughput fallback devices.

To address this gap, you will implement an inspection framework that queries OpenVINO runtime properties to determine actual physical device allocation under `AUTO` hints, builds a compile-success matrix testing static versus dynamic shapes across mock device capabilities, and writes regression tests to ensure device resolution logic never fails silently on invalid targets.
