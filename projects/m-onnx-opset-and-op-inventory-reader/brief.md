We are running out of memory deploying ONNX models to our low-power edge devices. A major bottleneck is the deployment container size itself: just to introspect a model file before inference (to check if our stripped-down ONNX Runtime `.ort` build supports its operators), we have to install the massive Python `onnx` package. This pulls in a huge protobuf dependency and inflates our image by hundreds of megabytes.

We need a pure-Python, zero-dependency way to read ONNX binaries just enough to extract the required operator sets and count the operators. Because ONNX uses a standard Protobuf layout, we can manually parse the wire format.

We also want to estimate how many bytes we will save when converting the `.onnx` model to an `.ort` format. The ORT converter strips out node names (field 3) and docstrings (field 6) from the graph nodes.

Build a lightweight binary parser that walks the `.onnx` protobuf wire format to find `opset_import` (field 8), `graph` (field 7), and inside the graph, the `node` elements (field 5). Extract the operator types and calculate the total size of the stripped fields.
