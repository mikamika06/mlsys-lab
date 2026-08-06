# Ticket: Lightweight Inspection and Parsing of .mlpackage Bundles Without CoreMLTools

## Symptom
When deploying edge models packaged as Apple `.mlpackage` directories into restricted embedded environments or custom runtime runners, dependency bloat prevents the installation of heavy frameworks like `coremltools`. Engineers attempting to inspect model structures or verify bundle integrity find themselves unable to read the internal `Manifest.json`, compute accurate disk usage breakdowns for individual asset files, or parse binary MIL (Model Intermediate Language) weight blob headers. Consequently, deployment pipelines fail early with opaque parsing errors, or worse, ship packages with mismatched weight files and broken asset references that only manifest as runtime segmentation faults on target hardware.

## Context
An `.mlpackage` is a directory bundle containing a `Manifest.json` file at its root, a `Data/` directory housing model definitions, and binary weight blobs. Without standard heavy tooling available in production runtimes, we need a lightweight, pure-Python module capable of parsing the manifest structure, performing precise byte attribution across all contained artifacts, and safely decoding custom binary MIL weight blob headers.

## Objective
Implement a self-contained library under `mlpackage/` that correctly extracts manifest details and summarizes model components, calculates exact byte contributions of package files, and correctly parses binary MIL weight headers, accompanied by a rigorous regression test suite.
