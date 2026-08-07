import hashlib
import json

FEATURE_OS_MAP = {
    "fp16": (14, 0),
    "int8_dotprod": (14, 5),
    "metal3": (16, 0),
    "vulkan_13": (15, 0),
}

DEVICES = [
    {
        "id": "dev_a",
        "os_version": (13, 5),
        "features": ["fp16"],
        "ram_mb": 3072,
    },
    {
        "id": "dev_b",
        "os_version": (14, 2),
        "features": ["fp16"],
        "ram_mb": 2048,
    },
    {
        "id": "dev_c",
        "os_version": (16, 1),
        "features": ["fp16", "int8_dotprod", "metal3"],
        "ram_mb": 6144,
    },
    {
        "id": "dev_d",
        "os_version": (15, 0),
        "features": ["fp16", "vulkan_13"],
        "ram_mb": 4096,
    },
    {
        "id": "dev_e",
        "os_version": (16, 5),
        "features": ["fp16", "int8_dotprod"],
        "ram_mb": 1024,
    },
]

VARIANTS = [
    {
        "id": "var_fp16",
        "min_os": (14, 0),
        "required_features": ["fp16"],
        "min_ram_mb": 2048,
        "download_bytes": 45000000,
        "utility": 80.0,
        "target_cohorts": ["cohort_1", "cohort_2"],
    },
    {
        "id": "var_metal3",
        "min_os": (15, 0),
        "required_features": ["fp16", "metal3"],
        "min_ram_mb": 4096,
        "download_bytes": 70000000,
        "utility": 95.0,
        "target_cohorts": ["cohort_2"],
    },
    {
        "id": "var_int8",
        "min_os": (14, 0),
        "required_features": ["fp16", "int8_dotprod"],
        "min_ram_mb": 2048,
        "download_bytes": 25000000,
        "utility": 70.0,
        "target_cohorts": ["cohort_1", "cohort_3"],
    },
    {
        "id": "var_vulkan",
        "min_os": (14, 5),
        "required_features": ["vulkan_13"],
        "min_ram_mb": 3072,
        "download_bytes": 35000000,
        "utility": 75.0,
        "target_cohorts": ["cohort_3"],
    },
]

RAW_MANIFESTS = [
    {
        "name": "model_alpha",
        "layers": [
            {"weight": [0.12345678, -0.98765432], "name": "dense1"},
            {"weight": [0.5, 0.25], "name": "embed"},
        ],
        "version": 2,
        "metadata": {"author": "ml-team", "precision": "fp16"},
    },
    {
        "name": "model_beta",
        "layers": [
            {"weight": [1.0000001, 0.0000002], "name": "layer_a"},
        ],
        "version": 1,
        "metadata": {"precision": "int8"},
    },
]


def compute_os_floor(variant, feature_os_map):
    req_os = variant.get("min_os", (0, 0))
    for feat in variant.get("required_features", []):
        if feat in feature_os_map:
            feat_os = feature_os_map[feat]
            if feat_os > req_os:
                req_os = feat_os
    return req_os


def filter_eligible_devices(devices, variant, feature_os_map):
    os_floor = compute_os_floor(variant, feature_os_map)
    eligible = []
    excluded = {}
    for dev in devices:
        dev_id = dev["id"]
        if dev["os_version"] < os_floor:
            excluded[dev_id] = "os_below_floor"
            continue
        missing_feat = [
            f for f in variant.get("required_features", []) if f not in dev["features"]
        ]
        if missing_feat:
            excluded[dev_id] = "missing_feature"
            continue
        if dev["ram_mb"] < variant.get("min_ram_mb", 0):
            excluded[dev_id] = "insufficient_ram"
            continue
        eligible.append(dev_id)
    return {"eligible": sorted(eligible), "excluded": excluded}


def convert_variant_manifest(raw_manifest):
    def clean_data(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [clean_data(x) for x in obj]
        return obj

    cleaned = clean_data(raw_manifest)
    json_bytes = json.dumps(
        cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(json_bytes).hexdigest()
    return {"manifest_bytes": json_bytes, "digest": digest}


def select_variant_set(variants, download_budget):
    valid_variants = [
        v for v in variants if v.get("download_bytes", 0) <= download_budget
    ]
    best_combo = []
    best_utility = -1.0

    n = len(valid_variants)
    for i in range(1 << n):
        subset = [valid_variants[j] for j in range(n) if (i & (1 << j))]
        total_size = sum(v["download_bytes"] for v in subset)
        if total_size <= download_budget:
            total_utility = sum(v["utility"] for v in subset)
            if total_utility > best_utility:
                best_utility = total_utility
                best_combo = subset
            elif total_utility == best_utility:
                if total_size < sum(v["download_bytes"] for v in best_combo):
                    best_combo = subset

    return sorted([v["id"] for v in best_combo])
