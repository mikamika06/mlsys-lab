def classify_accuracy_collapse(baseline_acc, reduced_acc_list, threshold=0.15):
    collapsed_sizes = []
    for item in reduced_acc_list:
        size = item["size"]
        acc = item["accuracy"]
        if (baseline_acc - acc) > threshold:
            collapsed_sizes.append(size)
    return {
        "is_collapsed": len(collapsed_sizes) > 0,
        "collapsed_sizes": sorted(collapsed_sizes)
    }
