def discretize(val, min_val, max_val, bin_count):
    discrete_val = int(bin_count * normalize(
        val,
        min_val,
        max_val,
        ))
    return enforce_bins(
        discrete_val,
        0,
        bin_count - 1,
        )


def normalize(val, min_val, max_val):
    return (float(val) - min_val) / (float(max_val) - min_val)


def enforce_bins(val, min_bin, max_bin):
    if val < min_bin:
        val = min_bin
    elif val > max_bin:
        val = max_bin
    return val
