from typing import List



# parses layer index range written in flexible string formats - comma seperated + hyphenated into a list
def parse_layer_subset(spec: str, n_layers: int) -> List[int]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(n_layers))
    out: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-")
            lo, hi = int(a), int(b)
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(p))
    out = sorted(set(x for x in out if 0 <= x < n_layers))
    if not out:
        raise ValueError(f"Layer subset {spec!r} produced no valid indices for n_layers={n_layers}")
    return out


# parses flexible range written in string to a list of floats
def parse_fraction_grid(spec: str) -> List[float]:
    vals = []
    for part in spec.split(","):
        x = float(part.strip())
        if not (0.0 < x <= 1.0):
            raise ValueError(f"Invalid retained fraction {x}; must be in (0,1].")
        vals.append(x)
    vals = sorted(set(vals))
    return vals