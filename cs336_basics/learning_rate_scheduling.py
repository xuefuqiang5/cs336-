import math
def lr_schedule(
    t: int,
    T_w: int,
    T_c: int,
    alpha_max: float,
    alpha_min: float,
) -> float:
    """
    Warm-up + Cosine Annealing + Constant LR schedule
    """

    # Warm-up
    if t < T_w:
        return alpha_max * t / T_w

    # Cosine annealing
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine * (alpha_max - alpha_min)

    # Post-annealing
    else:
        return alpha_min