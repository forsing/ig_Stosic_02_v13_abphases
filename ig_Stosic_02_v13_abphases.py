from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
trag STD/GEO duž TRACE → next po trenutnoj fazi.

A/B faze STD/GEO duž istorije (paper 2 geo_ab_phase).

Za svaki historical query t (poslednjih TRACE kola):
  s_STD = p_cond − p_glob
  s_GEO = p_cond · s_STD
  phase_t = GEO ako strength(GEO)≥strength(STD) inače STD

Štampa trag (udeo GEO, broj prekida), poslednje faze.
Final next = kontroler trenutne faze na last.
Ban last. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
MIN_PAIR = 20
TRACE = 80
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def pair_transition_tables(draws: np.ndarray) -> tuple[dict, dict]:
    present = np.zeros((len(draws), FRONT_N), dtype=np.uint8)
    for i, d in enumerate(draws):
        for x in d.tolist():
            present[i, int(x) - 1] = 1
    pair_count: dict[tuple[int, int], int] = {}
    pair_next: dict[tuple[int, int], np.ndarray] = {}
    for t in range(len(draws) - 1):
        xs = np.where(present[t] == 1)[0]
        ys = np.where(present[t + 1] == 1)[0]
        for a, b in combinations(xs.tolist(), 2):
            key = (a, b) if a < b else (b, a)
            pair_count[key] = pair_count.get(key, 0) + 1
            if key not in pair_next:
                pair_next[key] = np.zeros(FRONT_N, dtype=np.float64)
            for yi in ys:
                pair_next[key][yi] += 1.0
    return pair_count, pair_next


def conditional_from_query_pairs(
    query: np.ndarray,
    pair_count: dict,
    pair_next: dict,
    min_pair: int = MIN_PAIR,
) -> np.ndarray:
    carriers = sorted(int(x) - 1 for x in query.tolist())
    masses = []
    for a, b in combinations(carriers, 2):
        key = (a, b)
        c = pair_count.get(key, 0)
        if c < min_pair:
            continue
        masses.append(pair_next[key] / float(c))
    if not masses:
        for a, b in combinations(carriers, 2):
            key = (a, b)
            c = pair_count.get(key, 0)
            if c <= 0:
                continue
            masses.append(pair_next[key] / float(c))
    if not masses:
        mass = np.ones(FRONT_N, dtype=np.float64)
    else:
        mass = np.mean(np.stack(masses, axis=0), axis=0)
    mass = mass + 1e-6
    return mass / mass.sum()


def strength(vec: np.ndarray, ban: set[int]) -> float:
    mask = np.ones(FRONT_N, dtype=bool)
    for x in ban:
        mask[x - 1] = False
    return float(np.linalg.norm(np.clip(vec[mask], 0, None)))


def phase_at(
    query: np.ndarray,
    p_glob: np.ndarray,
    pair_count: dict,
    pair_next: dict,
) -> tuple[str, float, float, np.ndarray, np.ndarray]:
    ban = set(int(x) for x in query.tolist())
    p_cond = conditional_from_query_pairs(query, pair_count, pair_next)
    s_std = p_cond - p_glob
    s_geo = p_cond * s_std
    str_std = strength(s_std, ban)
    str_geo = strength(s_geo, ban)
    phase = "GEO" if str_geo >= str_std else "STD"
    return phase, str_std, str_geo, s_std, s_geo


def scores_dict(vec: np.ndarray, ban: set[int]) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        out[n] = -1e18 if n in ban else float(vec[i])
    return out


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
    ban: set[int],
) -> float:
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws: np.ndarray, score: dict[int, float], ban: set[int]) -> list[int]:
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v13(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    p_glob = global_p(draws)
    pair_count, pair_next = pair_transition_tables(draws)

    start = max(0, len(draws) - TRACE)
    trace = []
    for t in range(start, len(draws)):
        phase, str_std, str_geo, _, _ = phase_at(
            draws[t], p_glob, pair_count, pair_next
        )
        # phase code: 0=STD, 1=GEO (kao geo_ab_phase)
        code = 1 if phase == "GEO" else 0
        trace.append((t, code, phase, str_std, str_geo))

    phases = [c for _, c, *_ in trace]
    switches = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1])
    geo_frac = float(np.mean(phases)) if phases else 0.0

    cur_phase, str_std, str_geo, s_std, s_geo = phase_at(
        last, p_glob, pair_count, pair_next
    )
    vec = s_geo if cur_phase == "GEO" else s_std
    score = scores_dict(vec, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | TRACE={TRACE} | ig_02_v13 A/B phases")
    print(f"last: {last.tolist()}")
    print()

    print("=== trag faza (0=STD, 1=GEO) ===")
    print(
        {
            "geo_frac": round(geo_frac, 4),
            "switches": switches,
            "trace_n": len(trace),
            "phase_now": cur_phase,
            "strength_STD_now": round(str_std, 6),
            "strength_GEO_now": round(str_geo, 6),
        }
    )
    print("poslednjih 8:", [(t, c, ph) for t, c, ph, *_ in trace[-8:]])
    print()

    print(f"=== next (ig_02_v13 fazni kontroler={cur_phase}) ===")
    print("next:", combo)
    print("overlap last:", sorted(set(combo) & ban))


if __name__ == "__main__":
    run_ig_02_v13()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | TRACE=80 | ig_02_v13 A/B phases
last: [4, 5, 6, 11, 12, 18, 28]
00x 
=== trag faza (0=STD, 1=GEO) ===
{'geo_frac': 0.0, 'switches': 0, 'trace_n': 80, 'phase_now': 'STD', 'strength_STD_now': 0.00612, 'strength_GEO_now': 0.000174}
poslednjih 8: [(4642, 0, 'STD'), (4643, 0, 'STD'), (4644, 0, 'STD'), (4645, 0, 'STD'), (4646, 0, 'STD'), (4647, 0, 'STD'), (4648, 0, 'STD'), (4649, 0, 'STD')]

=== next (ig_02_v13 fazni kontroler=STD) ===
next: [9, 10, 15, 16, 25, 29, 36]
overlap last: []
"""
