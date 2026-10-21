"""Détection de signal sur les fiches réaction : ROR, PRR, BCPNN, GPS.

Formules reprises de vigipy (github.com/pavanvincent/vigipy) :
  - ROR   : IC 95 % de Woolf, p-value de Wald unilatérale, signal si borne basse > 1
  - PRR   : IC 95 %, p-value, Chi² de Yates, signal d'Evans (PRR ≥ 2, Chi² ≥ 4, n11 ≥ 3)
  - BCPNN : IC (information component) analytique, bornes 95 %, P(H0), signal si 2^IC025 > 1
  - GPS   : EBGM, EB05 / EB95 (mélange Gamma–Poisson, DuMouchel), P(H0), signal si EB05 ≥ 2

Tableau 2×2 au niveau du SIGNALEMENT, dans le périmètre (L1, L2 ou substance) :
  N   = signalements du périmètre
  n1j = signalements où la substance est suspecte
  ni1 = signalements contenant la réaction
  n11 = signalements avec la substance ET la réaction

Les a priori du GPS sont estimés par maximum de vraisemblance sur TOUTES les paires
substance × réaction de la classe L1 parente, puis mis en cache sur disque
(raw_stats_cache/_gps_priors.json) : l'estimation n'est faite qu'une fois par classe.

Installation dans le serveur (après fast_profiles.install) :
    import signal_methods
    signal_methods.install(RAW_STATS_ENGINE)
et, juste avant le bloc  if __name__ == "__main__":
    signal_methods.install_routes(globals())
"""

import json
import math
import os
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, gdtr, polygamma
from scipy.stats import norm

import fast_profiles as fp

MIN_EVENTS = 4
EPS = float(np.finfo(np.float32).eps)
GPS_BOUNDS = ((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1))
GPS_INIT = np.array([0.2041, 0.05816, 1.415, 1.838, 0.0969])
GPS_FIT_MAX_POINTS = 150_000          # échantillon pour l'estimation des a priori
METHODS = ("ror", "prr", "bcpnn", "gps")


# ======================================================================
# Méthodes (vectorisées, d'après vigipy)
# ======================================================================

def _exp_safe(x):
    out = np.full_like(x, np.inf, dtype=np.float64)
    ok = x <= np.log(np.finfo(np.float64).max)
    out[ok] = np.exp(x[ok])
    return out


def ror_stats(n11, n1j, ni1, N):
    n10 = n1j - n11 + 1e-7
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)
    log_ror = np.log(n11 * n00 / (n10 * n01))
    sd = np.sqrt(1.0 / n11 + 1.0 / n10 + 1.0 / n01 + 1.0 / n00)
    lb = np.exp(norm.ppf(0.025, log_ror, sd))
    ub = _exp_safe(norm.ppf(0.975, log_ror, sd))
    p = np.clip(1 - norm.cdf(log_ror, 0.0, sd), 0, 1)
    return {"value": np.exp(log_ror), "lb": lb, "ub": ub, "p": p, "signal": lb > 1}


def prr_stats(n11, n1j, ni1, N):
    n10 = n1j - n11
    n01 = ni1 - n11 + 1e-7
    n00 = N - (n11 + n10 + n01)
    log_prr = np.log((n11 / (n11 + n10)) / (n01 / (n01 + n00)))
    sd = np.sqrt(np.maximum(1 / n11 - 1 / (n11 + n10) + 1 / n01 - 1 / (n01 + n00), 1e-300))
    prr = np.exp(log_prr)
    lb = np.exp(norm.ppf(0.025, log_prr, sd))
    ub = _exp_safe(norm.ppf(0.975, log_prr, sd))
    p = np.clip(1 - norm.cdf(log_prr, 0.0, sd), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = (N * (np.maximum(0, np.abs(n11 * n00 - n10 * n01) - N / 2) ** 2)) / (
            (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    chi2 = np.nan_to_num(chi2, nan=0.0, posinf=0.0)
    signal = (n11 >= 3) & (chi2 >= 4) & (prr >= 2)
    return {"value": prr, "lb": lb, "ub": ub, "p": p, "chi2": chi2, "signal": signal}


def bcpnn_stats(n11, n1j, ni1, N):
    tri = lambda x: polygamma(1, x)
    p1, p2 = 1 + n1j, 1 + N - n1j
    q1, q2 = 1 + ni1, 1 + N - ni1
    r1 = 1 + n11
    r2b = N - n11 - 1 + (2 + N) ** 2 / (q1 * p1)
    ic = (digamma(r1) - digamma(r1 + r2b) - (digamma(p1) - digamma(p1 + p2) + digamma(q1) - digamma(q1 + q2))) / np.log(2)
    var = ((tri(r1) - tri(r1 + r2b)) + (tri(p1) - tri(p1 + p2)) + (tri(q1) - tri(q1 + q2))) / np.log(2) ** 2
    sd = np.sqrt(var)
    lb = norm.ppf(0.025, ic, sd)
    ub = norm.ppf(0.975, ic, sd)
    p_h0 = norm.cdf(0.0, ic, sd)
    return {"value": 2.0 ** ic, "lb": 2.0 ** lb, "ub": 2.0 ** ub, "ic": ic, "ic025": lb,
            "p_h0": p_h0, "signal": lb > 0}


def _gps_quantiles(threshold, Q, a1, b1, a2, b2, max_iter=35, tol=1e-5):
    M = np.maximum(1000.0, 10.0 * np.maximum(a1 / np.maximum(b1, 1e-6), a2 / np.maximum(b2, 1e-6)))
    m = np.zeros_like(M)
    x = np.ones_like(M)
    for _ in range(max_iter):
        cost = np.nan_to_num(Q * gdtr(b1, a1, x)) + np.nan_to_num((1.0 - Q) * gdtr(b2, a2, x)) - threshold
        if np.all(np.abs(cost) < tol):
            break
        S = np.sign(cost)
        xnew = np.where(S > 0, (x + m) * 0.5, (M + x) * 0.5)
        M = np.where(S > 0, x, M)
        m = np.where(S > 0, m, x)
        x = xnew
    return x


def _nb_logpmf(n11, r, beta, E, gl):
    p = np.clip(beta / (beta + E + 1e-10), 1e-10, 1 - 1e-10)
    return gammaln(n11 + r) - gl - gammaln(r) + r * np.log(p) + n11 * np.log(1.0 - p)


def _neg_loglik(pr, n11, E, gl):
    d1 = np.exp(_nb_logpmf(n11, pr[0], pr[1], E, gl))
    d2 = np.exp(_nb_logpmf(n11, pr[2], pr[3], E, gl))
    return np.sum(-np.log(pr[4] * d1 + (1.0 - pr[4]) * d2 + 1e-7))


def fit_gps_priors(n11, E, max_points=GPS_FIT_MAX_POINTS, seed=0):
    n11 = np.asarray(n11, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    if len(n11) > max_points:
        idx = np.random.default_rng(seed).choice(len(n11), max_points, replace=False)
        n11, E = n11[idx], E[idx]
    gl = gammaln(n11 + 1.0)
    res = minimize(_neg_loglik, x0=GPS_INIT, args=(n11, E, gl), method="SLSQP",
                   bounds=GPS_BOUNDS, options={"maxiter": 500})
    return [float(x) for x in res.x], str(res.message)


def gps_stats(n11, n1j, ni1, N, priors):
    a1, b1, a2, b2, w = priors
    E = n1j * ni1 / N
    gl = gammaln(n11 + 1.0)
    q1 = np.exp(_nb_logpmf(n11, a1, b1, E, gl))
    q2 = np.exp(_nb_logpmf(n11, a2, b2, E, gl))
    den = w * q1 + (1 - w) * q2
    Qn = np.where(den > 0, w * q1 / np.where(den > 0, den, 1), w)
    p_h0 = Qn * gdtr(np.maximum(b1 + E, 1e-10), a1 + n11, 1.0) + (1 - Qn) * gdtr(np.maximum(b2 + E, 1e-10), a2 + n11, 1.0)
    eblog2 = (Qn * (digamma(a1 + n11) - np.log(np.maximum(b1 + E, 1e-10)))
              + (1 - Qn) * (digamma(a2 + n11) - np.log(np.maximum(b2 + E, 1e-10)))) / np.log(2)
    eb05 = _gps_quantiles(0.05, Qn, a1 + n11, b1 + E, a2 + n11, b2 + E)
    eb95 = _gps_quantiles(0.95, Qn, a1 + n11, b1 + E, a2 + n11, b2 + E)
    return {"value": 2.0 ** eblog2, "lb": eb05, "ub": eb95, "expected": E, "p_h0": p_h0, "signal": eb05 >= 2}


def all_methods(n11, n1j, ni1, N, priors):
    n11, n1j, ni1 = (np.asarray(x, dtype=np.float64) for x in (n11, n1j, ni1))
    N = np.asarray(N, dtype=np.float64)
    return {
        "ror": ror_stats(n11, n1j, ni1, N),
        "prr": prr_stats(n11, n1j, ni1, N),
        "bcpnn": bcpnn_stats(n11, n1j, ni1, N),
        "gps": gps_stats(n11, n1j, ni1, N, priors),
    }


# ======================================================================
# Intégration au moteur
# ======================================================================

def _num(x):
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _parent_l1(engine, scope, code):
    if scope == "l1":
        return code
    if scope == "l2":
        return code[0]
    try:
        return engine._virtual_substance_parent(code)
    except Exception:
        return "ALL"   # substance hors classe ATC : a priori GPS de toute la base


def _priors_file(engine):
    root = getattr(engine, "root", None)
    return (root / "_gps_priors.json") if root is not None else None


def _load_priors_disk(engine):
    f = _priors_file(engine)
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f and f.exists() else {}
    except Exception:
        return {}


def _save_priors_disk(engine, key, value):
    f = _priors_file(engine)
    if not f:
        return
    data = _load_priors_disk(engine)
    data[key] = value
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, f)


_PRIOR_LOCK = threading.Lock()


def gps_priors(engine, l1_code):
    """A priori GPS de la classe L1 (mémoire > disque > estimation)."""
    key = f"l1:{l1_code}"
    cache = engine._derived.setdefault(("signal", "gps_priors"), {})
    if key in cache:
        return cache[key]
    with _PRIOR_LOCK:
        if key in cache:
            return cache[key]
        disk = _load_priors_disk(engine)
        if key in disk:
            cache[key] = disk[key]
            return cache[key]
        if os.environ.get("EUVIGIL_FIT_PRIORS") != "1":
            # Dans le serveur : jamais d'estimation à la volée (trop gourmand en mémoire).
            value = {"priors": [float(x) for x in GPS_INIT], "pairs": 0, "fitted_on": 0,
                     "message": "a priori par défaut (DuMouchel) — lancer precalcul_gps.py"}
            cache[key] = value
            print(f"[signal] a priori GPS {key} absents : valeurs par défaut", file=sys.stderr, flush=True)
            return value
        t0 = time.time()
        n11, E, n_pairs = _all_pairs_counts(engine, "l1", l1_code)
        pri, msg = fit_gps_priors(n11, E)
        value = {"priors": pri, "pairs": int(n_pairs), "fitted_on": int(min(n_pairs, GPS_FIT_MAX_POINTS)),
                 "message": msg, "seconds": round(time.time() - t0, 1),
                 "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        cache[key] = value
        _save_priors_disk(engine, key, value)
        print(f"[signal] a priori GPS {key} estimés en {value['seconds']} s sur {value['fitted_on']} paires",
              file=sys.stderr, flush=True)
        return value


def _all_pairs_counts(engine, scope, code):
    """Comptes n11 et E de toutes les paires substance × réaction du périmètre."""
    _, _, _, pop_ids = fp._rx_population(engine, scope, code)
    inpop, totals = fp._substance_totals(engine, scope, code)
    pairs = fp._pairs(engine, scope, code)
    N = len(pop_ids)

    ids = pd.Index(list(pop_ids))
    s_rows = ids.get_indexer(inpop["safety_report_id"])
    s_codes, s_uni = pd.factorize(inpop["substance"])
    ok = s_rows >= 0
    S = sparse.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (s_rows[ok], s_codes[ok])),
                          shape=(len(ids), len(s_uni)))
    S.data[:] = 1

    r_rows = ids.get_indexer(pairs["id"])
    r_codes, r_uni = pd.factorize(pairs["key"].astype(str))
    ok = r_rows >= 0
    R = sparse.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (r_rows[ok], r_codes[ok])),
                          shape=(len(ids), len(r_uni)))
    R.data[:] = 1
    R.sum_duplicates()
    R.data[:] = 1

    C = (S.T @ R).tocoo()
    n1j = np.asarray((S > 0).sum(axis=0)).ravel().astype(np.float64)
    ni1 = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float64)
    n11 = C.data.astype(np.float64)
    E = n1j[C.row] * ni1[C.col] / N
    return n11, E, len(n11)


def _reaction_context(engine, scope, code, reaction):
    scope, code, _ = engine._scope_path(scope, code)
    reaction = str(reaction).strip()
    if not reaction:
        raise ValueError("réaction absente")
    hit = fp._rx_index(engine, scope, code).get(reaction.casefold())
    if hit is None:
        raise ValueError(f"réaction introuvable : {reaction}")
    canonical, arr = hit
    reaction_ids = set(pd.Series(arr).dropna().tolist())
    population, _, _, pop_ids = fp._rx_population(engine, scope, code)
    inpop, totals = fp._substance_totals(engine, scope, code)
    return scope, code, canonical, reaction_ids, population, pop_ids, inpop, totals


def reaction_signals(engine, scope, code, reaction, min_events=MIN_EVENTS):
    key = ("signal", "reaction", str(scope).lower(), str(code).upper(), str(reaction).strip().casefold(), int(min_events))
    with engine._lock:
        cached = engine._derived.get(key)
    if cached is not None:
        return cached

    scope, code, canonical, reaction_ids, population, pop_ids, inpop, totals = _reaction_context(engine, scope, code, reaction)
    N = len(pop_ids)
    ni1 = len(reaction_ids & pop_ids)
    a = inpop.loc[inpop["safety_report_id"].isin(reaction_ids), "substance"].value_counts()
    a = a[a >= int(min_events)]
    subs = [s for s in a.index.tolist() if s in totals.index]
    try:
        import scope_members
        _member = scope_members.scope_members(engine, scope, code)
    except Exception:
        _member = None
    n_all_subs = len(subs)
    if _member is not None:
        subs = [s for s in subs if _member(s)]
    n11 = a.loc[subs].to_numpy(dtype=np.float64)
    n1j = totals.loc[subs].to_numpy(dtype=np.float64)
    ni1_arr = np.full_like(n11, float(ni1))

    pri = gps_priors(engine, _parent_l1(engine, scope, code))
    res = all_methods(n11, n1j, ni1_arr, N, pri["priors"])

    rows = []
    for i, s in enumerate(subs):
        row = {"substance": str(s), "n11": int(n11[i]), "n1j": int(n1j[i]),
               "share": n11[i] / n1j[i] if n1j[i] else 0}
        for m in METHODS:
            r = res[m]
            row[m] = {k: (bool(v[i]) if k == "signal" else _num(v[i])) for k, v in r.items()}
        row["signals"] = sum(1 for m in METHODS if row[m]["signal"])
        rows.append(row)
    rows.sort(key=lambda x: (x["ror"]["lb"] if isinstance(x["ror"]["lb"], float) else 0), reverse=True)

    out = {
        "scope": {"type": scope, "code": code},
        "reaction": canonical,
        "N": N, "reaction_reports": ni1,
        "min_events": int(min_events),
        "count": len(rows),
        "signals": {m: int(sum(1 for r in rows if r[m]["signal"])) for m in METHODS},
        "all_four": int(sum(1 for r in rows if r["signals"] == 4)),
        "rows": rows,
        "scope_only": _member is not None,
        "substances_excluded": n_all_subs - len(subs),
        "gps_priors": pri,
        "criteria": {
            "ror": "borne basse IC 95 % > 1",
            "prr": "PRR ≥ 2, Chi² de Yates ≥ 4 et n11 ≥ 3 (critères d'Evans)",
            "bcpnn": "borne basse IC 95 % de l'IC > 0 (soit 2^IC025 > 1)",
            "gps": "EB05 ≥ 2 (seuil FDA)",
        },
    }
    with engine._lock:
        engine._derived[key] = out
    return out


def _period_codes(engine, scope, code, population, period):
    """Code entier de la période de réception de chaque signalement (YYYYMM ou YYYY, 0 si inconnu).

    Tableau int32 mis en cache par périmètre : 4 octets par signalement au lieu de chaînes.
    """
    key = ("signal", "period_codes", period, str(scope).lower(), str(code).upper())
    with engine._lock:
        hit = engine._derived.get(key)
    if hit is not None:
        return hit
    used, out = "year", None
    if period == "month":
        if "month_id" in population:
            m = pd.to_numeric(population["month_id"], errors="coerce")
            if m.notna().mean() > 0.5 and m.dropna().between(190001, 210012).all():
                out, used = m.fillna(0).to_numpy(dtype=np.int32), "month"
        if out is None and "gateway_date" in population:
            raw = population["gateway_date"].astype("string").str.strip()
            d = pd.to_datetime(raw, errors="coerce", format="%d/%m/%Y %H:%M:%S")
            if d.notna().mean() < 0.5:
                d = pd.to_datetime(raw.str[:10], errors="coerce", format="%d/%m/%Y")
            if d.notna().mean() < 0.5:
                d = pd.to_datetime(raw, errors="coerce", format="ISO8601")
            del raw
            if d.notna().mean() > 0.5:
                out = (d.dt.year * 100 + d.dt.month).fillna(0).to_numpy(dtype=np.int32)
                used = "month"
            del d
    if out is None:
        out = pd.to_numeric(population["gateway_year"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        used = "year"
    res = (out, used)
    with engine._lock:
        engine._derived[key] = res
    return res


def reaction_signal_timeline(engine, scope, code, reaction, substance, period="month"):
    """Évolution cumulée (par mois ou par année) des 4 indicateurs pour une paire substance × réaction.

    Même logique que le LongitudinalModel de vigipy en mode cumulatif : à chaque période,
    tous les signalements reçus jusque-là. Un point n'est un signal que si n11 ≥ 4.
    """
    period = "year" if str(period).lower()[:1] in ("y", "a") else "month"
    key = ("signal", "timeline", period, str(scope).lower(), str(code).upper(), str(reaction).strip().casefold(), str(substance).strip().upper())
    with engine._lock:
        cached = engine._derived.get(key)
    if cached is not None:
        return cached

    scope, code, canonical, reaction_ids, population, pop_ids, inpop, totals = _reaction_context(engine, scope, code, reaction)
    substance = " ".join(str(substance).upper().split())
    sub_ids = inpop.loc[inpop["substance"] == substance, "safety_report_id"].unique()
    if len(sub_ids) == 0:
        raise ValueError(f"substance absente du périmètre : {substance}")

    codes, used = _period_codes(engine, scope, code, population, period)
    ids = population["safety_report_id"]
    rx = ids.isin(reaction_ids).to_numpy()
    sb = ids.isin(sub_ids).to_numpy()
    ok = codes > 0
    uniq, inv = np.unique(codes[ok], return_inverse=True)
    k = len(uniq)
    cN = np.cumsum(np.bincount(inv, minlength=k)).astype(np.float64)
    cR = np.cumsum(np.bincount(inv, weights=rx[ok], minlength=k))
    cS = np.cumsum(np.bincount(inv, weights=sb[ok], minlength=k))
    cB = np.cumsum(np.bincount(inv, weights=(rx & sb)[ok], minlength=k))
    del rx, sb, inv
    keep = cB >= 1
    if not keep.any():
        raise ValueError("aucun signalement commun à cette substance et cette réaction")
    uniq, cN, cR, cS, n11 = uniq[keep], cN[keep], cR[keep], cS[keep], cB[keep]

    pri = gps_priors(engine, _parent_l1(engine, scope, code))
    r = all_methods(n11, cS, cR, cN, pri["priors"])

    points = []
    for i, c in enumerate(uniq.tolist()):
        if used == "month":
            yy, mm = divmod(int(c), 100)
            lab = f"{yy:04d}-{mm:02d}"
        else:
            yy, mm, lab = int(c), 1, str(int(c))
        p = {"period": lab, "year": yy, "t": yy + (mm - 1) / 12.0,
             "n11": int(n11[i]), "n1j": int(cS[i]), "ni1": int(cR[i]), "N": int(cN[i])}
        for m in METHODS:
            p[m] = {kk: (bool(v[i] and n11[i] >= MIN_EVENTS) if kk == "signal" else _num(v[i])) for kk, v in r[m].items()}
        points.append(p)

    first = {m: next((p["period"] for p in points if p[m]["signal"]), None) for m in METHODS}
    out = {"scope": {"type": scope, "code": code}, "reaction": canonical, "substance": substance,
           "period": used, "points": points, "first_signal": first,
           "first_signal_year": {m: (int(v[:4]) if v else None) for m, v in first.items()},
           "min_events": MIN_EVENTS,
           "criteria": {"ror": "LB(IC 95 %) > 1", "prr": "PRR ≥ 2 et Chi² ≥ 4 et n11 ≥ 3",
                        "bcpnn": "LB(IC 95 %) de 2^IC > 1", "gps": "EB05 ≥ 2"},
           "note": "Cumul par date de réception. A priori GPS de la classe L1 parente."}
    with engine._lock:
        engine._derived[key] = out
    return out


def install(engine):
    import types
    engine.reaction_signals = types.MethodType(reaction_signals, engine)
    engine.reaction_signal_timeline = types.MethodType(reaction_signal_timeline, engine)
    engine.gps_priors = types.MethodType(gps_priors, engine)
    print("[signal] ROR / PRR / BCPNN / GPS disponibles", file=sys.stderr, flush=True)


# ======================================================================
# Routes HTTP : /api/reaction-signals et /api/reaction-signal-timeline
# ======================================================================

def install_routes(g):
    from http.server import BaseHTTPRequestHandler
    engine = g["RAW_STATS_ENGINE"]
    handlers = [v for v in g.values()
                if isinstance(v, type) and issubclass(v, BaseHTTPRequestHandler) and v is not BaseHTTPRequestHandler
                and "do_GET" in v.__dict__]
    if not handlers:
        print("[signal] aucun gestionnaire HTTP trouvé : routes non installées", file=sys.stderr, flush=True)
        return

    def send(h, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        h.send_response(status)
        h.send_header("Content-Type", "application/json; charset=utf-8")
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()
        h.wfile.write(body)

    for cls in handlers:
        original = cls.do_GET

        def do_GET(self, _original=original):
            u = urlparse(self.path)
            path = u.path.rstrip("/")
            if path not in ("/api/reaction-signals", "/api/reaction-signal-timeline"):
                return _original(self)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if path == "/api/reaction-signals":
                    out = engine.reaction_signals(q.get("scope", "l1"), q.get("code", ""), q.get("reaction", ""),
                                                  int(q.get("min_events", MIN_EVENTS)))
                else:
                    out = engine.reaction_signal_timeline(q.get("scope", "l1"), q.get("code", ""),
                                                          q.get("reaction", ""), q.get("substance", ""),
                                                          q.get("period", "month"))
                send(self, 200, out)
            except ValueError as e:
                send(self, 404, {"error": str(e)})
            except Exception as e:
                if type(e).__name__ == "RawStatsNotFound":
                    send(self, 404, {"error": str(e)})
                else:
                    print(f"[signal] erreur {path} : {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    send(self, 500, {"error": f"{type(e).__name__}: {e}"})

        cls.do_GET = do_GET
    print(f"[signal] routes installées sur {', '.join(c.__name__ for c in handlers)}", file=sys.stderr, flush=True)
