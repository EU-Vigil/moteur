"""Versions indexées de reaction_profile() et country_profile().

Même résultat JSON que raw_stats_engine (mêmes formules, mêmes seuils, même ordre),
mais au lieu de refiltrer tout le périmètre à chaque réaction ou pays, on construit
UNE fois par périmètre (L1, L2, substance) un index :

  - réaction (casefold) -> signalements concernés
  - signalement -> pays
  - totaux par réaction, par valeur de dimension, par substance active

Chaque fiche réaction / pays devient alors quelques comptages vectorisés.
En cas d'erreur inattendue, on retombe sur la méthode d'origine.

Installation (dans le serveur, AVANT stats_memo.install) :
    import fast_profiles
    fast_profiles.install(RAW_STATS_ENGINE)
"""

import math
import sys
import types

import numpy as np
import pandas as pd

RX_ICSR_COLS = [
    "safety_report_id", "gateway_date", "gateway_year", "month_id", "report_type",
    "age_group", "age_group_reported", "sex", "parent_child", "primary_source_qual",
    "regulatory_country", "eea", "literature", "seriousness",
]
CTY_ICSR_COLS = [
    "safety_report_id", "gateway_year", "sex", "age_group", "seriousness",
    "regulatory_country", "primary_source_qual",
]
ROR_DIMENSIONS = ["regulatory_country", "sex", "age_group", "primary_source_qual"]
ROR_INVALID = {"", "not specified", "not available", "n/a", "nan", "none"}
COUNTRY_INVALID = {"not specified", "not available", "n/a"}


# ------------------------------------------------------------------
# Extraction des noms (copie exacte du moteur)
# ------------------------------------------------------------------

def clean_drug_name(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    low = text.casefold()
    if low == "not reported" or low.startswith("not available") or low.startswith("[truncated"):
        return None
    if text.startswith("["):
        close = text.find("]")
        if close > 1:
            return text[1:close].strip()
    marker = " ("
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    return text or None


def extract_active_substances(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    low = text.casefold()
    if low == "not reported" or low.startswith("not available") or low.startswith("[truncated"):
        return []
    substances = []
    left = text.find("[")
    right = text.find("]", left + 1)
    if left != -1 and right != -1 and right > left + 1:
        raw = text[left + 1:right]
        substances = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        base = text.split(" (", 1)[0].strip()
        if base:
            substances = [x.strip() for x in base.split(",") if x.strip()]
    cleaned = []
    for name in substances:
        name = " ".join(name.upper().split())
        if name in {"", "N/A", "NOT AVAILABLE", "NOT REPORTED", "UNKNOWN"}:
            continue
        cleaned.append(name)
    return cleaned


def _ror(a, b, c, d):
    corrected = any(x == 0 for x in (a, b, c, d))
    if corrected:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    ror = aa * dd / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    low = math.exp(math.log(ror) - 1.96 * se)
    high = math.exp(math.log(ror) + 1.96 * se)
    return ror, low, high, corrected


def _ordered(keys):
    """Ordre de parcours identique à groupby() (clés triées)."""
    return sorted(keys)


# ------------------------------------------------------------------
# Index par périmètre
# ------------------------------------------------------------------

def _get(self, key, build):
    with self._lock:
        v = self._derived.get(key)
    if v is None:
        v = build()
        with self._lock:
            self._derived[key] = v
    return v


def _pairs(self, scope, code):
    """Paires (signalement, réaction) dédoublonnées, dans l'ordre du cache."""
    def build():
        reactions = self._load(scope, code, "reactions", ["safety_report_id", "reaction_pt"])
        pts = reactions["reaction_pt"].astype("string").str.strip()
        ok = (pts.notna() & (pts != "")).to_numpy(dtype=bool)
        df = pd.DataFrame({
            "id": reactions["safety_report_id"].to_numpy()[ok],
            "pt": pts.to_numpy()[ok],
            "raw": reactions["reaction_pt"].to_numpy()[ok],
        })
        df = df.drop_duplicates(["id", "pt"]).reset_index(drop=True)
        df["key"] = pd.Series(df["pt"], dtype="string").str.casefold().astype("category")
        df["pt"] = df["pt"].astype("category")
        return df
    return _get(self, ("fast", scope, code, "pairs"), build)


def _rx_index(self, scope, code):
    def build():
        pairs = _pairs(self, scope, code)
        ids = pairs["id"].to_numpy()
        raw = pairs["raw"].to_numpy()
        index = {}
        for key, pos in pairs.groupby("key", observed=True).indices.items():
            index[key] = (str(raw[pos.min()]), ids[pos])
        return index
    return _get(self, ("fast", scope, code, "rx_index"), build)


def _rx_population(self, scope, code):
    def build():
        icsr = self._load(scope, code, "icsr", RX_ICSR_COLS)
        population = icsr.drop_duplicates("safety_report_id")
        pop_years = population["gateway_year"].dropna().astype(int).value_counts().sort_index()
        dims = {}
        for dim in ROR_DIMENSIONS:
            values = population[dim].astype("string").str.strip()
            valid = (values.notna() & ~values.str.casefold().isin(ROR_INVALID)).to_numpy(dtype=bool)
            vals_valid = values[valid]
            dims[dim] = (valid, vals_valid, vals_valid.value_counts())
        pop_ids = population["safety_report_id"]
        all_population_ids = set(pop_ids.dropna().tolist())
        return population, pop_years, dims, all_population_ids
    return _get(self, ("fast", scope, code, "rx_population"), build)


def _all_active(self, scope, code):
    derived_key = (scope, code, "all_suspect_active_substances")

    def build():
        suspects = self._load(scope, code, "suspect_drugs", ["safety_report_id", "drug"])
        rows = []
        for row in suspects[["safety_report_id", "drug"]].itertuples(index=False):
            for substance in extract_active_substances(row.drug):
                rows.append((row.safety_report_id, substance))
        if rows:
            df = pd.DataFrame(rows, columns=["safety_report_id", "substance"])
            return df.drop_duplicates(["safety_report_id", "substance"])
        return pd.DataFrame(columns=["safety_report_id", "substance"])
    return _get(self, derived_key, build)


def _substance_totals(self, scope, code):
    def build():
        _, _, _, all_population_ids = _rx_population(self, scope, code)
        active = _all_active(self, scope, code)
        if active.empty:
            return active, pd.Series(dtype="int64")
        inpop = active[active["safety_report_id"].isin(all_population_ids)]
        totals = inpop.groupby("substance")["safety_report_id"].nunique()
        return inpop, totals
    return _get(self, ("fast", scope, code, "substance_totals"), build)


def _cty_index(self, scope, code):
    def build():
        icsr = self._load(scope, code, "icsr", CTY_ICSR_COLS).drop_duplicates("safety_report_id")
        countries = icsr["regulatory_country"].astype("string").str.strip()
        cf = countries.str.casefold()
        positions = {k: v for k, v in cf.groupby(cf, sort=False).indices.items()}
        valid = (countries.notna() & (countries != "") & ~cf.isin(COUNTRY_INVALID)).fillna(False).to_numpy(dtype=bool)
        ids = icsr["safety_report_id"].to_numpy()
        valid_ids = set(ids[valid].tolist())
        valid_cf = set(cf[valid].dropna().unique().tolist())

        pairs = _pairs(self, scope, code)
        id_to_cf = pd.Series(cf.to_numpy(), index=ids)
        id_to_cf = id_to_cf[~id_to_cf.index.duplicated()]
        pcf = pairs["id"].map(id_to_cf)
        pcf = pcf.astype("category")
        in_valid = pairs["id"].isin(valid_ids).to_numpy(dtype=bool)
        valid_totals = pairs.loc[in_valid, "pt"].value_counts()
        valid_totals = valid_totals[valid_totals > 0]
        return icsr, countries, positions, valid_ids, valid_cf, pcf, in_valid, valid_totals
    return _get(self, ("fast", scope, code, "cty_index"), build)


# ------------------------------------------------------------------
# reaction_profile
# ------------------------------------------------------------------

def fast_reaction_profile(self, scope, code, reaction, top_countries=15, top_ror=6):
    scope, code, _ = self._scope_path(scope, code)
    reaction = str(reaction).strip()
    if not reaction:
        raise ValueError("réaction absente")

    cache_key = (scope, code, reaction.casefold(), int(top_countries), int(top_ror))
    with self._lock:
        cached = self._reaction_profiles.get(cache_key)
    if cached is not None:
        return cached

    index = _rx_index(self, scope, code)
    hit = index.get(reaction.casefold())
    if hit is None:
        raise ValueError(f"réaction introuvable : {reaction}")
    canonical, rx_ids_arr = hit
    reaction_ids = set(pd.Series(rx_ids_arr).dropna().tolist())

    population, pop_years, dims, all_population_ids = _rx_population(self, scope, code)
    is_rx = population["safety_report_id"].isin(reaction_ids).to_numpy(dtype=bool)
    selected = population.loc[is_rx]

    total_reports = int(len(population))
    reaction_reports = int(len(selected))

    # Chronologie
    years = selected["gateway_year"].dropna().astype(int)
    year_counts = years.value_counts().sort_index()
    all_years = sorted(set(pop_years.index.tolist()) | set(year_counts.index.tolist()))
    timeline = []
    for year in all_years:
        reaction_n = int(year_counts.get(year, 0))
        population_n = int(pop_years.get(year, 0))
        timeline.append({
            "year": int(year),
            "reports": reaction_n,
            "population_reports": population_n,
            "share": reaction_n / population_n if population_n else 0,
        })
    first_year = int(years.min()) if len(years) else None
    last_year = int(years.max()) if len(years) else None

    # Pays
    country_counts = selected["regulatory_country"].fillna("Not Specified").replace("", "Not Specified").value_counts()
    countries = [
        {"value": str(v), "count": int(c), "share": float(c) / reaction_reports if reaction_reports else 0}
        for v, c in country_counts.head(int(top_countries)).items()
    ]

    profile = {
        "sex": self._distribution(selected["sex"], "Not Specified"),
        "age_group": self._distribution(selected["age_group"], "Not Specified"),
        "seriousness": self._distribution(selected["seriousness"], "Not Available"),
    }

    # ROR par dimension
    def ror_dimension(dim, minimum_reaction=5, minimum_group=100):
        valid, vals_valid, group_totals = dims[dim]
        total = int(len(vals_valid))
        if total == 0:
            return []
        rx_valid = is_rx[valid]
        total_reaction = int(rx_valid.sum())
        a_counts = vals_valid[rx_valid].value_counts()
        output = []
        for value in _ordered(group_totals.index.tolist()):
            group_total = int(group_totals[value])
            a = int(a_counts.get(value, 0))
            b = group_total - a
            comparison_total = total - group_total
            c = total_reaction - a
            d = comparison_total - c
            if group_total < minimum_group or a < minimum_reaction or comparison_total <= 0:
                continue
            ror, low, high, corrected = _ror(a, b, c, d)
            output.append({
                "value": str(value),
                "reaction_reports": a,
                "group_total": group_total,
                "group_share": a / group_total if group_total else 0,
                "comparison_reports": c,
                "comparison_total": comparison_total,
                "comparison_share": c / comparison_total if comparison_total else 0,
                "ror": ror,
                "ci95_low": low,
                "ci95_high": high,
                "corrected": corrected,
                "signal": a >= minimum_reaction and low > 1,
            })
        output = [x for x in output if x["ci95_low"] > 1]
        output.sort(key=lambda x: (x["ci95_low"], x["reaction_reports"]), reverse=True)
        return output[:int(top_ror)]

    ror_highlights = {dim: ror_dimension(dim) for dim in ROR_DIMENSIONS}

    # Médicaments suspects
    suspects = self._load(scope, code, "suspect_drugs", ["safety_report_id", "drug"])
    reaction_suspects = suspects[suspects["safety_report_id"].isin(reaction_ids)].copy()
    reaction_suspects["_drug_name"] = reaction_suspects["drug"].map(clean_drug_name)
    reaction_suspects = (
        reaction_suspects.dropna(subset=["_drug_name"])
        .drop_duplicates(["safety_report_id", "_drug_name"])
    )
    suspect_counts = reaction_suspects["_drug_name"].value_counts().head(20)
    suspect_drugs = [
        {"label": str(l), "reports": int(c), "share": int(c) / reaction_reports if reaction_reports else 0}
        for l, c in suspect_counts.items()
    ]

    active_rows = []
    for row in reaction_suspects[["safety_report_id", "drug"]].itertuples(index=False):
        for substance in extract_active_substances(row.drug):
            active_rows.append((row.safety_report_id, substance))
    if active_rows:
        active_df = pd.DataFrame(active_rows, columns=["safety_report_id", "substance"])
        active_df = active_df.drop_duplicates(["safety_report_id", "substance"])
        active_counts = active_df["substance"].value_counts().head(25)
        suspect_active_substances = [
            {"label": str(l), "reports": int(c), "share": int(c) / reaction_reports if reaction_reports else 0}
            for l, c in active_counts.items()
        ]
    else:
        suspect_active_substances = []

    # ROR par substance active suspecte
    suspect_ror = []
    inpop, totals = _substance_totals(self, scope, code)
    if not inpop.empty:
        total_population = len(all_population_ids)
        total_reaction = len(reaction_ids & all_population_ids)
        a_counts = inpop.loc[inpop["safety_report_id"].isin(reaction_ids), "substance"].value_counts()
        for substance in _ordered(totals.index.tolist()):
            group_total = int(totals[substance])
            if group_total < 100:
                continue
            a = int(a_counts.get(substance, 0))
            if a < 5:
                continue
            b = group_total - a
            other_total = total_population - group_total
            c = total_reaction - a
            d = other_total - c
            if other_total <= 0:
                continue
            ror, low, high, corrected = _ror(a, b, c, d)
            suspect_ror.append({
                "substance": str(substance),
                "reaction_reports": a,
                "substance_reports": group_total,
                "reaction_share": a / group_total if group_total else 0,
                "comparison_reports": c,
                "comparison_total": other_total,
                "comparison_share": c / other_total if other_total else 0,
                "ror": ror,
                "ci95_low": low,
                "ci95_high": high,
                "corrected": corrected,
                "signal": low > 1 and a >= 5,
            })
        suspect_ror.sort(key=lambda x: (x["ci95_low"], x["reaction_reports"]), reverse=True)

    result = {
        "scope": {"type": scope, "code": code},
        "reaction": canonical,
        "summary": {
            "population_reports": total_reports,
            "reaction_reports": reaction_reports,
            "share": reaction_reports / total_reports if total_reports else 0,
            "first_year": first_year,
            "last_year": last_year,
        },
        "timeline": timeline,
        "countries": countries,
        "profile": profile,
        "suspect_drugs": suspect_drugs,
        "suspect_active_substances": suspect_active_substances,
        "suspect_substance_ror": suspect_ror[:25],
        "ror_highlights": ror_highlights,
        "method": {
            "ror_min_reaction_reports": 5,
            "ror_min_group_reports": 100,
            "multiple_testing_correction": False,
            "exploratory": True,
        },
        "cache": {"hit": True, "type": "raw-parquet"},
    }
    with self._lock:
        self._reaction_profiles[cache_key] = result
    return result


# ------------------------------------------------------------------
# country_profile
# ------------------------------------------------------------------

def fast_country_profile(self, scope, code, country):
    scope, code, _ = self._scope_path(scope, code)
    country = str(country).strip()
    if not country:
        raise ValueError("pays absent")

    key = (scope, code, country.casefold())
    with self._lock:
        cached = self._country_profiles.get(key)
    if cached is not None:
        return cached

    icsr, countries, positions, valid_ids, valid_cf, pcf, in_valid, valid_totals = _cty_index(self, scope, code)
    cf_key = country.casefold()
    pos = positions.get(cf_key)
    if pos is None or len(pos) == 0:
        raise ValueError(f"pays introuvable : {country}")
    pos = np.sort(pos)
    canonical_country = str(countries.iloc[pos[0]])
    selected = icsr.iloc[pos]

    total_population = int(len(icsr))
    country_total = int(len(selected))

    timeline_counts = selected["gateway_year"].dropna().astype(int).value_counts().sort_index()
    timeline = [{"year": int(y), "reports": int(c)} for y, c in timeline_counts.items()]

    profile = {
        "sex": self._distribution(selected["sex"], "Not Specified"),
        "age_group": self._distribution(selected["age_group"], "Not Specified"),
        "seriousness": self._distribution(selected["seriousness"], "Not Available"),
        "primary_source_qual": self._distribution(selected["primary_source_qual"], "Not Specified"),
    }

    pairs = _pairs(self, scope, code)
    in_country = (pcf == cf_key).to_numpy(dtype=bool)
    reaction_counts = pairs.loc[in_country, "pt"].astype(str).value_counts()
    top_reactions = [{"label": str(l), "reports": int(c)} for l, c in reaction_counts.head(25).items()]

    ror_rows = []
    if cf_key in valid_cf:
        group_total = country_total
        other_total = len(valid_ids) - group_total
        a_counts = pairs.loc[in_country & in_valid, "pt"].value_counts()
        a_counts = a_counts[a_counts >= 5]
        if group_total >= 100 and other_total > 0:
            for rx in _ordered([str(x) for x in a_counts.index.tolist()]):
                a = int(a_counts[rx])
                b = group_total - a
                c = int(valid_totals[rx]) - a
                d = other_total - c
                ror, low, high, corrected = _ror(a, b, c, d)
                if low <= 1:
                    continue
                ror_rows.append({
                    "reaction": str(rx),
                    "reports": a,
                    "group_total": group_total,
                    "ror": ror,
                    "ci95_low": low,
                    "ci95_high": high,
                })
        ror_rows.sort(key=lambda x: (x["ci95_low"], x["reports"]), reverse=True)

    result = {
        "scope": {"type": scope, "code": code},
        "country": canonical_country,
        "summary": {
            "reports": country_total,
            "population_reports": total_population,
            "share": country_total / total_population if total_population else 0,
        },
        "timeline": timeline,
        "profile": profile,
        "top_reactions": top_reactions,
        "reaction_ror": ror_rows[:20],
        "cache": {"hit": True, "type": "raw-parquet"},
    }
    with self._lock:
        self._country_profiles[key] = result
    return result


# ------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------

def _with_fallback(fast, original, name):
    def wrapped(*args, **kwargs):
        try:
            return fast(*args, **kwargs)
        except ValueError:
            raise
        except Exception as e:
            if type(e).__name__ == "RawStatsNotFound":
                raise
            print(f"[fast] {name} : repli sur la version d'origine ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            return original(*args, **kwargs)
    wrapped.__name__ = name
    return wrapped


def install(engine):
    for name, fast in (("reaction_profile", fast_reaction_profile), ("country_profile", fast_country_profile)):
        original = getattr(engine, name)
        setattr(engine, name, _with_fallback(types.MethodType(fast, engine), original, name))
    print("[fast] reaction_profile et country_profile indexés", file=sys.stderr, flush=True)
