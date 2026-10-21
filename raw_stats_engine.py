# EUVIGIL_COMPACT_REPORT_IDS_V1
import numpy as np
from pathlib import Path
import math
import threading

import pandas as pd


ROOT = Path(
    "/root/ema/export/raw_stats_cache"
)


class RawStatsNotFound(Exception):
    pass


class RawStatsEngine:

    def __init__(self, root=ROOT):

        self.root = Path(root)

        self._lock = threading.RLock()

        # DataFrames conservés en RAM après
        # leur première utilisation.
        self._frames = {}

        # Fiches de réactions calculées à la demande.
        self._reaction_profiles = {}

        # DataFrames dérivés coûteux à reconstruire.
        self._derived = {}

        # Fiches géographiques calculées à la demande.
        self._country_profiles = {}


    # ========================================================
    # CHEMINS
    # ========================================================

    def _virtual_l2_substance_codes(
        self,
        code,
        parent_path
    ):
        """
        Retourne les substance_code appartenant au L2 demandé,
        en utilisant scope_substances du cache L1 parent.
        """

        code = str(code).strip().upper()

        key = (
            "virtual-l2-substance-codes",
            code
        )

        with self._lock:
            cached = self._derived.get(key)

        if cached is not None:
            return cached

        source = (
            parent_path
            / "scope_substances"
        )

        if not source.exists():
            raise RawStatsNotFound(
                f"scope_substances absent pour {code}"
            )

        df = pd.read_parquet(
            source,
            columns=[
                "substance_code",
                "atc_code",
            ]
        )

        atc = (
            df["atc_code"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
        )

        mask = (
            atc
            .str.slice(0, 3)
            .eq(code)
        )

        values = pd.to_numeric(
            df.loc[
                mask,
                "substance_code"
            ],
            errors="coerce"
        ).dropna()

        codes = frozenset(
            values
            .astype("int64")
            .tolist()
        )

        if not codes:
            raise RawStatsNotFound(
                f"sous-classe absente du cache : l2:{code}"
            )

        with self._lock:
            self._derived[key] = codes

        return codes


    def _virtual_l2_report_ids(
        self,
        code,
        parent_path
    ):
        """
        Signalements appartenant au L2 demandé.

        Ils sont déterminés par :
          L2 -> substance_code -> scope_links -> safety_report_id
        """

        code = str(code).strip().upper()

        key = (
            "virtual-l2-report-ids",
            code
        )

        with self._lock:
            cached = self._derived.get(key)

        if cached is not None:
            return cached

        substance_codes = (
            self._virtual_l2_substance_codes(
                code,
                parent_path
            )
        )

        source = (
            parent_path
            / "scope_links"
        )

        if not source.exists():
            raise RawStatsNotFound(
                f"scope_links absent pour {code}"
            )

        links = pd.read_parquet(
            source,
            columns=[
                "safety_report_id",
                "substance_code",
            ]
        )

        selected = links[
            links["substance_code"]
            .isin(substance_codes)
        ]

        values = pd.to_numeric(
            selected["safety_report_id"],
            errors="coerce"
        ).dropna()

        ids = np.unique(
            values.to_numpy(dtype=np.int64, copy=False)
        )

        if ids.size == 0:
            raise RawStatsNotFound(
                f"aucun signalement dans le cache pour l2:{code}"
            )

        with self._lock:
            self._derived[key] = ids

        return ids


    # EUVIGIL_FAST_SUBSTANCE_INDEX_V1
    def _virtual_substance_parent(self, code):
        """
        Recherche une substance dans un index global construit
        une seule fois depuis les caches scope_substances L1.
        """
        target = int(code)
        index_key = ("fast-substance-parent-index",)

        with self._lock:
            index = self._derived.get(index_key)

        if index is None:
            index = {}
            l1_root = self.root / "l1"

            if not l1_root.is_dir():
                raise RawStatsNotFound(
                    "aucun cache L1 disponible"
                )

            for parent_path in sorted(l1_root.iterdir()):
                if not parent_path.is_dir():
                    continue

                source_path = parent_path / "scope_substances"
                if not source_path.exists():
                    continue

                df = pd.read_parquet(
                    source_path,
                    columns=["substance_code"]
                )

                codes = pd.to_numeric(
                    df["substance_code"],
                    errors="coerce"
                ).dropna()

                parent = parent_path.name.strip().upper()

                for value in codes.unique():
                    index.setdefault(int(value), parent)

            with self._lock:
                existing = self._derived.get(index_key)

                if existing is None:
                    self._derived[index_key] = index
                else:
                    index = existing

        parent = index.get(target)

        if parent is None:
            raise RawStatsNotFound(
                f"substance absente des caches L1 : {code}"
            )

        return parent


    def _virtual_substance_report_ids(
        self,
        code,
        parent_path
    ):
        """
        Signalements contenant exactement la substance demandée.
        """

        code = str(int(code))
        target = int(code)

        key = (
            "virtual-substance-report-ids",
            code
        )

        with self._lock:
            cached = self._derived.get(key)

        if cached is not None:
            return cached

        source = (
            parent_path
            / "scope_links"
        )

        if not source.exists():
            raise RawStatsNotFound(
                f"scope_links absent pour substance:{code}"
            )

        links = pd.read_parquet(
            source,
            columns=[
                "safety_report_id",
                "substance_code",
            ]
        )

        selected = links[
            pd.to_numeric(
                links["substance_code"],
                errors="coerce"
            ).eq(target)
        ]

        values = pd.to_numeric(
            selected["safety_report_id"],
            errors="coerce"
        ).dropna()

        ids = np.unique(
            values.to_numpy(dtype=np.int64, copy=False)
        )

        if ids.size == 0:
            raise RawStatsNotFound(
                f"aucun signalement pour substance:{code}"
            )

        with self._lock:
            self._derived[key] = ids

        return ids


    # EU_VACCIN_VIRTUAL_J07_V1
    _VACCINE_GROUPS = frozenset({"COVID-19", "INFLUENZA", "AUTRES"})

    @staticmethod
    def _vaccine_group_for_name(name):
        text = str(name or "").strip().upper()
        if "COVID" in text:
            return "COVID-19"
        if "INFLUENZA" in text or "GRIPPE" in text:
            return "INFLUENZA"
        return "AUTRES"

    def _virtual_vaccine_substance_codes(self, code):
        code = str(code).strip().upper()
        key = ("virtual-vaccines-substance-codes", code)
        with self._lock:
            cached = self._derived.get(key)
        if cached is not None:
            return cached
        src = self.root / "l2" / "J07" / "scope_substances"
        if not src.is_dir():
            raise RawStatsNotFound("Cache J07/scope_substances absent")
        df = pd.read_parquet(src, columns=["substance_code", "substance_name"])
        group = df["substance_name"].map(self._vaccine_group_for_name)
        codes = frozenset(pd.to_numeric(
            df.loc[group.eq(code), "substance_code"], errors="coerce"
        ).dropna().astype("int64").tolist())
        if not codes:
            raise RawStatsNotFound(f"Groupe vaccinal vide : {code}")
        with self._lock:
            self._derived[key] = codes
        return codes

    def _virtual_vaccine_report_ids(self, code):
        code = str(code).strip().upper()
        key = ("virtual-vaccines-report-ids", code)
        with self._lock:
            cached = self._derived.get(key)
        if cached is not None:
            return cached
        path = self.root / "l2" / "J07" / "scope_links"
        if not path.is_dir():
            raise RawStatsNotFound("Cache J07/scope_links absent")
        codes = self._virtual_vaccine_substance_codes(code)
        links = pd.read_parquet(path, columns=["safety_report_id", "substance_code"])
        selected = links.loc[links["substance_code"].isin(codes), "safety_report_id"]
        ids = np.unique(
            pd.to_numeric(selected, errors="coerce")
            .dropna()
            .to_numpy(dtype=np.int64, copy=False)
        )
        if ids.size == 0:
            raise RawStatsNotFound(f"Aucun ICSR dans le groupe {code}")
        with self._lock:
            self._derived[key] = ids
        return ids

    def _scope_path(self, scope, code):

        scope = str(scope).strip().lower()

        if scope not in {
            "l1",
            "l2",
            "substance"
        }:
            raise ValueError(
                "scope invalide"
            )

        if scope in {"l1", "l2"}:
            code = (
                str(code)
                .strip()
                .upper()
            )
        else:
            code = str(int(code))

        # EUVIGIL_J07_MAIN_CLASS_ALIAS_V1
        # J07 : classe principale fonctionnelle,
        # données physiques et statistiques conservées en l2/J07.
        if scope == "l1" and code == "J07":
            scope = "l2"

        # ----------------------------------------------------
        # Cache physique exact
        # ----------------------------------------------------

        path = (
            self.root
            / scope
            / code
        )

        if path.is_dir():
            return scope, code, path

        # Groupes vaccinaux virtuels : fichiers J07 existants.
        if scope == "l2" and code in self._VACCINE_GROUPS:
            parent_path = self.root / "l2" / "J07"
            if not parent_path.is_dir():
                raise RawStatsNotFound("Cache l2:J07 absent")
            self._virtual_vaccine_substance_codes(code)
            return scope, code, parent_path

        # ----------------------------------------------------
        # L2 VIRTUEL :
        #
        # N02 n'a pas besoin d'un cache raw_stats_cache/l2/N02.
        # On réutilise raw_stats_cache/l1/N puis _load()
        # applique le sous-périmètre N02.
        # ----------------------------------------------------

        if scope == "l2":

            valid = (
                len(code) == 3
                and code[0].isalpha()
                and code[1:].isdigit()
            )

            if not valid:
                raise RawStatsNotFound(
                    f"code L2 invalide : {code}"
                )

            parent = code[0]

            parent_path = (
                self.root
                / "l1"
                / parent
            )

            if parent_path.is_dir():

                # Valide aussi que N02 existe réellement
                # dans les substances du cache N.
                self._virtual_l2_substance_codes(
                    code,
                    parent_path
                )

                return (
                    scope,
                    code,
                    parent_path
                )

        # ----------------------------------------------------
        # SUBSTANCE VIRTUELLE
        #
        # substance:18863 -> cache L1 parent, par exemple N.
        # Le filtrage exact sera effectué dans _load().
        # ----------------------------------------------------

        if scope == "substance":

            parent = (
                self._virtual_substance_parent(
                    code
                )
            )

            parent_path = (
                self.root
                / "l1"
                / parent
            )

            if parent_path.is_dir():

                # Vérifie que la substance possède
                # réellement des signalements.
                self._virtual_substance_report_ids(
                    code,
                    parent_path
                )

                return (
                    scope,
                    code,
                    parent_path
                )

        raise RawStatsNotFound(
            f"cache absent : {scope}:{code}"
        )


    # ========================================================
    # CHARGEMENT PANDAS LAZY
    # ========================================================

    def _load(
        self,
        scope,
        code,
        name,
        columns=None
    ):

        scope, code, path = (
            self._scope_path(
                scope,
                code
            )
        )

        key = (
            scope,
            code,
            name,
            tuple(columns or ())
        )

        with self._lock:

            if key in self._frames:
                return self._frames[key]


        # ----------------------------------------------------
        # Détection d'un L2 virtuel
        # ----------------------------------------------------

        virtual_l2 = (
            scope == "l2"
            and path
            == (
                self.root
                / "l1"
                / code[0]
            )
        )

        virtual_vaccine = (
            scope == "l2"
            and code in self._VACCINE_GROUPS
            and path == self.root / "l2" / "J07"
        )

        virtual_substance = (
            scope == "substance"
            and path.parent
            == (
                self.root
                / "l1"
            )
        )

        virtual_scope = (
            virtual_l2
            or virtual_substance
            or virtual_vaccine
        )


        requested_columns = (
            list(columns)
            if columns is not None
            else None
        )

        read_columns = (
            list(requested_columns)
            if requested_columns is not None
            else None
        )


        # ----------------------------------------------------
        # Colonne nécessaire au filtrage
        #
        # - datasets de définition du scope :
        #       filtrage par substance_code
        #
        # - datasets liés aux cas :
        #       filtrage par safety_report_id
        # ----------------------------------------------------

        filter_column = None

        if virtual_scope:

            if name in {
                "scope_links",
                "scope_substances",
                "substances",
            }:
                filter_column = (
                    "substance_code"
                )

            else:
                filter_column = (
                    "safety_report_id"
                )

            if (
                read_columns is not None
                and filter_column
                not in read_columns
            ):
                read_columns.append(
                    filter_column
                )


        source = (
            path
            / name
        )

        if not source.exists():
            raise RawStatsNotFound(
                f"jeu absent : "
                f"{scope}:{code}/{name}"
            )


        df = pd.read_parquet(
            source,
            columns=read_columns
        )


        # ----------------------------------------------------
        # APPLICATION DU SOUS-PERIMETRE L2
        # ----------------------------------------------------

        if virtual_scope:

            if filter_column not in df.columns:
                raise RawStatsNotFound(
                    f"impossible de filtrer "
                    f"{name} pour l2:{code} : "
                    f"{filter_column} absent"
                )

            if filter_column == "substance_code":

                if virtual_vaccine:
                    allowed = self._virtual_vaccine_substance_codes(code)
                elif virtual_l2:

                    allowed = (
                        self._virtual_l2_substance_codes(
                            code,
                            path
                        )
                    )

                else:

                    allowed = frozenset({
                        int(code)
                    })

            else:

                if virtual_vaccine:
                    allowed = self._virtual_vaccine_report_ids(code)
                elif virtual_l2:

                    allowed = (
                        self._virtual_l2_report_ids(
                            code,
                            path
                        )
                    )

                else:

                    allowed = (
                        self._virtual_substance_report_ids(
                            code,
                            path
                        )
                    )

            df = df[
                df[filter_column]
                .isin(allowed)
            ].copy()


            # La colonne ajoutée uniquement pour filtrer
            # n'est pas exposée au reste du moteur.
            if requested_columns is not None:
                df = df[
                    requested_columns
                ]


        with self._lock:
            self._frames[key] = df

        return df


    def available(self, scope, code):

        try:
            self._scope_path(
                scope,
                code
            )
            return True

        except (
            RawStatsNotFound,
            ValueError
        ):
            return False


    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _distribution(
        series,
        default=None
    ):

        if default is not None:
            series = (
                series
                .fillna(default)
                .replace("", default)
            )

        result = (
            series
            .value_counts(
                dropna=False
            )
        )

        items = []

        for value, count in result.items():

            if pd.isna(value):
                value = None

            items.append({
                "value": value,
                "count": int(count)
            })

        return items


    @staticmethod
    def _ranking(
        df,
        column,
        limit=15,
        ignore=None
    ):

        work = df[
            ["safety_report_id", column]
        ].copy()

        work[column] = (
            work[column]
            .astype("string")
            .str.strip()
        )

        work = work[
            work[column].notna()
            & (work[column] != "")
        ]

        if ignore:

            ignored = {
                str(x).strip().casefold()
                for x in ignore
            }

            work = work[
                ~work[column]
                .str.casefold()
                .isin(ignored)
            ]

        # Un libellé donné ne compte qu'une fois
        # par signalement.
        work = work.drop_duplicates(
            [
                "safety_report_id",
                column
            ]
        )

        counts = (
            work[column]
            .value_counts()
            .head(limit)
        )

        return [
            {
                "label": str(label),
                "reports": int(count)
            }
            for label, count
            in counts.items()
        ]


    # ========================================================
    # ANALYSE GENERALE
    # ========================================================

    def analyse(self, scope, code):

        scope, code, _ = (
            self._scope_path(
                scope,
                code
            )
        )

        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_date",
                "gateway_year",
                "month_id",
                "report_type",
                "age_group",
                "age_group_reported",
                "sex",
                "parent_child",
                "primary_source_qual",
                "regulatory_country",
                "eea",
                "literature",
                "seriousness",
            ]
        )

        links = self._load(
            scope,
            code,
            "scope_links",
            [
                "safety_report_id",
                "substance_code"
            ]
        )


        # ----------------------------------------------------
        # Résumé
        # ----------------------------------------------------

        total_reports = int(
            icsr["safety_report_id"]
            .nunique()
        )

        substance_count = int(
            links["substance_code"]
            .nunique()
        )


        years = (
            icsr["gateway_year"]
            .dropna()
        )

        if len(years):

            years_int = (
                years
                .astype(int)
            )

            first_year = int(
                years_int.min()
            )

            last_year = int(
                years_int.max()
            )

        else:

            first_year = None
            last_year = None


        # ----------------------------------------------------
        # Timeline
        # ----------------------------------------------------

        counts = (
            icsr["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        timeline = [
            {
                "year": int(year),
                "reports": int(count)
            }
            for year, count
            in counts.items()
        ]


        # ----------------------------------------------------
        # Profil historique attendu par analyse.html
        # ----------------------------------------------------

        profile = {

            "sex":
                self._distribution(
                    icsr["sex"],
                    "Not Specified"
                ),

            "age_group":
                self._distribution(
                    icsr["age_group"],
                    "Not Specified"
                ),

            "seriousness":
                self._distribution(
                    icsr["seriousness"],
                    "Not Available"
                ),
        }


        # ----------------------------------------------------
        # Toutes les dimensions supplémentaires.
        #
        # Elles ne sont pas encore forcément affichées dans
        # la page, mais l'API les expose déjà.
        # ----------------------------------------------------

        dimensions = {

            "report_type":
                self._distribution(
                    icsr["report_type"],
                    "Not Specified"
                ),

            "age_group_reported":
                self._distribution(
                    icsr["age_group_reported"],
                    "Not Specified"
                ),

            "parent_child":
                self._distribution(
                    icsr["parent_child"],
                    "Not Specified"
                ),

            "primary_source_qual":
                self._distribution(
                    icsr["primary_source_qual"],
                    "Not Specified"
                ),

            "regulatory_country":
                self._distribution(
                    icsr["regulatory_country"],
                    "Not Specified"
                ),

            "eea":
                self._distribution(
                    icsr["eea"],
                    "Not Specified"
                ),

            "literature":
                self._distribution(
                    icsr["literature"],
                    "Not Specified"
                ),
        }


        return {

            "scope": {
                "type": scope,
                "code": code
            },

            "summary": {
                "reports": total_reports,
                "substances":
                    substance_count,
                "first_year":
                    first_year,
                "last_year":
                    last_year
            },

            "timeline": timeline,

            "profile": profile,

            "dimensions":
                dimensions,

            "special_mentions":
                self.special_mentions(
                    scope,
                    code
                ),

            "cache": {
                "hit": True,
                "type": "raw-parquet"
            }
        }


    # ========================================================
    # RANKINGS
    # ========================================================

    def ranking(
        self,
        scope,
        code,
        kind,
        limit=15
    ):

        scope, code, _ = (
            self._scope_path(
                scope,
                code
            )
        )

        limit = max(
            1,
            min(
                int(limit),
                100
            )
        )


        if kind == "reactions":

            df = self._load(
                scope,
                code,
                "reactions",
                [
                    "safety_report_id",
                    "reaction_pt"
                ]
            )

            items = self._ranking(
                df,
                "reaction_pt",
                limit
            )


        elif kind == "suspect_drugs":

            df = self._load(
                scope,
                code,
                "suspect_drugs",
                [
                    "safety_report_id",
                    "drug"
                ]
            )

            items = self._ranking(
                df,
                "drug",
                limit,
                ignore={
                    "Not reported",
                    "Not available",
                    "[truncated, see ICSR form for full text]"
                }
            )


        elif kind == "concomitant_drugs":

            df = self._load(
                scope,
                code,
                "concomitant_drugs",
                [
                    "safety_report_id",
                    "drug"
                ]
            )

            items = self._ranking(
                df,
                "drug",
                limit,
                ignore={
                    "Not reported",
                    "Not available",
                    "[truncated, see ICSR form for full text]"
                }
            )


        else:

            raise ValueError(
                "kind invalide"
            )


        return {

            "scope": {
                "type": scope,
                "code": code
            },

            "kind": kind,

            "items": items,

            "pending": False,

            "cache": {
                "hit": True,
                "type": "raw-parquet"
            }
        }


    # ========================================================
    # DISPROPORTIONNALITE / ROR
    # ========================================================

    def ror(
        self,
        scope,
        code,
        reaction,
        dimension,
        value
    ):

        allowed_dimensions = {
            "regulatory_country",
            "sex",
            "age_group",
            "primary_source_qual",
            "eea",
            "literature",
            "report_type",
        }

        if dimension not in allowed_dimensions:
            raise ValueError(
                "dimension ROR invalide"
            )

        scope, code, _ = (
            self._scope_path(
                scope,
                code
            )
        )

        reaction = str(
            reaction
        ).strip()

        value = str(
            value
        ).strip()

        if not reaction:
            raise ValueError(
                "réaction absente"
            )

        if not value:
            raise ValueError(
                "valeur absente"
            )


        # ----------------------------------------------------
        # Population
        # ----------------------------------------------------

        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                dimension,
            ]
        )

        population = (
            icsr[
                [
                    "safety_report_id",
                    dimension,
                ]
            ]
            .drop_duplicates(
                "safety_report_id"
            )
            .copy()
        )


        # ----------------------------------------------------
        # Valeurs renseignées uniquement
        # ----------------------------------------------------

        values = (
            population[dimension]
            .astype("string")
            .str.strip()
        )

        invalid = {
            "",
            "not specified",
            "not available",
            "n/a",
            "nan",
            "none",
        }

        valid = (
            values.notna()
            &
            ~values
            .str.casefold()
            .isin(invalid)
        )

        population = (
            population
            .loc[valid]
            .copy()
        )

        population["_value"] = (
            population[dimension]
            .astype("string")
            .str.strip()
        )


        # ----------------------------------------------------
        # Signalements comportant exactement le PT MedDRA
        # ----------------------------------------------------

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt",
            ]
        )

        pts = (
            reactions["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        reaction_mask = (
            pts.notna()
            &
            (
                pts.str.casefold()
                ==
                reaction.casefold()
            )
        )

        matched = reactions.loc[
            reaction_mask,
            [
                "safety_report_id",
                "reaction_pt",
            ]
        ]

        if matched.empty:
            raise ValueError(
                f"réaction introuvable : {reaction}"
            )

        canonical_reaction = str(
            matched.iloc[0][
                "reaction_pt"
            ]
        )

        reaction_ids = set(
            matched[
                "safety_report_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # Groupe étudié / groupe comparateur
        # ----------------------------------------------------

        group_mask = (
            population["_value"]
            .str.casefold()
            ==
            value.casefold()
        )

        if not bool(
            group_mask.any()
        ):
            raise ValueError(
                f"valeur introuvable : {value}"
            )

        reaction_mask_pop = (
            population[
                "safety_report_id"
            ]
            .isin(reaction_ids)
        )


        #              réaction   pas réaction
        # groupe          a            b
        # autres          c            d

        a = int(
            (
                group_mask
                &
                reaction_mask_pop
            ).sum()
        )

        group_total = int(
            group_mask.sum()
        )

        b = (
            group_total
            - a
        )

        other_mask = ~group_mask

        c = int(
            (
                other_mask
                &
                reaction_mask_pop
            ).sum()
        )

        other_total = int(
            other_mask.sum()
        )

        d = (
            other_total
            - c
        )


        if (
            group_total == 0
            or other_total == 0
        ):
            raise ValueError(
                "comparaison impossible"
            )


        # ----------------------------------------------------
        # Haldane-Anscombe si cellule nulle
        # ----------------------------------------------------

        corrected = any(
            x == 0
            for x in (
                a,
                b,
                c,
                d
            )
        )

        if corrected:
            aa = a + 0.5
            bb = b + 0.5
            cc = c + 0.5
            dd = d + 0.5
        else:
            aa = float(a)
            bb = float(b)
            cc = float(c)
            dd = float(d)


        ror = (
            aa * dd
            /
            (
                bb * cc
            )
        )

        se = math.sqrt(
            1 / aa
            + 1 / bb
            + 1 / cc
            + 1 / dd
        )

        low = math.exp(
            math.log(ror)
            - 1.96 * se
        )

        high = math.exp(
            math.log(ror)
            + 1.96 * se
        )


        # ----------------------------------------------------
        # PRR, complément descriptif
        # ----------------------------------------------------

        p_group = (
            a / group_total
            if group_total
            else None
        )

        p_other = (
            c / other_total
            if other_total
            else None
        )

        prr = None

        if (
            p_group is not None
            and p_other
        ):
            prr = (
                p_group
                / p_other
            )


        return {

            "scope": {
                "type": scope,
                "code": code,
            },

            "reaction":
                canonical_reaction,

            "dimension":
                dimension,

            "value":
                value,

            "population": {
                "analysed":
                    int(
                        len(
                            population
                        )
                    ),

                "group":
                    group_total,

                "comparison":
                    other_total,
            },

            "table": {
                "a":
                    a,

                "b":
                    b,

                "c":
                    c,

                "d":
                    d,
            },

            "group": {
                "reaction_reports":
                    a,

                "total_reports":
                    group_total,

                "proportion":
                    (
                        a / group_total
                        if group_total
                        else 0
                    ),
            },

            "comparison": {
                "reaction_reports":
                    c,

                "total_reports":
                    other_total,

                "proportion":
                    (
                        c / other_total
                        if other_total
                        else 0
                    ),
            },

            "ror": {
                "value":
                    ror,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "corrected":
                    corrected,

                "disproportion":
                    (
                        a >= 5
                        and low > 1
                    ),
            },

            "prr":
                prr,

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


    # ========================================================
    # FICHE D'UNE REACTION
    # ========================================================

    def reaction_profile(
        self,
        scope,
        code,
        reaction,
        top_countries=15,
        top_ror=6
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        reaction = str(reaction).strip()

        if not reaction:
            raise ValueError(
                "réaction absente"
            )

        cache_key = (
            scope,
            code,
            reaction.casefold(),
            int(top_countries),
            int(top_ror)
        )

        with self._lock:
            cached = self._reaction_profiles.get(
                cache_key
            )

        if cached is not None:
            return cached


        # Même jeu de colonnes que analyse(),
        # afin de réutiliser le DataFrame déjà en RAM.
        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_date",
                "gateway_year",
                "month_id",
                "report_type",
                "age_group",
                "age_group_reported",
                "sex",
                "parent_child",
                "primary_source_qual",
                "regulatory_country",
                "eea",
                "literature",
                "seriousness",
            ]
        )

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt",
            ]
        )


        # ----------------------------------------------------
        # Identifier exactement le terme MedDRA
        # ----------------------------------------------------

        pts = (
            reactions["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        matched = reactions.loc[
            pts.notna()
            & (
                pts.str.casefold()
                ==
                reaction.casefold()
            ),
            [
                "safety_report_id",
                "reaction_pt"
            ]
        ].drop_duplicates(
            "safety_report_id"
        )

        if matched.empty:
            raise ValueError(
                f"réaction introuvable : {reaction}"
            )

        canonical = str(
            matched.iloc[0]["reaction_pt"]
        )

        reaction_ids = set(
            matched["safety_report_id"]
            .dropna()
            .tolist()
        )


        population = (
            icsr
            .drop_duplicates(
                "safety_report_id"
            )
            .copy()
        )

        population["_reaction"] = (
            population["safety_report_id"]
            .isin(reaction_ids)
        )

        selected = population.loc[
            population["_reaction"]
        ].copy()

        total_reports = int(
            len(population)
        )

        reaction_reports = int(
            len(selected)
        )


        # ----------------------------------------------------
        # CHRONOLOGIE
        # ----------------------------------------------------

        years = (
            selected["gateway_year"]
            .dropna()
            .astype(int)
        )

        year_counts = (
            years
            .value_counts()
            .sort_index()
        )

        # Nombre total de signalements du périmètre par année,
        # afin d'afficher aussi la proportion annuelle de la réaction.
        population_year_counts = (
            population["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        all_years = sorted(
            set(population_year_counts.index.tolist())
            | set(year_counts.index.tolist())
        )

        timeline = []

        for year in all_years:

            reaction_n = int(
                year_counts.get(year, 0)
            )

            population_n = int(
                population_year_counts.get(year, 0)
            )

            timeline.append({
                "year": int(year),
                "reports": reaction_n,
                "population_reports": population_n,
                "share": (
                    reaction_n / population_n
                    if population_n
                    else 0
                )
            })

        first_year = (
            int(years.min())
            if len(years)
            else None
        )

        last_year = (
            int(years.max())
            if len(years)
            else None
        )


        # ----------------------------------------------------
        # PAYS
        # ----------------------------------------------------

        country_series = (
            selected["regulatory_country"]
            .fillna("Not Specified")
            .replace("", "Not Specified")
        )

        country_counts = (
            country_series
            .value_counts()
        )

        countries = [
            {
                "value": str(value),
                "count": int(count),
                "share": (
                    float(count)
                    / reaction_reports
                    if reaction_reports
                    else 0
                )
            }
            for value, count
            in country_counts.head(
                int(top_countries)
            ).items()
        ]


        # ----------------------------------------------------
        # PROFIL DES CAS DE CETTE REACTION
        # ----------------------------------------------------

        profile = {
            "sex":
                self._distribution(
                    selected["sex"],
                    "Not Specified"
                ),

            "age_group":
                self._distribution(
                    selected["age_group"],
                    "Not Specified"
                ),

            "seriousness":
                self._distribution(
                    selected["seriousness"],
                    "Not Available"
                ),
        }


        # ----------------------------------------------------
        # CALCUL ROR POUR TOUTES LES VALEURS D'UNE DIMENSION
        # ----------------------------------------------------

        def ror_dimension(
            dimension,
            minimum_reaction=5,
            minimum_group=100
        ):

            values = (
                population[dimension]
                .astype("string")
                .str.strip()
            )

            invalid = {
                "",
                "not specified",
                "not available",
                "n/a",
                "nan",
                "none",
            }

            valid = (
                values.notna()
                &
                ~values.str.casefold().isin(
                    invalid
                )
            )

            work = (
                population.loc[
                    valid,
                    [
                        dimension,
                        "_reaction"
                    ]
                ]
                .copy()
            )

            work["_value"] = (
                work[dimension]
                .astype("string")
                .str.strip()
            )

            total = int(
                len(work)
            )

            total_reaction = int(
                work["_reaction"].sum()
            )

            if total == 0:
                return []

            grouped = (
                work
                .groupby(
                    "_value",
                    dropna=False
                )["_reaction"]
                .agg(
                    group_total="size",
                    reaction_reports="sum"
                )
            )

            output = []

            for value, row in grouped.iterrows():

                group_total = int(
                    row["group_total"]
                )

                a = int(
                    row["reaction_reports"]
                )

                b = group_total - a

                comparison_total = (
                    total - group_total
                )

                c = (
                    total_reaction - a
                )

                d = (
                    comparison_total - c
                )

                if (
                    group_total
                    < minimum_group
                    or
                    a
                    < minimum_reaction
                    or
                    comparison_total <= 0
                ):
                    continue


                corrected = any(
                    x == 0
                    for x in (
                        a, b, c, d
                    )
                )

                if corrected:
                    aa = a + 0.5
                    bb = b + 0.5
                    cc = c + 0.5
                    dd = d + 0.5
                else:
                    aa = float(a)
                    bb = float(b)
                    cc = float(c)
                    dd = float(d)


                ror = (
                    aa * dd
                    /
                    (bb * cc)
                )

                se = math.sqrt(
                    1 / aa
                    + 1 / bb
                    + 1 / cc
                    + 1 / dd
                )

                low = math.exp(
                    math.log(ror)
                    - 1.96 * se
                )

                high = math.exp(
                    math.log(ror)
                    + 1.96 * se
                )


                output.append({
                    "value":
                        str(value),

                    "reaction_reports":
                        a,

                    "group_total":
                        group_total,

                    "group_share":
                        (
                            a / group_total
                            if group_total
                            else 0
                        ),

                    "comparison_reports":
                        c,

                    "comparison_total":
                        comparison_total,

                    "comparison_share":
                        (
                            c / comparison_total
                            if comparison_total
                            else 0
                        ),

                    "ror":
                        ror,

                    "ci95_low":
                        low,

                    "ci95_high":
                        high,

                    "corrected":
                        corrected,

                    "signal":
                        (
                            a >= minimum_reaction
                            and low > 1
                        )
                })


            # Pour une fiche grand public :
            # uniquement disproportions dont IC95% > 1.
            output = [
                x
                for x in output
                if x["ci95_low"] > 1
            ]

            # La borne basse est plus intéressante que
            # le ROR brut pour éviter de favoriser les
            # toutes petites catégories.
            output.sort(
                key=lambda x: (
                    x["ci95_low"],
                    x["reaction_reports"]
                ),
                reverse=True
            )

            return output[
                :int(top_ror)
            ]


        ror_highlights = {
            "regulatory_country":
                ror_dimension(
                    "regulatory_country"
                ),

            "sex":
                ror_dimension(
                    "sex"
                ),

            "age_group":
                ror_dimension(
                    "age_group"
                ),

            "primary_source_qual":
                ror_dimension(
                    "primary_source_qual"
                ),
        }


        # ----------------------------------------------------
        # MEDICAMENTS SUSPECTS PRESENTS DANS LES DOSSIERS
        # CONTENANT CETTE REACTION
        # ----------------------------------------------------

        suspects = self._load(
            scope,
            code,
            "suspect_drugs",
            [
                "safety_report_id",
                "drug",
            ]
        )

        reaction_suspects = (
            suspects[
                suspects["safety_report_id"]
                .isin(reaction_ids)
            ]
            .copy()
        )

        # Normalisation légère pour le grand public :
        # on extrait prioritairement le nom situé entre crochets.
        # La chaîne brute reste disponible dans le cache source.
        def clean_drug_name(value):

            if pd.isna(value):
                return None

            text = str(value).strip()

            if not text:
                return None

            low = text.casefold()

            if (
                low == "not reported"
                or low.startswith("not available")
                or low.startswith(
                    "[truncated"
                )
            ):
                return None

            # Cas :
            # [ALPRAZOLAM] (S - ...)
            if text.startswith("["):
                close = text.find("]")

                if close > 1:
                    return text[
                        1:close
                    ].strip()

            # Cas :
            # OLANZAPINE [OLANZAPINE] (S - ...)
            marker = " ("

            if marker in text:
                text = text.split(
                    marker,
                    1
                )[0].strip()

            return text or None


        reaction_suspects[
            "_drug_name"
        ] = reaction_suspects[
            "drug"
        ].map(
            clean_drug_name
        )

        reaction_suspects = (
            reaction_suspects
            .dropna(
                subset=["_drug_name"]
            )
            .drop_duplicates(
                [
                    "safety_report_id",
                    "_drug_name",
                ]
            )
        )

        suspect_counts = (
            reaction_suspects[
                "_drug_name"
            ]
            .value_counts()
            .head(20)
        )

        suspect_drugs = [
            {
                "label": str(label),
                "reports": int(count),
                "share": (
                    int(count)
                    / reaction_reports
                    if reaction_reports
                    else 0
                )
            }
            for label, count
            in suspect_counts.items()
        ]


        # ----------------------------------------------------
        # SUBSTANCES ACTIVES SUSPECTES
        #
        # On repart de la chaîne brute :
        #
        # ZYPREXA [OLANZAPINE] (...)
        # -> OLANZAPINE
        #
        # [OXYCODONE, OXYCODONE HYDROCHLORIDE] (...)
        # -> OXYCODONE
        # -> OXYCODONE HYDROCHLORIDE
        #
        # Une substance ne compte qu'une fois par ICSR.
        # ----------------------------------------------------

        def extract_active_substances(value):

            if pd.isna(value):
                return []

            text = str(value).strip()

            if not text:
                return []

            low = text.casefold()

            if (
                low == "not reported"
                or low.startswith("not available")
                or low.startswith("[truncated")
            ):
                return []

            substances = []

            # Priorité au contenu entre crochets.
            left = text.find("[")
            right = text.find("]", left + 1)

            if (
                left != -1
                and right != -1
                and right > left + 1
            ):
                raw = text[
                    left + 1:right
                ]

                substances = [
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                ]

            else:
                # Sinon, on prend la partie avant les détails "(S - ...)".
                base = text.split(
                    " (",
                    1
                )[0].strip()

                if base:
                    substances = [
                        x.strip()
                        for x in base.split(",")
                        if x.strip()
                    ]

            cleaned = []

            for name in substances:

                name = " ".join(
                    name.upper().split()
                )

                if name in {
                    "",
                    "N/A",
                    "NOT AVAILABLE",
                    "NOT REPORTED",
                    "UNKNOWN",
                }:
                    continue

                cleaned.append(name)

            return cleaned


        active_rows = []

        for row in reaction_suspects[
            [
                "safety_report_id",
                "drug"
            ]
        ].itertuples(
            index=False
        ):

            for substance in extract_active_substances(
                row.drug
            ):

                active_rows.append(
                    (
                        row.safety_report_id,
                        substance
                    )
                )


        if active_rows:

            active_df = pd.DataFrame(
                active_rows,
                columns=[
                    "safety_report_id",
                    "substance"
                ]
            )

            active_df = (
                active_df
                .drop_duplicates(
                    [
                        "safety_report_id",
                        "substance"
                    ]
                )
            )

            active_counts = (
                active_df["substance"]
                .value_counts()
                .head(25)
            )

            suspect_active_substances = [
                {
                    "label": str(label),
                    "reports": int(count),
                    "share": (
                        int(count)
                        / reaction_reports
                        if reaction_reports
                        else 0
                    )
                }
                for label, count
                in active_counts.items()
            ]

        else:

            suspect_active_substances = []



        # ----------------------------------------------------
        # INDEX GLOBAL DES SUBSTANCES ACTIVES SUSPECTES
        #
        # IMPORTANT :
        # le dénominateur du ROR doit utiliser TOUS les
        # signalements du périmètre, pas uniquement ceux
        # contenant la réaction étudiée.
        #
        # Ce DataFrame est construit une fois puis gardé en RAM.
        # ----------------------------------------------------

        derived_key = (
            scope,
            code,
            "all_suspect_active_substances"
        )

        with self._lock:
            all_active_df = (
                self._derived.get(
                    derived_key
                )
            )

        if all_active_df is None:

            all_active_rows = []

            for row in suspects[
                [
                    "safety_report_id",
                    "drug"
                ]
            ].itertuples(
                index=False
            ):

                for substance in (
                    extract_active_substances(
                        row.drug
                    )
                ):

                    all_active_rows.append(
                        (
                            row.safety_report_id,
                            substance
                        )
                    )


            if all_active_rows:

                all_active_df = pd.DataFrame(
                    all_active_rows,
                    columns=[
                        "safety_report_id",
                        "substance"
                    ]
                )

                # Une substance donnée compte au maximum
                # une fois par signalement.
                all_active_df = (
                    all_active_df
                    .drop_duplicates(
                        [
                            "safety_report_id",
                            "substance"
                        ]
                    )
                )

            else:

                all_active_df = pd.DataFrame(
                    columns=[
                        "safety_report_id",
                        "substance"
                    ]
                )


            with self._lock:
                self._derived[
                    derived_key
                ] = all_active_df


        # ----------------------------------------------------
        # ROR PAR SUBSTANCE ACTIVE SUSPECTE
        #
        # Une substance compte au maximum une fois par ICSR.
        # Comparateur : tous les autres signalements du même
        # périmètre N.
        # ----------------------------------------------------

        suspect_ror = []

        if not all_active_df.empty:

            # all_active_df contient les substances suspectes
            # sur l'ensemble du périmètre.
            all_population_ids = set(
                population[
                    "safety_report_id"
                ]
                .dropna()
                .tolist()
            )

            total_population = len(
                all_population_ids
            )

            total_reaction = len(
                reaction_ids
                & all_population_ids
            )

            substance_groups = (
                all_active_df
                .groupby("substance")[
                    "safety_report_id"
                ]
                .agg(
                    lambda x: set(
                        x.dropna().tolist()
                    )
                )
            )

            for substance, ids in \
                substance_groups.items():

                ids = ids & all_population_ids

                group_total = len(ids)

                if group_total < 100:
                    continue

                a = len(
                    ids & reaction_ids
                )

                if a < 5:
                    continue

                b = group_total - a

                other_total = (
                    total_population
                    - group_total
                )

                c = (
                    total_reaction
                    - a
                )

                d = (
                    other_total
                    - c
                )

                if other_total <= 0:
                    continue

                corrected = any(
                    x == 0
                    for x in (
                        a, b, c, d
                    )
                )

                if corrected:
                    aa = a + 0.5
                    bb = b + 0.5
                    cc = c + 0.5
                    dd = d + 0.5
                else:
                    aa = float(a)
                    bb = float(b)
                    cc = float(c)
                    dd = float(d)

                ror = (
                    aa * dd
                    /
                    (bb * cc)
                )

                se = math.sqrt(
                    1 / aa
                    + 1 / bb
                    + 1 / cc
                    + 1 / dd
                )

                low = math.exp(
                    math.log(ror)
                    - 1.96 * se
                )

                high = math.exp(
                    math.log(ror)
                    + 1.96 * se
                )

                suspect_ror.append({
                    "substance":
                        str(substance),

                    "reaction_reports":
                        a,

                    "substance_reports":
                        group_total,

                    "reaction_share":
                        (
                            a / group_total
                            if group_total
                            else 0
                        ),

                    "comparison_reports":
                        c,

                    "comparison_total":
                        other_total,

                    "comparison_share":
                        (
                            c / other_total
                            if other_total
                            else 0
                        ),

                    "ror":
                        ror,

                    "ci95_low":
                        low,

                    "ci95_high":
                        high,

                    "corrected":
                        corrected,

                    "signal":
                        (
                            low > 1
                            and a >= 5
                        )
                })


            # Classement principal :
            # borne basse IC95 %, puis nombre de dossiers.
            suspect_ror.sort(
                key=lambda x: (
                    x["ci95_low"],
                    x["reaction_reports"]
                ),
                reverse=True
            )

        result = {

            "scope": {
                "type": scope,
                "code": code,
            },

            "reaction":
                canonical,

            "summary": {
                "population_reports":
                    total_reports,

                "reaction_reports":
                    reaction_reports,

                "share":
                    (
                        reaction_reports
                        / total_reports
                        if total_reports
                        else 0
                    ),

                "first_year":
                    first_year,

                "last_year":
                    last_year,
            },

            "timeline":
                timeline,

            "countries":
                countries,

            "profile":
                profile,

            "suspect_drugs":
                suspect_drugs,

            "suspect_active_substances":
                suspect_active_substances,

            "suspect_substance_ror":
                suspect_ror[:25],

            "ror_highlights":
                ror_highlights,

            "method": {
                "ror_min_reaction_reports":
                    5,

                "ror_min_group_reports":
                    100,

                "multiple_testing_correction":
                    False,

                "exploratory":
                    True,
            },

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


        with self._lock:
            self._reaction_profiles[
                cache_key
            ] = result

        return result


    # ========================================================
    # MENTIONS SPECIALES
    # ========================================================

    def special_mentions(self, scope, code):

        scope, code, path = self._scope_path(
            scope,
            code
        )

        key = (
            scope,
            code,
            "special_mentions"
        )

        with self._lock:
            cached = self._derived.get(key)

        if cached is not None:
            return cached


        # Lecture ponctuelle :
        # on ne conserve pas les énormes chaînes brutes en RAM.
        r = pd.read_parquet(
            path / "reactions",
            columns=[
                "safety_report_id",
                "reaction_pt",
                "reaction_details",
                "reaction_raw",
            ]
        )

        # ----------------------------------------------------
        # L2 VIRTUEL :
        # path pointe physiquement vers le cache L1 parent.
        # On restreint donc cette lecture ponctuelle aux seuls
        # signalements appartenant réellement à la sous-classe.
        # ----------------------------------------------------

        virtual_l2 = (
            scope == "l2"
            and path
            == (
                self.root
                / "l1"
                / code[0]
            )
        )

        virtual_substance = (
            scope == "substance"
            and path.parent
            == (
                self.root
                / "l1"
            )
        )

        if virtual_l2:

            allowed_ids = (
                self._virtual_l2_report_ids(
                    code,
                    path
                )
            )

        elif virtual_substance:

            allowed_ids = (
                self._virtual_substance_report_ids(
                    code,
                    path
                )
            )

            r = r[
                r["safety_report_id"]
                .isin(allowed_ids)
            ].copy()


        # ----------------------------------------------------
        # Death : PT MedDRA exact
        # ----------------------------------------------------

        pt = (
            r["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        death_mask = (
            pt.notna()
            &
            (
                pt.str.casefold()
                == "death"
            )
        )

        death_reports = int(
            r.loc[
                death_mask,
                "safety_report_id"
            ]
            .nunique()
        )


        # ----------------------------------------------------
        # Hospitalisation :
        # détection TEXTUELLE dans les chaînes réaction.
        #
        # "hospital" couvre notamment :
        # hospitalization / hospitalisation /
        # hospitalized / hospitalised...
        # ----------------------------------------------------

        hospital_mask = pd.Series(
            False,
            index=r.index
        )

        for column in (
            "reaction_pt",
            "reaction_details",
            "reaction_raw",
        ):

            hospital_mask |= (
                r[column]
                .astype("string")
                .str.contains(
                    "hospital",
                    case=False,
                    na=False,
                    regex=False
                )
            )


        hospital_reports = int(
            r.loc[
                hospital_mask,
                "safety_report_id"
            ]
            .nunique()
        )


        result = {
            "death_reports":
                death_reports,

            "hospitalisation_reports":
                hospital_reports,

            "hospitalisation_method":
                "text-search: hospital*",
        }


        with self._lock:
            self._derived[key] = result

        return result


    # ========================================================
    # FICHE PAYS
    # ========================================================

    def country_profile(
        self,
        scope,
        code,
        country
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        country = str(country).strip()

        if not country:
            raise ValueError(
                "pays absent"
            )

        key = (
            scope,
            code,
            country.casefold()
        )

        with self._lock:
            cached = self._country_profiles.get(
                key
            )

        if cached is not None:
            return cached


        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_year",
                "sex",
                "age_group",
                "seriousness",
                "regulatory_country",
                "primary_source_qual",
            ]
        ).drop_duplicates(
            "safety_report_id"
        )


        countries = (
            icsr["regulatory_country"]
            .astype("string")
            .str.strip()
        )

        mask = (
            countries.notna()
            &
            (
                countries.str.casefold()
                == country.casefold()
            )
        )

        if not bool(mask.any()):
            raise ValueError(
                f"pays introuvable : {country}"
            )

        canonical_country = str(
            countries.loc[mask].iloc[0]
        )

        selected = icsr.loc[
            mask
        ].copy()

        total_population = int(
            len(icsr)
        )

        country_total = int(
            len(selected)
        )


        # ----------------------------------------------------
        # Chronologie
        # ----------------------------------------------------

        timeline_counts = (
            selected["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        timeline = [
            {
                "year": int(year),
                "reports": int(count)
            }
            for year, count
            in timeline_counts.items()
        ]


        # ----------------------------------------------------
        # Profil
        # ----------------------------------------------------

        profile = {
            "sex":
                self._distribution(
                    selected["sex"],
                    "Not Specified"
                ),

            "age_group":
                self._distribution(
                    selected["age_group"],
                    "Not Specified"
                ),

            "seriousness":
                self._distribution(
                    selected["seriousness"],
                    "Not Available"
                ),

            "primary_source_qual":
                self._distribution(
                    selected["primary_source_qual"],
                    "Not Specified"
                ),
        }


        # ----------------------------------------------------
        # Réactions
        # ----------------------------------------------------

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt",
            ]
        )

        country_ids = set(
            selected["safety_report_id"]
            .tolist()
        )

        rp = reactions[
            reactions["safety_report_id"]
            .isin(country_ids)
        ].copy()

        rp["reaction_pt"] = (
            rp["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        rp = rp[
            rp["reaction_pt"].notna()
            & (rp["reaction_pt"] != "")
        ].drop_duplicates(
            [
                "safety_report_id",
                "reaction_pt"
            ]
        )

        reaction_counts = (
            rp["reaction_pt"]
            .value_counts()
        )

        top_reactions = [
            {
                "label": str(label),
                "reports": int(count)
            }
            for label, count
            in reaction_counts.head(25).items()
        ]


        # ----------------------------------------------------
        # Réactions disproportionnées dans ce pays
        # comparé aux autres pays renseignés
        # ----------------------------------------------------

        valid_country = (
            countries.notna()
            &
            (countries != "")
            &
            ~countries.str.casefold().isin(
                {
                    "not specified",
                    "not available",
                    "n/a",
                }
            )
        )

        valid_population = icsr.loc[
            valid_country
        ]

        valid_ids = set(
            valid_population[
                "safety_report_id"
            ].tolist()
        )

        group_ids = (
            country_ids
            & valid_ids
        )

        other_ids = (
            valid_ids
            - group_ids
        )

        pairs = reactions[
            reactions["safety_report_id"]
            .isin(valid_ids)
        ][
            [
                "safety_report_id",
                "reaction_pt"
            ]
        ].copy()

        pairs["reaction_pt"] = (
            pairs["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        pairs = pairs[
            pairs["reaction_pt"].notna()
            & (pairs["reaction_pt"] != "")
        ].drop_duplicates(
            [
                "safety_report_id",
                "reaction_pt"
            ]
        )


        ror_rows = []

        for reaction, block in pairs.groupby(
            "reaction_pt"
        ):

            ids = set(
                block["safety_report_id"]
                .tolist()
            )

            a = len(ids & group_ids)

            if a < 5:
                continue

            b = len(group_ids) - a

            c = len(ids & other_ids)
            d = len(other_ids) - c

            if (
                len(group_ids) < 100
                or
                len(other_ids) <= 0
            ):
                continue

            corrected = any(
                x == 0
                for x in (a,b,c,d)
            )

            if corrected:
                aa=a+0.5
                bb=b+0.5
                cc=c+0.5
                dd=d+0.5
            else:
                aa=float(a)
                bb=float(b)
                cc=float(c)
                dd=float(d)

            ror = (
                aa * dd
                /
                (bb * cc)
            )

            se = math.sqrt(
                1/aa
                + 1/bb
                + 1/cc
                + 1/dd
            )

            low = math.exp(
                math.log(ror)
                - 1.96*se
            )

            high = math.exp(
                math.log(ror)
                + 1.96*se
            )

            if low <= 1:
                continue

            ror_rows.append({
                "reaction":
                    str(reaction),

                "reports":
                    a,

                "group_total":
                    len(group_ids),

                "ror":
                    ror,

                "ci95_low":
                    low,

                "ci95_high":
                    high,
            })


        ror_rows.sort(
            key=lambda x: (
                x["ci95_low"],
                x["reports"]
            ),
            reverse=True
        )


        result = {
            "scope": {
                "type": scope,
                "code": code,
            },

            "country":
                canonical_country,

            "summary": {
                "reports":
                    country_total,

                "population_reports":
                    total_population,

                "share":
                    (
                        country_total
                        / total_population
                        if total_population
                        else 0
                    ),
            },

            "timeline":
                timeline,

            "profile":
                profile,

            "top_reactions":
                top_reactions,

            "reaction_ror":
                ror_rows[:20],

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


        with self._lock:
            self._country_profiles[
                key
            ] = result

        return result


    # ========================================================
    # FICHE HOSPITALISATION
    #
    # Détection textuelle : "hospital*" dans
    # reaction_pt / reaction_details / reaction_raw
    # ========================================================

    def hospitalisation_profile(
        self,
        scope,
        code
    ):

        scope, code, path = self._scope_path(
            scope,
            code
        )

        key = (
            scope,
            code,
            "hospitalisation_profile"
        )

        with self._lock:
            cached = self._derived.get(key)

        if cached is not None:
            return cached


        # ----------------------------------------------------
        # Identifier les signalements comportant hospital*
        # ----------------------------------------------------

        reactions_raw = pd.read_parquet(
            path / "reactions",
            columns=[
                "safety_report_id",
                "reaction_pt",
                "reaction_details",
                "reaction_raw",
            ]
        )

        virtual_l2 = (
            scope == "l2"
            and path
            == (
                self.root
                / "l1"
                / code[0]
            )
        )

        virtual_substance = (
            scope == "substance"
            and path.parent
            == (
                self.root
                / "l1"
            )
        )

        if virtual_l2:

            allowed_ids = (
                self._virtual_l2_report_ids(
                    code,
                    path
                )
            )

        elif virtual_substance:

            allowed_ids = (
                self._virtual_substance_report_ids(
                    code,
                    path
                )
            )

            reactions_raw = reactions_raw[
                reactions_raw["safety_report_id"]
                .isin(allowed_ids)
            ].copy()

        mask = pd.Series(
            False,
            index=reactions_raw.index
        )

        for column in (
            "reaction_pt",
            "reaction_details",
            "reaction_raw",
        ):

            mask |= (
                reactions_raw[column]
                .astype("string")
                .str.contains(
                    "hospital",
                    case=False,
                    na=False,
                    regex=False
                )
            )

        hospital_ids = set(
            reactions_raw.loc[
                mask,
                "safety_report_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # Population
        # ----------------------------------------------------

        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_year",
                "sex",
                "age_group",
                "seriousness",
                "regulatory_country",
                "primary_source_qual",
            ]
        ).drop_duplicates(
            "safety_report_id"
        )

        selected = icsr[
            icsr["safety_report_id"]
            .isin(hospital_ids)
        ].copy()

        total_reports = int(
            len(icsr)
        )

        hospital_reports = int(
            len(selected)
        )


        # ----------------------------------------------------
        # Chronologie
        # ----------------------------------------------------

        timeline_counts = (
            selected["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        population_year_counts = (
            icsr["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        years = sorted(
            set(
                population_year_counts.index.tolist()
            )
            |
            set(
                timeline_counts.index.tolist()
            )
        )

        timeline = []

        for year in years:

            n = int(
                timeline_counts.get(
                    year,
                    0
                )
            )

            total = int(
                population_year_counts.get(
                    year,
                    0
                )
            )

            timeline.append({
                "year":
                    int(year),

                "reports":
                    n,

                "population_reports":
                    total,

                "share":
                    (
                        n / total
                        if total
                        else 0
                    )
            })


        # ----------------------------------------------------
        # Pays
        # ----------------------------------------------------

        country_counts = (
            selected[
                "regulatory_country"
            ]
            .fillna("Not Specified")
            .replace(
                "",
                "Not Specified"
            )
            .value_counts()
        )

        countries = [
            {
                "value":
                    str(value),

                "count":
                    int(count),

                "share":
                    (
                        int(count)
                        / hospital_reports
                        if hospital_reports
                        else 0
                    )
            }
            for value, count
            in country_counts.head(20).items()
        ]


        # ----------------------------------------------------
        # Profil
        # ----------------------------------------------------

        profile = {
            "sex":
                self._distribution(
                    selected["sex"],
                    "Not Specified"
                ),

            "age_group":
                self._distribution(
                    selected["age_group"],
                    "Not Specified"
                ),

            "seriousness":
                self._distribution(
                    selected["seriousness"],
                    "Not Available"
                ),

            "primary_source_qual":
                self._distribution(
                    selected[
                        "primary_source_qual"
                    ],
                    "Not Specified"
                ),
        }


        # ----------------------------------------------------
        # Termes MedDRA les plus fréquents dans ces dossiers
        # ----------------------------------------------------

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt"
            ]
        )

        r = reactions[
            reactions["safety_report_id"]
            .isin(hospital_ids)
        ].copy()

        r["reaction_pt"] = (
            r["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        r = r[
            r["reaction_pt"].notna()
            & (r["reaction_pt"] != "")
        ].drop_duplicates(
            [
                "safety_report_id",
                "reaction_pt"
            ]
        )

        top_reactions = [
            {
                "label":
                    str(label),

                "reports":
                    int(count)
            }
            for label, count
            in (
                r["reaction_pt"]
                .value_counts()
                .head(25)
                .items()
            )
        ]


        result = {

            "scope": {
                "type": scope,
                "code": code,
            },

            "summary": {
                "population_reports":
                    total_reports,

                "hospitalisation_reports":
                    hospital_reports,

                "share":
                    (
                        hospital_reports
                        / total_reports
                        if total_reports
                        else 0
                    ),
            },

            "timeline":
                timeline,

            "countries":
                countries,

            "profile":
                profile,

            "top_reactions":
                top_reactions,

            "method": {
                "type":
                    "text-search",

                "pattern":
                    "hospital*",

                "fields": [
                    "reaction_pt",
                    "reaction_details",
                    "reaction_raw",
                ]
            },

            "cache": {
                "hit": True,
                "type": "raw-parquet"
            }
        }


        with self._lock:
            self._derived[key] = result

        return result


    # ========================================================
    # OPTIONS DE COMPARAISON
    # ========================================================

    def comparison_options(
        self,
        scope,
        code
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        links = self._load(
            scope,
            code,
            "scope_links",
            [
                "safety_report_id",
                "substance_code",
            ]
        )

        substances = self._load(
            scope,
            code,
            "scope_substances",
            [
                "substance_code",
                "substance_name",
                "atc_code",
                "atc_l2",
                "atc_name",
            ]
        ).copy()


        # Seulement les substances réellement présentes
        # dans les signalements du cache.
        active_codes = set(
            links["substance_code"]
            .dropna()
            .unique()
            .tolist()
        )

        substances = substances[
            substances["substance_code"]
            .isin(active_codes)
        ].copy()


        # ----------------------------------------------------
        # SUBSTANCES
        # ----------------------------------------------------

        substance_counts = (
            links
            .groupby("substance_code")[
                "safety_report_id"
            ]
            .nunique()
            .to_dict()
        )

        substance_options = []

        for row in substances.itertuples(
            index=False
        ):

            name = (
                str(row.substance_name).strip()
                if row.substance_name is not None
                else str(row.substance_code)
            )

            substance_options.append({
                "code":
                    str(int(row.substance_code)),

                "label":
                    name,

                "reports":
                    int(
                        substance_counts.get(
                            row.substance_code,
                            0
                        )
                    ),
            })

        substance_options.sort(
            key=lambda x: (
                x["label"].casefold(),
                x["code"]
            )
        )


        # ----------------------------------------------------
        # SOUS-CLASSES ATC L2
        # ----------------------------------------------------

        substances["_l2_code"] = (
            substances["atc_code"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str[:3]
        )

        valid_l2 = substances[
            substances["_l2_code"]
            .str.match(
                r"^[A-Z][0-9][0-9]$",
                na=False
            )
        ].copy()

        l2_options = []

        for l2_code, block in valid_l2.groupby(
            "_l2_code"
        ):

            codes = set(
                block["substance_code"]
                .dropna()
                .tolist()
            )

            ids = links[
                links["substance_code"]
                .isin(codes)
            ][
                "safety_report_id"
            ].nunique()


            labels = []

            if "atc_l2" in block.columns:

                labels = [
                    str(x).strip()
                    for x in (
                        block["atc_l2"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    if str(x).strip()
                    and str(x).strip().upper()
                        != str(l2_code).upper()
                ]


            label = (
                labels[0]
                if labels
                else str(l2_code)
            )

            l2_options.append({
                "code":
                    str(l2_code),

                "label":
                    label,

                "reports":
                    int(ids),
            })



        # EUVIGIL_J07_COMPARE_OPTIONS_V2
        # Les groupes sont virtuels et ne sont pas des codes ATC.
        if scope == "l2" and code == "J07":
            vaccine_labels = {
                "COVID-19": "Vaccins COVID-19",
                "INFLUENZA": "Vaccins contre la grippe",
                "AUTRES": "Autres vaccins",
            }

            l2_options = []

            for group_code, group_label in vaccine_labels.items():
                group_codes = self._virtual_vaccine_substance_codes(
                    group_code
                )

                reports = links.loc[
                    links["substance_code"].isin(group_codes),
                    "safety_report_id"
                ].nunique()

                if reports:
                    l2_options.append({
                        "code": group_code,
                        "label": group_label,
                        "reports": int(reports),
                    })

        l2_options.sort(
            key=lambda x: x["code"]
        )


        return {
            "scope": {
                "type": scope,
                "code": code,
            },

            "l2":
                l2_options,

            "substances":
                substance_options,

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


    # ========================================================
    # ANALYSE COMPARATIVE
    # ========================================================

        # EUVIGIL_COMPARE_FAST_LOOPS_V1
    def compare(
        self,
        scope,
        code,
        target_scope,
        target_code
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        target_scope = (
            str(target_scope)
            .strip()
            .lower()
        )

        target_code = (
            str(target_code)
            .strip()
        )

        if target_scope not in {
            "l2",
            "substance"
        }:
            raise ValueError(
                "target_scope doit être "
                "l2 ou substance"
            )


        cache_key = (
            scope,
            code,
            "compare",
            target_scope,
            target_code.upper()
        )

        # EUVIGIL_COMPARE_NO_RESULT_RAM_CACHE_V1
        # Résultat complet conservé uniquement dans le cache JSON disque.


        # ----------------------------------------------------
        # DONNEES DE BASE
        # ----------------------------------------------------

        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_year",
                "sex",
                "age_group",
                "seriousness",
                "regulatory_country",
                "primary_source_qual",
                "eea",
                "literature",
            ]
        ).drop_duplicates(
            "safety_report_id"
        ).copy()


        links = None

        substances = self._load(
            scope,
            code,
            "scope_substances",
            [
                "substance_code",
                "substance_name",
                "atc_code",
                "atc_l2",
                "atc_name",
            ]
        ).copy()


        population_ids = set(
            icsr["safety_report_id"]
            .dropna()
            .tolist()
        )


        # ----------------------------------------------------
        # DEFINIR LA CIBLE
        # ----------------------------------------------------

        if target_scope == "substance":

            try:
                target_int = int(
                    target_code
                )
            except Exception:
                raise ValueError(
                    "code substance invalide"
                )

            target_substance_codes = {
                target_int
            }

            meta = substances[
                substances["substance_code"]
                == target_int
            ]

            if meta.empty:
                raise ValueError(
                    "substance absente du périmètre"
                )

            target_label = str(
                meta.iloc[0][
                    "substance_name"
                ]
            )

            target_code_norm = str(
                target_int
            )


        else:

            target_code_norm = (
                target_code.upper()
            )

            atc = (
                substances["atc_code"]
                .astype("string")
                .str.strip()
                .str.upper()
            )

            # EUVIGIL_J07_VIRTUAL_TARGET_V2
            if (
                scope == "l2"
                and code == "J07"
                and target_code_norm in self._VACCINE_GROUPS
            ):
                vaccine_codes = self._virtual_vaccine_substance_codes(
                    target_code_norm
                )
                meta = substances[
                    substances["substance_code"].isin(vaccine_codes)
                ]
            else:
                meta = substances[
                    atc.str.startswith(
                        target_code_norm,
                        na=False
                    )
                ]

            if meta.empty:
                raise ValueError(
                    "sous-classe ATC absente "
                    "du périmètre"
                )

            target_substance_codes = set(
                meta[
                    "substance_code"
                ]
                .dropna()
                .tolist()
            )


            labels = []

            if "atc_l2" in meta.columns:

                labels = [
                    str(x).strip()
                    for x in (
                        meta["atc_l2"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    if str(x).strip()
                    and str(x).strip().upper()
                       != target_code_norm
                ]

            target_label = (
                labels[0]
                if labels
                else target_code_norm
            )




        # EUVIGIL_J07_TARGET_LABEL_V2
        if (
            scope == "l2"
            and code == "J07"
            and target_scope == "l2"
            and target_code_norm in self._VACCINE_GROUPS
        ):
            target_label = {
                "COVID-19": "Vaccins COVID-19",
                "INFLUENZA": "Vaccins contre la grippe",
                "AUTRES": "Autres vaccins",
            }[target_code_norm]

        # EUVIGIL_COMPARE_L2_IDS_V1
        # Voie rapide validée pour A -> A01 uniquement.
        target_ids = None

        # EUVIGIL_COMPARE_L2_GENERAL_V1
        validated_file = (
            self.root / "_compare_l2_validated.txt"
        )

        validated_l2 = set()

        try:
            validated_l2 = {
                line.strip().upper()
                for line in validated_file.read_text().splitlines()
                if line.strip()
            }
        except OSError:
            pass

        if (
            scope == "l1"
            and target_scope == "l2"
            and target_code_norm in validated_l2
            and target_code_norm.startswith(code)
        ):
            try:
                l2_path = (
                    self.root / "l2" / target_code_norm / "icsr"
                )

                if l2_path.exists():
                    l2_frame = pd.read_parquet(
                        l2_path,
                        columns=["safety_report_id"]
                    )

                    target_ids = set(
                        l2_frame["safety_report_id"]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    target_ids &= population_ids

            except Exception:
                target_ids = None

        if target_ids is None:
            links = self._load(
                scope,
                code,
                "scope_links",
                [
                    "safety_report_id",
                    "substance_code",
                ]
            )

            target_ids = set(
                links[
                    links["substance_code"].isin(
                        target_substance_codes
                    )
                ]["safety_report_id"]
                .dropna()
                .unique()
                .tolist()
            )

            target_ids &= population_ids

        comparison_ids = (
            population_ids
            - target_ids
        )


        if not target_ids:
            raise ValueError(
                "aucun signalement "
                "dans le groupe cible"
            )

        if not comparison_ids:
            raise ValueError(
                "aucun signalement "
                "dans le groupe comparateur"
            )


        icsr["_group"] = (
            icsr["safety_report_id"]
            .isin(target_ids)
        )

        target_df = icsr[
            icsr["_group"]
        ].copy()

        comparison_df = icsr[
            ~icsr["_group"]
        ].copy()


        target_total = int(
            len(target_df)
        )

        comparison_total = int(
            len(comparison_df)
        )


        # ----------------------------------------------------
        # DISTRIBUTIONS
        # ----------------------------------------------------

        def dist(df, field, default):

            values = (
                df[field]
                .fillna(default)
                .replace("", default)
                .value_counts()
            )

            total = len(df)

            return [
                {
                    "value":
                        (
                            None
                            if pd.isna(value)
                            else str(value)
                        ),

                    "count":
                        int(count),

                    "share":
                        (
                            int(count) / total
                            if total
                            else 0
                        )
                }
                for value, count
                in values.items()
            ]


        profile = {}

        for field, default in (
            (
                "sex",
                "Not Specified"
            ),
            (
                "age_group",
                "Not Specified"
            ),
            (
                "seriousness",
                "Not Available"
            ),
            (
                "primary_source_qual",
                "Not Specified"
            ),
            (
                "eea",
                "Not Specified"
            ),
            (
                "literature",
                "Not Specified"
            ),
        ):

            profile[field] = {
                "target":
                    dist(
                        target_df,
                        field,
                        default
                    ),

                "comparison":
                    dist(
                        comparison_df,
                        field,
                        default
                    ),
            }


        # ----------------------------------------------------
        # CHRONOLOGIE
        # ----------------------------------------------------

        ty = (
            target_df["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
        )

        cy = (
            comparison_df["gateway_year"]
            .dropna()
            .astype(int)
            .value_counts()
        )

        years = sorted(
            set(ty.index.tolist())
            |
            set(cy.index.tolist())
        )

        timeline = []

        for year in years:

            a = int(
                ty.get(year, 0)
            )

            b = int(
                cy.get(year, 0)
            )

            total = a + b

            timeline.append({
                "year":
                    int(year),

                "target":
                    a,

                "comparison":
                    b,

                "target_share":
                    (
                        a / total
                        if total
                        else 0
                    )
            })


        # ----------------------------------------------------
        # FONCTION 2x2 / OR
        # ----------------------------------------------------

        def odds_ratio(a, b, c, d):

            corrected = any(
                x == 0
                for x in (
                    a, b, c, d
                )
            )

            if corrected:
                aa = a + 0.5
                bb = b + 0.5
                cc = c + 0.5
                dd = d + 0.5
            else:
                aa = float(a)
                bb = float(b)
                cc = float(c)
                dd = float(d)

            value = (
                aa * dd
                /
                (bb * cc)
            )

            se = math.sqrt(
                1 / aa
                + 1 / bb
                + 1 / cc
                + 1 / dd
            )

            low = math.exp(
                math.log(value)
                - 1.96 * se
            )

            high = math.exp(
                math.log(value)
                + 1.96 * se
            )

            return (
                value,
                low,
                high,
                corrected
            )


        # ----------------------------------------------------
        # OR GRAVITE
        #
        # On compare uniquement Serious / Non-Serious :
        # "Not Available" est exclu.
        # ----------------------------------------------------

        target_serious = int(
            (
                target_df[
                    "seriousness"
                ]
                == "Serious"
            ).sum()
        )

        target_nonserious = int(
            (
                target_df[
                    "seriousness"
                ]
                == "Non-Serious"
            ).sum()
        )

        comparison_serious = int(
            (
                comparison_df[
                    "seriousness"
                ]
                == "Serious"
            ).sum()
        )

        comparison_nonserious = int(
            (
                comparison_df[
                    "seriousness"
                ]
                == "Non-Serious"
            ).sum()
        )


        serious_or = None

        if (
            target_serious
            + target_nonserious
            > 0
            and
            comparison_serious
            + comparison_nonserious
            > 0
        ):

            value, low, high, corrected = (
                odds_ratio(
                    target_serious,
                    target_nonserious,
                    comparison_serious,
                    comparison_nonserious
                )
            )

            serious_or = {
                "or":
                    value,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "corrected":
                    corrected,

                "target": {
                    "serious":
                        target_serious,

                    "known":
                        (
                            target_serious
                            + target_nonserious
                        ),

                    "share":
                        (
                            target_serious
                            /
                            (
                                target_serious
                                + target_nonserious
                            )
                        )
                },

                "comparison": {
                    "serious":
                        comparison_serious,

                    "known":
                        (
                            comparison_serious
                            + comparison_nonserious
                        ),

                    "share":
                        (
                            comparison_serious
                            /
                            (
                                comparison_serious
                                + comparison_nonserious
                            )
                        )
                }
            }


        # ----------------------------------------------------
        # REACTIONS
        # ----------------------------------------------------


        # EUVIGIL_COMPARE_SHARED_REACTIONS_V1
        # Préparation mutualisée entre comparaisons du même périmètre.
        prepared_key = (
            scope,
            code,
            "compare_prepared_reactions_v1"
        )

        with self._lock:
            prepared = self._derived.get(prepared_key)

        if prepared is None:
            base_reactions = self._load(
                scope,
                code,
                "reactions",
                [
                    "safety_report_id",
                    "reaction_pt",
                ]
            ).copy()

            base_reactions = base_reactions[
                base_reactions["safety_report_id"].isin(
                    population_ids
                )
            ]

            base_reactions["reaction_pt"] = (
                base_reactions["reaction_pt"]
                .astype("string")
                .str.strip()
            )

            base_reactions = base_reactions[
                base_reactions["reaction_pt"].notna()
                &
                (base_reactions["reaction_pt"] != "")
            ].drop_duplicates(
                ["safety_report_id", "reaction_pt"]
            )

            prepared_codes, prepared_labels = (
                pd.factorize(
                    base_reactions["reaction_pt"],
                    sort=True
                )
            )

            # Les tableaux partagés ne doivent pas être modifiés.
            prepared_codes.setflags(write=False)

            candidate = (
                base_reactions,
                prepared_codes,
                prepared_labels,
            )

            with self._lock:
                prepared = self._derived.setdefault(
                    prepared_key,
                    candidate
                )

        base_reactions, reaction_codes, reaction_labels = (
            prepared
        )

        # Copie superficielle : les colonnes d'origine restent partagées.
        # _target est propre à chaque comparaison.
        reactions = base_reactions.copy(deep=False)

        reactions["_target"] = (
            reactions["safety_report_id"]
            .isin(target_ids)
        )

        valid = reaction_codes >= 0
        codes = reaction_codes[valid]

        target_flags = reactions["_target"].to_numpy(
            dtype=bool,
            copy=False
        )[valid]

        total_counts = np.bincount(
            codes,
            minlength=len(reaction_labels)
        ).astype("int64")

        target_counts = np.bincount(
            codes[target_flags],
            minlength=len(reaction_labels)
        ).astype("int64")

        reaction_agg = pd.DataFrame(
            {
                "target_reports": target_counts,
                "total_reports": total_counts
            },
            index=pd.Index(
                reaction_labels,
                name="reaction_pt"
            )
        )

        reaction_rows = []

        for reaction, target_raw, total_raw in reaction_agg.itertuples(index=True, name=None):

            a = int(
                target_raw
            )

            c = int(
                total_raw
                - target_raw
            )

            b = (
                target_total
                - a
            )

            d = (
                comparison_total
                - c
            )

            # On évite des signaux purement anecdotiques.
            if (
                a + c < 10
                or
                (
                    a < 5
                    and c < 5
                )
            ):
                continue

            value, low, high, corrected = (
                odds_ratio(
                    a, b, c, d
                )
            )

            reaction_rows.append({
                "reaction":
                    str(reaction),

                "target_reports":
                    a,

                "comparison_reports":
                    c,

                "target_share":
                    (
                        a / target_total
                    ),

                "comparison_share":
                    (
                        c / comparison_total
                    ),

                "ror":
                    value,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "corrected":
                    corrected,
            })


        positive_reactions = [
            x
            for x in reaction_rows
            if (
                x["target_reports"] >= 5
                and x["ci95_low"] > 1
            )
        ]

        positive_reactions.sort(
            key=lambda x: (
                x["ci95_low"],
                x["target_reports"]
            ),
            reverse=True
        )


        negative_reactions = [
            x
            for x in reaction_rows
            if (
                x["comparison_reports"] >= 5
                and x["ci95_high"] < 1
            )
        ]

        negative_reactions.sort(
            key=lambda x: (
                x["ci95_high"],
                -x["comparison_reports"]
            )
        )


        # ----------------------------------------------------
        # TOP REACTIONS DES DEUX GROUPES
        # ----------------------------------------------------

        target_top = (
            reactions[
                reactions["_target"]
            ][
                "reaction_pt"
            ]
            .value_counts()
            .head(20)
        )

        comparison_top = (
            reactions[
                ~reactions["_target"]
            ][
                "reaction_pt"
            ]
            .value_counts()
            .head(20)
        )


        top_reactions = {
            "target": [
                {
                    "label":
                        str(label),

                    "reports":
                        int(count)
                }
                for label, count
                in target_top.items()
            ],

            "comparison": [
                {
                    "label":
                        str(label),

                    "reports":
                        int(count)
                }
                for label, count
                in comparison_top.items()
            ],
        }


        # ----------------------------------------------------
        # SIMILARITE DES PROFILS DE REACTIONS
        #
        # Cosinus des prévalences de déclaration.
        # Indicateur descriptif uniquement.
        # ----------------------------------------------------

        p1 = (
            reaction_agg[
                "target_reports"
            ]
            .astype(float)
            /
            target_total
        )

        p2 = (
            (
                reaction_agg[
                    "total_reports"
                ]
                -
                reaction_agg[
                    "target_reports"
                ]
            )
            .astype(float)
            /
            comparison_total
        )

        dot = float(
            (p1 * p2).sum()
        )

        norm1 = math.sqrt(
            float(
                (p1 * p1).sum()
            )
        )

        norm2 = math.sqrt(
            float(
                (p2 * p2).sum()
            )
        )

        reaction_similarity = (
            dot
            /
            (norm1 * norm2)
            if norm1 and norm2
            else None
        )


        # ----------------------------------------------------
        # PAYS SUR-REPRESENTES DANS LE GROUPE CIBLE
        # ----------------------------------------------------

        countries = (
            icsr[
                [
                    "safety_report_id",
                    "regulatory_country",
                    "_group"
                ]
            ]
            .copy()
        )

        countries[
            "regulatory_country"
        ] = (
            countries[
                "regulatory_country"
            ]
            .astype("string")
            .str.strip()
        )

        countries = countries[
            countries[
                "regulatory_country"
            ].notna()
            &
            (
                countries[
                    "regulatory_country"
                ] != ""
            )
            &
            ~countries[
                "regulatory_country"
            ]
            .str.casefold()
            .isin(
                {
                    "not specified",
                    "not available",
                    "n/a",
                }
            )
        ]


        country_rows = []

        grouped_country = (
            countries
            .groupby(
                "regulatory_country"
            )["_group"]
            .agg(
                target_reports="sum",
                total_reports="size"
            )
        )

        for country, target_raw, total_raw in grouped_country.itertuples(index=True, name=None):

            a = int(
                target_raw
            )

            c = int(
                total_raw
                - target_raw
            )

            b = (
                target_total
                - a
            )

            d = (
                comparison_total
                - c
            )

            if (
                a + c < 100
                or
                (
                    a < 5
                    and c < 5
                )
            ):
                continue

            value, low, high, corrected = (
                odds_ratio(
                    a, b, c, d
                )
            )

            country_rows.append({
                "country":
                    str(country),

                "target_reports":
                    a,

                "comparison_reports":
                    c,

                "target_share":
                    (
                        a / target_total
                    ),

                "comparison_share":
                    (
                        c / comparison_total
                    ),

                "or":
                    value,

                "ci95_low":
                    low,

                "ci95_high":
                    high,
            })


        country_positive = [
            x
            for x in country_rows
            if (
                x["target_reports"] >= 5
                and x["ci95_low"] > 1
            )
        ]

        country_positive.sort(
            key=lambda x: (
                x["ci95_low"],
                x["target_reports"]
            ),
            reverse=True
        )


        # ----------------------------------------------------
        # RESULTAT
        # ----------------------------------------------------

        result = {

            "scope": {
                "type":
                    scope,

                "code":
                    code,
            },

            "target": {
                "type":
                    target_scope,

                "code":
                    target_code_norm,

                "label":
                    target_label,

                "reports":
                    target_total,
            },

            "comparison": {
                "type":
                    "remainder",

                "label":
                    f"Reste de {code}",

                "reports":
                    comparison_total,
            },

            "summary": {
                "population_reports":
                    (
                        target_total
                        + comparison_total
                    ),

                "target_reports":
                    target_total,

                "comparison_reports":
                    comparison_total,

                "target_share":
                    (
                        target_total
                        /
                        (
                            target_total
                            + comparison_total
                        )
                    ),

                "reaction_similarity":
                    reaction_similarity,
            },

            "profile":
                profile,

            "timeline":
                timeline,

            "seriousness_or":
                serious_or,

            "top_reactions":
                top_reactions,

            "reaction_ror_positive":
                positive_reactions[:25],

            "reaction_ror_negative":
                negative_reactions[:25],

            "country_overrepresentation":
                country_positive[:20],

            "method": {
                "comparison":
                    "target-vs-remainder",

                "reaction_min_total":
                    10,

                "reaction_min_group":
                    5,

                "country_min_total":
                    100,

                "multiple_testing_correction":
                    False,

                "exploratory":
                    True,
            },

            "cache": {
                "hit":
                    True,

                "type":
                    "raw-parquet",
            },
        }


        return result


    # ========================================================
    # PRESENCE DES SUBSTANCES DANS LES ICSR
    # ========================================================

    def _pair_presence(
        self,
        scope,
        code
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        key = (
            scope,
            code,
            "pair_presence"
        )

        with self._lock:
            cached = self._derived.get(
                key
            )

        if cached is not None:
            return cached


        # EUVIGIL_FAST_L2_PAIR_PRESENCE_V1
        # Pour les L2 virtuelles, réutiliser les associations
        # de la classe L1 parente et filtrer par ICSR.
        #
        # Ne concerne pas les L2 physiques, J07 ni les
        # groupes vaccinaux virtuels.
        if (
            scope == "l2"
            and len(code) == 3
            and code[0].isalpha()
            and code[1:].isdigit()
        ):
            parent = code[0]
            parent_path = self.root / "l1" / parent
            physical_l2 = self.root / "l2" / code

            # EUVIGIL_FAST_ALL_L2_PRESENCE_V1
            # J07 conserve son cache physique dédié.
            if (
                parent_path.is_dir()
                and code != "J07"
            ):
                try:
                    # Vérifier le périmètre exact du L2.
                    if physical_l2.is_dir():
                        # Périmètre réel de la L2 physique.
                        ids_frame = pd.read_parquet(
                            physical_l2 / "icsr",
                            columns=["safety_report_id"]
                        )

                        allowed_ids = pd.to_numeric(
                            ids_frame["safety_report_id"],
                            errors="coerce"
                        ).dropna().unique()

                        if len(allowed_ids) == 0:
                            raise RuntimeError(
                                f"Périmètre ICSR vide : {code}"
                            )
                    else:
                        # L2 virtuelle : liens du parent.
                        allowed_ids = self._virtual_l2_report_ids(
                            code, parent_path
                        )

                    # Charger l'index préconstruit du parent.
                    # Ce résultat est mutualisé dans _derived.
                    parent_presence = self._pair_presence(
                        "l1", parent
                    )

                    # Filtrer les ICSR, pas les substances :
                    # tous les médicaments partenaires restent.
                    filtered = parent_presence.loc[
                        parent_presence[
                            "safety_report_id"
                        ].isin(allowed_ids),
                        [
                            "safety_report_id",
                            "substance",
                            "role"
                        ]
                    ].copy()

                    with self._lock:
                        self._derived[key] = filtered

                    return filtered

                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Index L2 accéléré indisponible "
                        "pour %s/%s : %s. "
                        "Retour au chargement historique.",
                        scope, code, exc
                    )

        # Chemin historique et caches physiques ci-dessous.

        # EUVIGIL_DISK_PAIR_PRESENCE_V1
        # Caches physiques L1 et J07 uniquement.
        # En cas d'anomalie : extraction historique inchangée.
        if (
            (scope == "l1" and code in "ABCDGHJLMNPRSV")
            or (scope == "l2" and code == "J07")
        ):
            try:
                import sqlite3
                from pathlib import Path

                root = Path(self.root)
                stem = f"{scope}_{code}"

                cache_file = (
                    root / "_pair_presence_v1"
                    / f"{stem}.parquet"
                )
                state_file = (
                    root / "_pair_presence_work"
                    / f"{stem}.sqlite"
                )

                if not cache_file.is_file():
                    raise FileNotFoundError(cache_file)
                if not state_file.is_file():
                    raise FileNotFoundError(state_file)

                source_files = {}
                for kind in (
                    "suspect_drugs",
                    "concomitant_drugs"
                ):
                    directory = root / scope / code / kind
                    if not directory.is_dir():
                        raise FileNotFoundError(directory)

                    for file in directory.glob("*.parquet"):
                        st = file.stat()
                        source_files[str(file)] = (
                            st.st_size,
                            st.st_mtime_ns
                        )

                if not source_files:
                    raise RuntimeError("Sources vides")

                uri = state_file.resolve().as_uri() + "?mode=ro"
                with sqlite3.connect(uri, uri=True) as con:
                    manifest = {
                        name: (size, mtime)
                        for name, size, mtime in con.execute(
                            "SELECT filename, size_bytes, mtime_ns "
                            "FROM completed_files"
                        )
                    }

                if manifest != source_files:
                    raise RuntimeError(
                        f"Cache {stem} périmé : "
                        "manifest des sources différent"
                    )

                if (
                    cache_file.stat().st_mtime_ns
                    < max(mtime for _, mtime in source_files.values())
                ):
                    raise RuntimeError(
                        f"Cache {stem} antérieur aux sources"
                    )

                presence = pd.read_parquet(
                    cache_file,
                    columns=[
                        "safety_report_id",
                        "substance",
                        "role"
                    ]
                )

                if (
                    list(presence.columns) != [
                        "safety_report_id",
                        "substance",
                        "role"
                    ]
                    or presence.empty
                    or presence["safety_report_id"].isna().any()
                    or presence["substance"].isna().any()
                    or not presence["role"].isin(
                        ["suspect", "concomitant"]
                    ).all()
                ):
                    raise RuntimeError(
                        f"Cache {stem} invalide"
                    )

                with self._lock:
                    self._derived[key] = presence

                return presence

            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "EUVIGIL cache de présence ignoré "
                    "pour %s/%s : %s",
                    scope, code, exc
                )

        # Extraction historique conservée ci-dessous.

        suspects = self._load(
            scope,
            code,
            "suspect_drugs",
            [
                "safety_report_id",
                "drug",
            ]
        )

        concomitants = self._load(
            scope,
            code,
            "concomitant_drugs",
            [
                "safety_report_id",
                "drug",
            ]
        )


        def extract(value):

            if pd.isna(value):
                return []

            text = str(value).strip()

            if not text:
                return []

            low = text.casefold()

            if (
                low == "not reported"
                or low.startswith("not available")
                or low.startswith("[truncated")
            ):
                return []


            left = text.find("[")
            right = text.find(
                "]",
                left + 1
            )

            if (
                left != -1
                and right > left + 1
            ):

                raw = text[
                    left + 1:right
                ]

            else:

                raw = text.split(
                    " (",
                    1
                )[0]


            values = []

            for item in raw.split(","):

                item = " ".join(
                    item
                    .strip()
                    .upper()
                    .split()
                )

                if item in {
                    "",
                    "N/A",
                    "UNKNOWN",
                    "NOT AVAILABLE",
                    "NOT REPORTED",
                }:
                    continue

                values.append(
                    item
                )

            return values


        frames = []

        for role, source in (
            (
                "suspect",
                suspects
            ),
            (
                "concomitant",
                concomitants
            ),
        ):

            x = source[
                [
                    "safety_report_id",
                    "drug"
                ]
            ].copy()

            x["substance"] = (
                x["drug"]
                .map(extract)
            )

            x = x.explode(
                "substance"
            )

            x = x[
                x["substance"].notna()
                &
                (
                    x["substance"]
                    != ""
                )
            ][
                [
                    "safety_report_id",
                    "substance"
                ]
            ]

            x["role"] = role

            frames.append(x)


        presence = pd.concat(
            frames,
            ignore_index=True
        )

        presence = (
            presence
            .drop_duplicates(
                [
                    "safety_report_id",
                    "substance",
                    "role",
                ]
            )
        )


        with self._lock:
            self._derived[key] = (
                presence
            )

        return presence


    # ========================================================
    # SUBSTANCES DISPONIBLES POUR A+B
    # ========================================================

    def pair_options(
        self,
        scope,
        code
    ):

        scope, code, _ = (
            self._scope_path(
                scope,
                code
            )
        )

        presence = self._pair_presence(
            scope,
            code
        )


        any_presence = (
            presence
            .drop_duplicates(
                [
                    "safety_report_id",
                    "substance"
                ]
            )
        )

        total_counts = (
            any_presence
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )

        suspect_counts = (
            presence[
                presence["role"]
                == "suspect"
            ]
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )

        concomitant_counts = (
            presence[
                presence["role"]
                == "concomitant"
            ]
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )


        items = []

        for substance, count in (
            total_counts.items()
        ):

            # Les substances extrêmement anecdotiques
            # n'encombrent pas le sélecteur.
            if int(count) < 5:
                continue

            items.append({
                "name":
                    str(substance),

                "reports":
                    int(count),

                "suspect_reports":
                    int(
                        suspect_counts.get(
                            substance,
                            0
                        )
                    ),

                "concomitant_reports":
                    int(
                        concomitant_counts.get(
                            substance,
                            0
                        )
                    ),
            })


        items.sort(
            key=lambda x:
                x["name"].casefold()
        )


        return {
            "scope": {
                "type": scope,
                "code": code,
            },

            "substances":
                items,

            "roles": [
                {
                    "code": "any",
                    "label_fr":
                        "Présentes dans le même signalement, quel que soit le rôle",
                },
                {
                    "code": "suspect_suspect",
                    "label_fr":
                        "A suspecte + B suspecte",
                },
                {
                    "code": "a_suspect_b_concomitant",
                    "label_fr":
                        "A suspecte + B concomitante",
                },
                {
                    "code": "a_concomitant_b_suspect",
                    "label_fr":
                        "A concomitante + B suspecte",
                },
            ],

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


    # ========================================================
    # ANALYSE A+B
    # ========================================================

    def analyse_pair(
        self,
        scope,
        code,
        substance_a,
        substance_b,
        role_mode="any"
    ):

        scope, code, path = (
            self._scope_path(
                scope,
                code
            )
        )

        a_name = " ".join(
            str(substance_a)
            .strip()
            .upper()
            .split()
        )

        b_name = " ".join(
            str(substance_b)
            .strip()
            .upper()
            .split()
        )

        if (
            not a_name
            or not b_name
        ):
            raise ValueError(
                "deux substances sont nécessaires"
            )

        if a_name == b_name:
            raise ValueError(
                "A et B doivent être différentes"
            )


        role_mode = (
            str(role_mode)
            .strip()
            .lower()
        )

        allowed_modes = {
            "any",
            "suspect_suspect",
            "a_suspect_b_concomitant",
            "a_concomitant_b_suspect",
        }

        if role_mode not in allowed_modes:
            raise ValueError(
                "mode de rôle invalide"
            )


        cache_key = (
            scope,
            code,
            "pair",
            a_name,
            b_name,
            role_mode
        )

        with self._lock:
            cached = self._derived.get(
                cache_key
            )

        if cached is not None:
            return cached


        presence = self._pair_presence(
            scope,
            code
        )


        def ids_for(
            substance,
            role=None
        ):

            x = presence[
                presence["substance"]
                == substance
            ]

            if role is not None:

                x = x[
                    x["role"]
                    == role
                ]

            return set(
                x["safety_report_id"]
                .dropna()
                .unique()
                .tolist()
            )


        if role_mode == "any":

            a_ids = ids_for(
                a_name
            )

            b_ids = ids_for(
                b_name
            )

            role_label = (
                "présence dans le même signalement, "
                "quel que soit le rôle"
            )


        elif role_mode == "suspect_suspect":

            a_ids = ids_for(
                a_name,
                "suspect"
            )

            b_ids = ids_for(
                b_name,
                "suspect"
            )

            role_label = (
                "A suspecte + B suspecte"
            )


        elif role_mode == "a_suspect_b_concomitant":

            a_ids = ids_for(
                a_name,
                "suspect"
            )

            b_ids = ids_for(
                b_name,
                "concomitant"
            )

            role_label = (
                "A suspecte + B concomitante"
            )


        else:

            a_ids = ids_for(
                a_name,
                "concomitant"
            )

            b_ids = ids_for(
                b_name,
                "suspect"
            )

            role_label = (
                "A concomitante + B suspecte"
            )


        if not a_ids:
            raise ValueError(
                f"{a_name} absente pour ce rôle"
            )

        if not b_ids:
            raise ValueError(
                f"{b_name} absente pour ce rôle"
            )


        icsr = self._load(
            scope,
            code,
            "icsr",
            [
                "safety_report_id",
                "gateway_year",
                "sex",
                "age_group",
                "seriousness",
                "regulatory_country",
            ]
        ).drop_duplicates(
            "safety_report_id"
        )


        population_ids = set(
            icsr[
                "safety_report_id"
            ]
            .dropna()
            .tolist()
        )

        a_ids &= population_ids
        b_ids &= population_ids


        pair_ids = (
            a_ids
            & b_ids
        )

        a_without_b = (
            a_ids
            - b_ids
        )

        b_without_a = (
            b_ids
            - a_ids
        )


        if len(pair_ids) < 5:
            raise ValueError(
                "moins de 5 signalements "
                "contiennent cette association"
            )


        # ----------------------------------------------------
        # HELPER OR / IC95
        # ----------------------------------------------------

        def odds_ratio(
            event_exposed,
            total_exposed,
            event_reference,
            total_reference
        ):

            a = int(
                event_exposed
            )

            b = int(
                total_exposed
                - event_exposed
            )

            c = int(
                event_reference
            )

            d = int(
                total_reference
                - event_reference
            )

            corrected = any(
                x == 0
                for x in (
                    a, b, c, d
                )
            )

            if corrected:
                aa = a + 0.5
                bb = b + 0.5
                cc = c + 0.5
                dd = d + 0.5
            else:
                aa = float(a)
                bb = float(b)
                cc = float(c)
                dd = float(d)


            value = (
                aa * dd
                /
                (bb * cc)
            )

            se = math.sqrt(
                1 / aa
                + 1 / bb
                + 1 / cc
                + 1 / dd
            )

            low = math.exp(
                math.log(value)
                - 1.96 * se
            )

            high = math.exp(
                math.log(value)
                + 1.96 * se
            )

            return {
                "value":
                    value,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "corrected":
                    corrected,
            }


        # ----------------------------------------------------
        # REACTIONS
        # ----------------------------------------------------

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt",
            ]
        ).copy()

        relevant_ids = (
            pair_ids
            | a_without_b
            | b_without_a
        )

        reactions = reactions[
            reactions[
                "safety_report_id"
            ].isin(relevant_ids)
        ]

        reactions["reaction_pt"] = (
            reactions["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        reactions = reactions[
            reactions[
                "reaction_pt"
            ].notna()
            &
            (
                reactions[
                    "reaction_pt"
                ] != ""
            )
        ].drop_duplicates(
            [
                "safety_report_id",
                "reaction_pt"
            ]
        )


        pair_counts = (
            reactions[
                reactions[
                    "safety_report_id"
                ].isin(pair_ids)
            ][
                "reaction_pt"
            ]
            .value_counts()
        )

        a_counts = (
            reactions[
                reactions[
                    "safety_report_id"
                ].isin(a_without_b)
            ][
                "reaction_pt"
            ]
            .value_counts()
        )

        b_counts = (
            reactions[
                reactions[
                    "safety_report_id"
                ].isin(b_without_a)
            ][
                "reaction_pt"
            ]
            .value_counts()
        )


        all_effects = set(
            pair_counts.index.tolist()
        )

        effect_rows = []


        for reaction in all_effects:

            n_pair = int(
                pair_counts.get(
                    reaction,
                    0
                )
            )

            if n_pair < 5:
                continue

            n_a = int(
                a_counts.get(
                    reaction,
                    0
                )
            )

            n_b = int(
                b_counts.get(
                    reaction,
                    0
                )
            )


            ror_vs_a = None

            if len(a_without_b) > 0:

                ror_vs_a = odds_ratio(
                    n_pair,
                    len(pair_ids),
                    n_a,
                    len(a_without_b)
                )


            ror_vs_b = None

            if len(b_without_a) > 0:

                ror_vs_b = odds_ratio(
                    n_pair,
                    len(pair_ids),
                    n_b,
                    len(b_without_a)
                )


            # Signal statistique minimal :
            # même direction dans les deux comparaisons.
            dual_signal = (
                ror_vs_a is not None
                and ror_vs_b is not None
                and len(a_without_b) >= 50
                and len(b_without_a) >= 50
                and ror_vs_a["ci95_low"] > 1
                and ror_vs_b["ci95_low"] > 1
            )

            # Signal robuste pour affichage grand public :
            # - au moins 20 dossiers A+B avec l'effet
            # - au moins 5 dossiers avec l'effet dans chacun
            #   des deux comparateurs
            # - IC95% inférieur > 1 dans les deux comparaisons
            robust_signal = (
                dual_signal
                and n_pair >= 20
                and n_a >= 5
                and n_b >= 5
            )


            score = None

            if (
                ror_vs_a is not None
                and ror_vs_b is not None
            ):

                score = min(
                    ror_vs_a[
                        "ci95_low"
                    ],
                    ror_vs_b[
                        "ci95_low"
                    ]
                )


            effect_rows.append({
                "reaction":
                    str(reaction),

                "pair_reports":
                    n_pair,

                "pair_share":
                    (
                        n_pair
                        / len(pair_ids)
                    ),

                "a_without_b_reports":
                    n_a,

                "a_without_b_share":
                    (
                        n_a
                        / len(a_without_b)
                        if a_without_b
                        else None
                    ),

                "b_without_a_reports":
                    n_b,

                "b_without_a_share":
                    (
                        n_b
                        / len(b_without_a)
                        if b_without_a
                        else None
                    ),

                "ror_vs_a_without_b":
                    ror_vs_a,

                "ror_vs_b_without_a":
                    ror_vs_b,

                "dual_signal":
                    dual_signal,

                "robust_signal":
                    robust_signal,

                "_score":
                    score,
            })


        # ----------------------------------------------------
        # EFFETS QUI RESSORTENT DANS LES DEUX COMPARAISONS
        # ----------------------------------------------------

        # Signaux robustes : affichés en premier au public.
        robust_signals = [
            x
            for x in effect_rows
            if x["robust_signal"]
        ]

        robust_signals.sort(
            key=lambda x: (
                x["_score"]
                if x["_score"] is not None
                else 0,
                x["pair_reports"]
            ),
            reverse=True
        )


        # Signaux rares : statistiquement intéressants,
        # mais trop peu documentés pour être mis au premier plan.
        rare_signals = [
            x
            for x in effect_rows
            if (
                x["dual_signal"]
                and not x["robust_signal"]
            )
        ]

        rare_signals.sort(
            key=lambda x: (
                x["_score"]
                if x["_score"] is not None
                else 0,
                x["pair_reports"]
            ),
            reverse=True
        )


        # ----------------------------------------------------
        # EFFETS LES PLUS FREQUENTS DE A+B
        # ----------------------------------------------------

        frequent = sorted(
            effect_rows,
            key=lambda x:
                x["pair_reports"],
            reverse=True
        )[:25]


        # Enlever la clé interne.
        for row in effect_rows:
            row.pop(
                "_score",
                None
            )


        # ----------------------------------------------------
        # PROFIL DU GROUPE A+B
        # ----------------------------------------------------

        pair_df = icsr[
            icsr[
                "safety_report_id"
            ].isin(pair_ids)
        ].copy()


        def distribution(
            field,
            default
        ):

            counts = (
                pair_df[field]
                .fillna(default)
                .replace(
                    "",
                    default
                )
                .value_counts()
            )

            return [
                {
                    "value":
                        str(value),

                    "count":
                        int(count),

                    "share":
                        (
                            int(count)
                            / len(pair_df)
                            if len(pair_df)
                            else 0
                        )
                }
                for value, count
                in counts.items()
            ]


        profile = {
            "sex":
                distribution(
                    "sex",
                    "Not Specified"
                ),

            "age_group":
                distribution(
                    "age_group",
                    "Not Specified"
                ),

            "seriousness":
                distribution(
                    "seriousness",
                    "Not Available"
                ),

            "regulatory_country":
                distribution(
                    "regulatory_country",
                    "Not Specified"
                ),
        }


        # ----------------------------------------------------
        # CHRONOLOGIE A+B
        # ----------------------------------------------------

        pair_years = (
            pair_df[
                "gateway_year"
            ]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        population_years = (
            icsr[
                "gateway_year"
            ]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        years = sorted(
            set(
                population_years
                .index
                .tolist()
            )
            |
            set(
                pair_years
                .index
                .tolist()
            )
        )

        timeline = []

        for year in years:

            n = int(
                pair_years.get(
                    year,
                    0
                )
            )

            total = int(
                population_years.get(
                    year,
                    0
                )
            )

            timeline.append({
                "year":
                    int(year),

                "pair_reports":
                    n,

                "population_reports":
                    total,

                "share":
                    (
                        n / total
                        if total
                        else 0
                    )
            })


        # ----------------------------------------------------
        # PROFILS DES TROIS GROUPES
        # ----------------------------------------------------

        a_only_df = icsr[
            icsr["safety_report_id"]
            .isin(a_without_b)
        ].copy()

        b_only_df = icsr[
            icsr["safety_report_id"]
            .isin(b_without_a)
        ].copy()


        # ----------------------------------------------------
        # GRAVITE / EVENEMENTS PARTICULIERS
        # ----------------------------------------------------

        def event_metric(event_ids):

            event_ids = set(event_ids)

            pair_n = len(
                pair_ids & event_ids
            )

            a_n = len(
                a_without_b & event_ids
            )

            b_n = len(
                b_without_a & event_ids
            )


            vs_a = None

            if len(a_without_b):

                vs_a = odds_ratio(
                    pair_n,
                    len(pair_ids),
                    a_n,
                    len(a_without_b)
                )


            vs_b = None

            if len(b_without_a):

                vs_b = odds_ratio(
                    pair_n,
                    len(pair_ids),
                    b_n,
                    len(b_without_a)
                )


            return {
                "pair_reports":
                    pair_n,

                "pair_share":
                    (
                        pair_n / len(pair_ids)
                        if pair_ids
                        else 0
                    ),

                "a_without_b_reports":
                    a_n,

                "a_without_b_share":
                    (
                        a_n / len(a_without_b)
                        if a_without_b
                        else None
                    ),

                "b_without_a_reports":
                    b_n,

                "b_without_a_share":
                    (
                        b_n / len(b_without_a)
                        if b_without_a
                        else None
                    ),

                "or_vs_a_without_b":
                    vs_a,

                "or_vs_b_without_a":
                    vs_b,
            }


        # Serious : champ structuré ICSR
        serious_ids = set(
            icsr.loc[
                icsr["seriousness"]
                == "Serious",
                "safety_report_id"
            ]
            .tolist()
        )


        # Death : terme MedDRA exact
        death_ids = set(
            reactions.loc[
                reactions["reaction_pt"]
                .astype("string")
                .str.strip()
                .str.casefold()
                == "death",
                "safety_report_id"
            ]
            .tolist()
        )


        # Hospitalisation :
        # recherche textuelle dans les chaînes brutes.
        raw_reactions = pd.read_parquet(
            path / "reactions",
            columns=[
                "safety_report_id",
                "reaction_pt",
                "reaction_details",
                "reaction_raw",
            ]
        )

        raw_reactions = raw_reactions[
            raw_reactions[
                "safety_report_id"
            ].isin(relevant_ids)
        ]


        hospital_mask = pd.Series(
            False,
            index=raw_reactions.index
        )

        for column in (
            "reaction_pt",
            "reaction_details",
            "reaction_raw",
        ):

            hospital_mask |= (
                raw_reactions[column]
                .astype("string")
                .str.contains(
                    "hospital",
                    case=False,
                    regex=False,
                    na=False
                )
            )


        hospital_ids = set(
            raw_reactions.loc[
                hospital_mask,
                "safety_report_id"
            ]
            .unique()
            .tolist()
        )


        severity = {
            "death":
                event_metric(
                    death_ids
                ),

            "hospitalisation":
                event_metric(
                    hospital_ids
                ),

            "serious":
                event_metric(
                    serious_ids
                ),
        }


        # ----------------------------------------------------
        # AUTRES SUBSTANCES PRESENTES AVEC A+B
        #
        # Très important pour montrer la polypharmacie et
        # les facteurs de confusion possibles.
        # ----------------------------------------------------

        any_presence = (
            presence
            .drop_duplicates(
                [
                    "safety_report_id",
                    "substance"
                ]
            )
        )

        pair_other = any_presence[
            any_presence[
                "safety_report_id"
            ].isin(pair_ids)
            &
            ~any_presence[
                "substance"
            ].isin(
                {
                    a_name,
                    b_name
                }
            )
        ]


        other_counts = (
            pair_other
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )


        other_substances = [
            {
                "substance":
                    str(substance),

                "reports":
                    int(count),

                "share":
                    (
                        int(count)
                        / len(pair_ids)
                        if pair_ids
                        else 0
                    )
            }

            for substance, count
            in other_counts.head(25).items()
        ]


        # ----------------------------------------------------
        # PAYS DISPROPORTIONNELLEMENT REPRESENTES
        # A+B VS RESTE DU PERIMETRE
        # ----------------------------------------------------

        country_frame = icsr[
            [
                "safety_report_id",
                "regulatory_country"
            ]
        ].copy()

        country_frame[
            "regulatory_country"
        ] = (
            country_frame[
                "regulatory_country"
            ]
            .astype("string")
            .str.strip()
        )

        country_frame = country_frame[
            country_frame[
                "regulatory_country"
            ].notna()
            &
            (
                country_frame[
                    "regulatory_country"
                ] != ""
            )
            &
            ~country_frame[
                "regulatory_country"
            ]
            .str.casefold()
            .isin(
                {
                    "not specified",
                    "not available",
                    "n/a",
                }
            )
        ]


        rest_ids = (
            population_ids
            - pair_ids
        )

        country_disproportion = []


        for country, group in (
            country_frame.groupby(
                "regulatory_country"
            )
        ):

            ids = set(
                group[
                    "safety_report_id"
                ].tolist()
            )

            a = len(
                pair_ids & ids
            )

            c = len(
                rest_ids & ids
            )

            if a < 5:
                continue

            if a + c < 100:
                continue


            result_or = odds_ratio(
                a,
                len(pair_ids),
                c,
                len(rest_ids)
            )


            if result_or[
                "ci95_low"
            ] <= 1:
                continue


            country_disproportion.append({
                "country":
                    str(country),

                "pair_reports":
                    a,

                "pair_share":
                    (
                        a / len(pair_ids)
                        if pair_ids
                        else 0
                    ),

                "rest_reports":
                    c,

                "rest_share":
                    (
                        c / len(rest_ids)
                        if rest_ids
                        else 0
                    ),

                "or":
                    result_or[
                        "value"
                    ],

                "ci95_low":
                    result_or[
                        "ci95_low"
                    ],

                "ci95_high":
                    result_or[
                        "ci95_high"
                    ],
            })


        country_disproportion.sort(
            key=lambda x: (
                x["ci95_low"],
                x["pair_reports"]
            ),
            reverse=True
        )


        # ----------------------------------------------------
        # RESULTAT
        # ----------------------------------------------------

        result = {

            "scope": {
                "type":
                    scope,

                "code":
                    code,
            },

            "pair": {
                "substance_a":
                    a_name,

                "substance_b":
                    b_name,

                "role_mode":
                    role_mode,

                "role_label":
                    role_label,
            },

            "summary": {
                "population_reports":
                    len(population_ids),

                "a_reports":
                    len(a_ids),

                "b_reports":
                    len(b_ids),

                "pair_reports":
                    len(pair_ids),

                "a_without_b_reports":
                    len(a_without_b),

                "b_without_a_reports":
                    len(b_without_a),

                "pair_share":
                    (
                        len(pair_ids)
                        / len(population_ids)
                        if population_ids
                        else 0
                    ),

                "pair_share_of_a":
                    (
                        len(pair_ids)
                        / len(a_ids)
                        if a_ids
                        else 0
                    ),

                "pair_share_of_b":
                    (
                        len(pair_ids)
                        / len(b_ids)
                        if b_ids
                        else 0
                    ),

                "union_reports":
                    len(
                        a_ids | b_ids
                    ),

                "jaccard":
                    (
                        len(pair_ids)
                        / len(a_ids | b_ids)
                        if (a_ids | b_ids)
                        else 0
                    ),
            },

            "severity":
                severity,

            "other_substances":
                other_substances,

            "country_disproportion":
                country_disproportion[:20],

            "profile":
                profile,

            "timeline":
                timeline,

            "top_effects":
                frequent,

            "effect_signals":
                robust_signals[:30],

            "rare_effect_signals":
                rare_signals[:30],

            "method": {
                "unit":
                    "unique safety_report_id",

                "effect_min_pair_reports":
                    5,

                "robust_effect_min_pair_reports":
                    20,

                "robust_effect_min_each_comparator":
                    5,

                "comparator_min_reports":
                    50,

                "signal_definition":
                    "lower 95% CI > 1 versus both A-without-B and B-without-A",

                "multiple_testing_correction":
                    False,

                "hospitalisation_detection":
                    "text search hospital* in reaction strings",

                "country_comparison":
                    "A+B versus remainder of current scope",

                "third_substances":
                    "descriptive co-reporting only",

                "causal":
                    False,
            },

            "cache": {
                "hit":
                    True,

                "type":
                    "raw-parquet",
            },
        }


        with self._lock:
            self._derived[
                cache_key
            ] = result

        return result


    # ========================================================
    # RECHERCHE DE TERMES DE REACTION
    # ========================================================

    def reaction_search(
        self,
        scope,
        code,
        query="",
        limit=30
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        key = (
            scope,
            code,
            "reaction_search_counts"
        )

        with self._lock:
            counts = self._derived.get(key)

        if counts is None:

            reactions = self._load(
                scope,
                code,
                "reactions",
                [
                    "safety_report_id",
                    "reaction_pt",
                ]
            ).copy()

            reactions["reaction_pt"] = (
                reactions["reaction_pt"]
                .astype("string")
                .str.strip()
            )

            reactions = reactions[
                reactions["reaction_pt"].notna()
                &
                (
                    reactions["reaction_pt"]
                    != ""
                )
            ].drop_duplicates(
                [
                    "safety_report_id",
                    "reaction_pt"
                ]
            )

            counts = (
                reactions["reaction_pt"]
                .value_counts()
            )

            with self._lock:
                self._derived[key] = counts


        q = (
            str(query or "")
            .strip()
            .casefold()
        )

        rows = counts

        if q:

            mask = (
                rows.index
                .astype(str)
                .str.casefold()
                .str.contains(
                    q,
                    regex=False
                )
            )

            rows = rows[mask]


        result = []

        for label, count in (
            rows.head(
                max(
                    1,
                    min(
                        int(limit),
                        100
                    )
                )
            ).items()
        ):

            result.append({
                "reaction":
                    str(label),

                "reports":
                    int(count),
            })


        return {
            "scope": {
                "type": scope,
                "code": code,
            },

            "query":
                str(query or ""),

            "reactions":
                result,

            "cache": {
                "hit": True,
                "type": "raw-parquet",
            },
        }


    # ========================================================
    # RECHERCHE INVERSE
    #
    # A + effet E -> quels partenaires B ressortent ?
    # ========================================================

    def pair_scan(
        self,
        scope,
        code,
        substance_a,
        reaction,
        role_mode="any"
    ):

        scope, code, _ = self._scope_path(
            scope,
            code
        )

        a_name = " ".join(
            str(substance_a or "")
            .strip()
            .upper()
            .split()
        )

        reaction_query = (
            str(reaction or "")
            .strip()
        )

        role_mode = (
            str(role_mode or "any")
            .strip()
            .lower()
        )


        if not a_name:
            raise ValueError(
                "substance A absente"
            )

        if not reaction_query:
            raise ValueError(
                "effet indésirable absent"
            )


        allowed_modes = {
            "any",
            "suspect_suspect",
            "a_suspect_b_concomitant",
            "a_concomitant_b_suspect",
        }

        if role_mode not in allowed_modes:
            raise ValueError(
                "mode de rôle invalide"
            )


        cache_key = (
            scope,
            code,
            "pair_scan",
            a_name,
            reaction_query.casefold(),
            role_mode,
        )

        with self._lock:
            cached = self._derived.get(
                cache_key
            )

        if cached is not None:
            return cached


        # ----------------------------------------------------
        # PRESENCE DES SUBSTANCES
        # ----------------------------------------------------

        presence = self._pair_presence(
            scope,
            code
        )


        if role_mode == "any":

            a_presence = (
                presence[
                    presence["substance"]
                    == a_name
                ]
                .drop_duplicates(
                    "safety_report_id"
                )
            )

            b_presence = (
                presence[
                    [
                        "safety_report_id",
                        "substance"
                    ]
                ]
                .drop_duplicates()
            )

            role_label = (
                "A et B présents dans le même signalement, "
                "quel que soit le rôle"
            )


        elif role_mode == "suspect_suspect":

            a_presence = (
                presence[
                    (
                        presence["substance"]
                        == a_name
                    )
                    &
                    (
                        presence["role"]
                        == "suspect"
                    )
                ]
                .drop_duplicates(
                    "safety_report_id"
                )
            )

            b_presence = (
                presence[
                    presence["role"]
                    == "suspect"
                ][
                    [
                        "safety_report_id",
                        "substance"
                    ]
                ]
                .drop_duplicates()
            )

            role_label = (
                "A suspecte + B suspecte"
            )


        elif role_mode == "a_suspect_b_concomitant":

            a_presence = (
                presence[
                    (
                        presence["substance"]
                        == a_name
                    )
                    &
                    (
                        presence["role"]
                        == "suspect"
                    )
                ]
                .drop_duplicates(
                    "safety_report_id"
                )
            )

            b_presence = (
                presence[
                    presence["role"]
                    == "concomitant"
                ][
                    [
                        "safety_report_id",
                        "substance"
                    ]
                ]
                .drop_duplicates()
            )

            role_label = (
                "A suspecte + B concomitante"
            )


        else:

            a_presence = (
                presence[
                    (
                        presence["substance"]
                        == a_name
                    )
                    &
                    (
                        presence["role"]
                        == "concomitant"
                    )
                ]
                .drop_duplicates(
                    "safety_report_id"
                )
            )

            b_presence = (
                presence[
                    presence["role"]
                    == "suspect"
                ][
                    [
                        "safety_report_id",
                        "substance"
                    ]
                ]
                .drop_duplicates()
            )

            role_label = (
                "A concomitante + B suspecte"
            )


        a_ids = set(
            a_presence[
                "safety_report_id"
            ]
            .dropna()
            .tolist()
        )

        if not a_ids:
            raise ValueError(
                f"{a_name} absente pour ce rôle"
            )


        # ----------------------------------------------------
        # EFFET MEDDRA EXACT
        # ----------------------------------------------------

        reactions = self._load(
            scope,
            code,
            "reactions",
            [
                "safety_report_id",
                "reaction_pt",
            ]
        ).copy()

        reactions["reaction_pt"] = (
            reactions["reaction_pt"]
            .astype("string")
            .str.strip()
        )

        wanted = (
            reaction_query
            .casefold()
        )

        reaction_mask = (
            reactions["reaction_pt"]
            .notna()
            &
            (
                reactions["reaction_pt"]
                .str.casefold()
                == wanted
            )
        )

        matched = reactions.loc[
            reaction_mask
        ]

        if matched.empty:
            raise ValueError(
                f"réaction introuvable : {reaction_query}"
            )


        canonical_reaction = str(
            matched[
                "reaction_pt"
            ].iloc[0]
        )

        effect_ids = set(
            matched[
                "safety_report_id"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        a_event_ids = (
            a_ids
            & effect_ids
        )

        a_total = len(
            a_ids
        )

        a_event_total = len(
            a_event_ids
        )


        # ----------------------------------------------------
        # CANDIDATS B
        #
        # On calcule tous les partenaires dans une seule passe.
        # ----------------------------------------------------

        b_presence = b_presence[
            b_presence["substance"]
            != a_name
        ].copy()


        # Nombre total de dossiers par B.
        b_total = (
            b_presence
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )


        # Nombre de dossiers B contenant l'effet.
        b_event = (
            b_presence[
                b_presence[
                    "safety_report_id"
                ].isin(effect_ids)
            ]
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )


        # Nombre de dossiers A+B.
        pair_total = (
            b_presence[
                b_presence[
                    "safety_report_id"
                ].isin(a_ids)
            ]
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )


        # Nombre de dossiers A+B contenant l'effet.
        pair_event = (
            b_presence[
                b_presence[
                    "safety_report_id"
                ].isin(a_event_ids)
            ]
            .groupby(
                "substance"
            )[
                "safety_report_id"
            ]
            .nunique()
        )


        # ----------------------------------------------------
        # ROR + IC95
        # ----------------------------------------------------

        def reporting_or(
            event_exposed,
            total_exposed,
            event_reference,
            total_reference
        ):

            aa0 = int(
                event_exposed
            )

            bb0 = int(
                total_exposed
                - event_exposed
            )

            cc0 = int(
                event_reference
            )

            dd0 = int(
                total_reference
                - event_reference
            )


            corrected = any(
                x == 0
                for x in (
                    aa0,
                    bb0,
                    cc0,
                    dd0
                )
            )


            if corrected:

                aa = aa0 + 0.5
                bb = bb0 + 0.5
                cc = cc0 + 0.5
                dd = dd0 + 0.5

            else:

                aa = float(aa0)
                bb = float(bb0)
                cc = float(cc0)
                dd = float(dd0)


            value = (
                aa * dd
                /
                (bb * cc)
            )

            se = math.sqrt(
                1 / aa
                + 1 / bb
                + 1 / cc
                + 1 / dd
            )

            low = math.exp(
                math.log(value)
                - 1.96 * se
            )

            high = math.exp(
                math.log(value)
                + 1.96 * se
            )


            return {
                "value":
                    value,

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "corrected":
                    corrected,
            }


        rows = []


        for b_name, b_n_value in (
            b_total.items()
        ):

            b_n = int(
                b_n_value
            )

            pair_n = int(
                pair_total.get(
                    b_name,
                    0
                )
            )

            pair_e = int(
                pair_event.get(
                    b_name,
                    0
                )
            )


            # Rien à analyser.
            if pair_n < 5:
                continue

            if pair_e < 5:
                continue


            b_e = int(
                b_event.get(
                    b_name,
                    0
                )
            )


            a_without_b_n = (
                a_total
                - pair_n
            )

            a_without_b_e = (
                a_event_total
                - pair_e
            )


            b_without_a_n = (
                b_n
                - pair_n
            )

            b_without_a_e = (
                b_e
                - pair_e
            )


            if (
                a_without_b_n <= 0
                or b_without_a_n <= 0
            ):
                continue


            ror_a = reporting_or(
                pair_e,
                pair_n,
                a_without_b_e,
                a_without_b_n
            )


            ror_b = reporting_or(
                pair_e,
                pair_n,
                b_without_a_e,
                b_without_a_n
            )


            dual = (
                ror_a[
                    "ci95_low"
                ] > 1
                and
                ror_b[
                    "ci95_low"
                ] > 1
            )


            robust = (
                dual
                and pair_n >= 20
                and pair_e >= 10
                and a_without_b_n >= 50
                and b_without_a_n >= 50
                and a_without_b_e >= 5
                and b_without_a_e >= 5
            )


            score = min(
                ror_a[
                    "ci95_low"
                ],
                ror_b[
                    "ci95_low"
                ]
            )


            pair_share = (
                pair_e
                / pair_n
                if pair_n
                else 0
            )

            a_only_share = (
                a_without_b_e
                / a_without_b_n
                if a_without_b_n
                else 0
            )

            b_only_share = (
                b_without_a_e
                / b_without_a_n
                if b_without_a_n
                else 0
            )


            rows.append({

                "substance_b":
                    str(b_name),

                "pair_reports":
                    pair_n,

                "pair_effect_reports":
                    pair_e,

                "pair_effect_share":
                    pair_share,

                "a_without_b_reports":
                    a_without_b_n,

                "a_without_b_effect_reports":
                    a_without_b_e,

                "a_without_b_effect_share":
                    a_only_share,

                "b_without_a_reports":
                    b_without_a_n,

                "b_without_a_effect_reports":
                    b_without_a_e,

                "b_without_a_effect_share":
                    b_only_share,

                "ror_vs_a_without_b":
                    ror_a,

                "ror_vs_b_without_a":
                    ror_b,

                "dual_signal":
                    dual,

                "robust_signal":
                    robust,

                "conservative_score":
                    score,

                "share_of_a_with_b":
                    (
                        pair_n / a_total
                        if a_total
                        else 0
                    ),
            })


        # ----------------------------------------------------
        # CLASSEMENTS
        # ----------------------------------------------------

        robust_rows = [
            x
            for x in rows
            if x[
                "robust_signal"
            ]
        ]

        robust_rows.sort(
            key=lambda x: (
                x[
                    "conservative_score"
                ],
                x[
                    "pair_effect_reports"
                ]
            ),
            reverse=True
        )


        rare_rows = [
            x
            for x in rows
            if (
                x[
                    "dual_signal"
                ]
                and not x[
                    "robust_signal"
                ]
            )
        ]

        rare_rows.sort(
            key=lambda x: (
                x[
                    "conservative_score"
                ],
                x[
                    "pair_effect_reports"
                ]
            ),
            reverse=True
        )


        frequency_rows = sorted(
            rows,
            key=lambda x: (
                x[
                    "pair_effect_reports"
                ],
                x[
                    "pair_effect_share"
                ]
            ),
            reverse=True
        )


        result = {

            "scope": {
                "type":
                    scope,

                "code":
                    code,
            },

            "query": {
                "substance_a":
                    a_name,

                "reaction":
                    canonical_reaction,

                "role_mode":
                    role_mode,

                "role_label":
                    role_label,
            },

            "summary": {
                "a_reports":
                    a_total,

                "a_effect_reports":
                    a_event_total,

                "a_effect_share":
                    (
                        a_event_total
                        / a_total
                        if a_total
                        else 0
                    ),

                "candidates_tested":
                    len(rows),

                "robust_signals":
                    len(
                        robust_rows
                    ),

                "rare_signals":
                    len(
                        rare_rows
                    ),
            },

            "robust_associations":
                robust_rows[:50],

            "rare_associations":
                rare_rows[:50],

            "most_frequent":
                frequency_rows[:30],

            "method": {
                "comparison":
                    "A+B versus A-without-B AND B-without-A",

                "ranking":
                    "minimum lower bound of the two 95% confidence intervals",

                "min_pair_reports":
                    20,

                "min_pair_effect_reports":
                    10,

                "min_comparator_reports":
                    50,

                "min_comparator_effect_reports":
                    5,

                "rare_min_pair_effect_reports":
                    5,

                "multiple_testing_correction":
                    False,

                "causal":
                    False,
            },

            "cache": {
                "hit":
                    True,

                "type":
                    "raw-parquet",
            },
        }


        with self._lock:
            self._derived[
                cache_key
            ] = result

        return result


ENGINE = RawStatsEngine()
