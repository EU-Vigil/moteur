# EU-Vigil — Statistical Engines

Statistical analysis engines developed for EU-Vigil, the European pharmacovigilance data exploration platform.

Project: https://eu-vigil.com · Contact: contact@eu-vigil.com

## English

### Purpose
These Python modules provide statistical and descriptive analysis of pharmacovigilance reports extracted from EudraVigilance. The methods support exploratory analysis of reported adverse drug reactions. They do not establish medical causality.

### Statistical methods
- Reporting Odds Ratio (ROR)
- Proportional Reporting Ratio (PRR)
- Bayesian Confidence Propagation Neural Network (BCPNN)
- Gamma Poisson Shrinker (GPS)
- Descriptive distributions and temporal profiles
- Comparative analysis between populations of reports

### Components
| Module | Description |
|---|---|
| `raw_stats_engine.py` | Main statistical engine |
| `signal_methods.py` | Statistical signal detection |
| `fast_profiles.py` | Optimized descriptive profiles |
| `compare_disk_cache.py` | Persistent comparison caching |
| `stats_memo.py` | Statistical memoization |

### Limitations
Spontaneous reporting databases are subject to underreporting, reporting biases, incomplete information and confounding. Statistical signals are hypotheses requiring further investigation, not evidence of a causal relationship. The source code is provided separately from the underlying pharmacovigilance datasets.

## Français

### Objectif
Ces modules Python constituent les moteurs d’analyse statistique d’EU-Vigil, plateforme d’exploration des données européennes de pharmacovigilance. Ils permettent d’analyser les signalements d’effets indésirables extraits d’EudraVigilance.

### Méthodes statistiques
- Reporting Odds Ratio (ROR)
- Proportional Reporting Ratio (PRR)
- Bayesian Confidence Propagation Neural Network (BCPNN)
- Gamma Poisson Shrinker (GPS)
- Distributions descriptives et évolutions temporelles
- Comparaison de populations de signalements

### Limites
Les bases de pharmacovigilance reposent sur des déclarations spontanées et présentent notamment des biais de notification, des données manquantes et des facteurs de confusion. Les données scientifiques sources ne sont pas distribuées dans ce dépôt.

## Licence / License
Copyright © 2026 EU-Vigil.

Ce projet est distribué sous licence Apache License, Version 2.0. Vous pouvez utiliser, modifier et redistribuer les moteurs statistiques conformément aux conditions de cette licence.

This project is licensed under the Apache License, Version 2.0. See `LICENSE` for the full license text.
