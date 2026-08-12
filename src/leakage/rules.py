"""Domain rules for target-proxy detection.

Rules are keyword fragments matched against normalised column names. Loaded
from ``config/leakage_rules.yaml`` when present, otherwise from the built-in
defaults below.

'generic' deliberately contains no clinical terms. A run with
``--domain generic`` will not warn about glucose or insulin columns; use
``--domain diabetes`` when the target actually is a glycaemic outcome.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml

BUILTIN_RULES: Dict[str, List[str]] = {
    "generic": [
        "target", "label", "ground truth", "true label", "outcome copy",
        "y true", "y pred", "prediction", "predicted", "score",
    ],
    "diabetes": [
        "glucose", "hba1c", "a1c", "glycated haemoglobin", "glycated hemoglobin",
        "insulin", "c peptide", "ogtt", "glucose tolerance", "fructosamine",
        "metformin", "glipizide", "glyburide", "pioglitazone", "sitagliptin",
        "liraglutide", "semaglutide", "antidiabetic", "diabetes diagnosis",
        "diabetes code", "diabetes medication",
    ],
    "mortality": [
        "death", "died", "deceased", "expire", "date of death", "dod",
        "discharge disposition", "hospice", "time of death",
    ],
    "readmission": [
        "readmit", "readmission", "days to readmission", "next admission",
        "return visit", "bounce back",
    ],
    "sepsis": [
        "sofa", "qsofa", "sirs", "vasopressor", "norepinephrine", "lactate",
        "blood culture", "antibiotic start",
    ],
}

ICD_PATTERNS: Dict[str, List[str]] = {
    "diabetes": [r"^250(\.\d+)?$", r"^E0[89]", r"^E1[0-3]"],
}


def normalise(name: str) -> str:
    return re.sub(r"[\s_\-./]+", " ", str(name).strip().lower())


def load_rules(path: str | Path | None = None) -> Dict[str, List[str]]:
    rules = {k: list(v) for k, v in BUILTIN_RULES.items()}
    if path is None:
        return rules
    path = Path(path)
    if not path.exists():
        return rules
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    for domain, spec in loaded.items():
        terms: List[str] = []
        if isinstance(spec, dict):
            for value in spec.values():
                if isinstance(value, list):
                    terms.extend(str(v) for v in value)
        elif isinstance(spec, list):
            terms = [str(v) for v in spec]
        rules[str(domain).lower()] = [normalise(t) for t in terms]
    return rules


def proxy_terms(domain: str, rules: Dict[str, List[str]] | None = None) -> List[str]:
    """Terms for ``domain``, always including the generic set."""
    rules = rules or BUILTIN_RULES
    domain = (domain or "generic").lower()
    terms = list(rules.get("generic", []))
    if domain != "generic":
        terms.extend(rules.get(domain, []))
    return sorted({normalise(t) for t in terms if t})
