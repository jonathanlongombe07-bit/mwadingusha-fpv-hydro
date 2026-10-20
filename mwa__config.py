"""Chargement de la configuration (params.yaml) sous forme d'objet à attributs."""
from pathlib import Path
from types import SimpleNamespace
import copy
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "params.yaml"
    with open(path, encoding="utf-8") as f:
        return _ns(yaml.safe_load(f))


def replace(cfg, **sections):
    """Copie profonde avec surcharge : replace(cfg, hydro={'Q_tur_max_op': 65})."""
    new = copy.deepcopy(cfg)
    for sec, vals in sections.items():
        for k, v in vals.items():
            setattr(getattr(new, sec), k, v)
    return new
