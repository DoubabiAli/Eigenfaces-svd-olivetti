"""Utilitaires communs : chargement de la base Olivetti et dossier de sortie."""
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def load_olivetti():
    """Retourne images (400, 64, 64) dans [0, 1], X (400, 4096), labels (400,)."""
    d = fetch_olivetti_faces(shuffle=False)
    return d.images, d.data, d.target
