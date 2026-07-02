from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_SRC = ROOT / "src"
sys.path.insert(0, str(PY_SRC))

project = "contdid"
author = "Brantly Callaway, Andrew Goodman-Bacon, and Pedro H. C. Sant'Anna"
copyright = "2026, contdid contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "automation"]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
autodoc_typehints = "description"
nitpicky = False
