"""Funciones de manejo/redifinición del target para controlar fuga.
- redefine_target: crea un nuevo target como diferencia entre target y base.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import pandas as pd

METADATA_DIR = Path("outputs/metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def redefine_target(df: pd.DataFrame, target: str, base_col: str, new_name: str | None = None) -> pd.DataFrame:
    """Redefine el target como target - base_col y persiste metadata para reconstrucción.
    Si base_col no existe o target no existe, devuelve el df sin cambios.
    """
    if target not in df.columns or base_col not in df.columns:
        return df
    new_name = new_name or (target.strip() + "_RED")
    df[new_name] = df[target] - df[base_col]
    mapping: Dict[str, str] = {
        "original_target": target,
        "base_col": base_col,
        "redefined_target": new_name,
        "formula": f"{new_name} = {target} - {base_col}",
    }
    (METADATA_DIR / "target_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def reconstruct_target(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el target original si existe metadata y columnas necesarias."""
    meta_path = METADATA_DIR / "target_mapping.json"
    if not meta_path.exists():
        return df
    try:
        mapping = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return df
    new_name = mapping.get("redefined_target")
    base_col = mapping.get("base_col")
    original = mapping.get("original_target")
    if new_name and base_col and original and new_name in df.columns and base_col in df.columns:
        df[original] = df[new_name] + df[base_col]
    return df
