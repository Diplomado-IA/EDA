#!/usr/bin/env python3
"""
Runner único: ejecuta el flujo completo (EDA -> Preproceso -> Entrenamiento -> Evaluación -> XAI)
Uso:
  python scripts/run_all.py [--stop-on-error] [--skip-train] [--skip-eval] [--skip-xai]
Notas:
  - Establece PYTHONPATH al raíz del proyecto para resolver imports (src/*).
  - Ejecuta condicionalmente cada paso si el script existe.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = []  # Consolidado: pipeline se ejecuta inline


def run_step(name: str, cmd: list[str], desc: str, env: dict, stop_on_error: bool, skip: bool) -> bool:
    # Detectar ruta del script .py dentro del comando
    script_path = None
    for c in cmd:
        if isinstance(c, str) and c.endswith('.py'):
            p = Path(c)
            script_path = p if p.is_absolute() else (PROJECT_ROOT / p)
            break
    if skip:
        print(f"[SKIP] {name}: omitiendo por bandera")
        return True
    if script_path is not None and not script_path.exists():
        print(f"[SKIP] {name}: {script_path.name} no existe")
        return True
    print("\n" + "="*80)
    print(f"➡️  {desc}")
    print("="*80)
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
        if res.returncode != 0:
            print(f"✖ {name} falló con código {res.returncode}")
            if stop_on_error:
                return False
        else:
            print(f"✔ {name} completado")
        return True
    except Exception as e:
        print(f"✖ Error ejecutando {name}: {e}")
        return not stop_on_error


def list_artifacts():
    print("\n=== Artifacts (data/processed) ===")
    dp = PROJECT_ROOT / "data" / "processed"
    if dp.exists():
        for p in sorted(dp.rglob("*")):
            if p.is_file():
                try:
                    sz = p.stat().st_size // 1024
                    print(f"{p.relative_to(PROJECT_ROOT)}\t{sz} KB")
                except Exception:
                    pass
    print("\n=== Artifacts (outputs, models, reports) ===")
    for base in [PROJECT_ROOT/"outputs", PROJECT_ROOT/"models", PROJECT_ROOT/"reports"]:
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    try:
                        sz = p.stat().st_size // 1024
                        print(f"{p.relative_to(PROJECT_ROOT)}\t{sz} KB")
                    except Exception:
                        pass


def main():
    stop_on_error = "--stop-on-error" in sys.argv
    skip_train = "--skip-train" in sys.argv
    skip_eval = "--skip-eval" in sys.argv
    skip_xai = "--skip-xai" in sys.argv

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    ok = True
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.pipeline import MLPipeline
        print("\n" + "="*80)
        print("➡️  Ejecutando pipeline (EDA+Preproceso)")
        print("="*80)
        MLPipeline().run_full_pipeline()
        print("✔ pipeline completado")
        ok = True
    except Exception as e:
        print(f"✖ pipeline falló: {e}")
        ok = False

    list_artifacts()
    print("\n" + ("✅ Flujo completado" if ok else "⚠️ Flujo completado con errores"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
