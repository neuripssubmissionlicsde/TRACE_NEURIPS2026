"""
Combina os dados sintéticos de todos os tickers da pasta djia_2019_2020_bear_amplified.
Para cada índice de geração, gera um único CSV com todos os tickers juntos,
salvando na própria pasta. Duplicatas são removidas dos arquivos finais.
"""

import os
import glob
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "results" / "djia_2019_2020_bear_amplified"

# Pastas que não são tickers
NON_TICKER_DIRS = {"finrl_output", "plots_finrl", "postprocess"}

def get_ticker_dirs(base: Path) -> list[Path]:
    return sorted(
        p for p in base.iterdir()
        if p.is_dir() and p.name not in NON_TICKER_DIRS
    )

def get_generation_indices(ticker_dir: Path) -> set[int]:
    indices = set()
    for f in ticker_dir.glob("synthetic_data_*.csv"):
        stem = f.stem.replace("synthetic_data_", "")
        if stem.isdigit():
            indices.add(int(stem))
    return indices

def combine_generation(gen_idx: int, ticker_dirs: list[Path]) -> pd.DataFrame | None:
    frames = []
    for tdir in ticker_dirs:
        fpath = tdir / f"synthetic_data_{gen_idx}.csv"
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    if before != after:
        print(f"  gen {gen_idx}: removidas {before - after} duplicatas")
    return combined

def combine_special(name: str, ticker_dirs: list[Path]) -> pd.DataFrame | None:
    frames = []
    for tdir in ticker_dirs:
        fpath = tdir / f"synthetic_data_{name}.csv"
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    if before != after:
        print(f"  {name}: removidas {before - after} duplicatas")
    return combined


def main():
    ticker_dirs = get_ticker_dirs(BASE_DIR)
    print(f"Tickers encontrados ({len(ticker_dirs)}): {[d.name for d in ticker_dirs]}")

    # Descobrir todos os índices numéricos disponíveis
    all_indices = get_generation_indices(ticker_dirs[0])
    print(f"Gerações numéricas encontradas: {min(all_indices)} a {max(all_indices)} ({len(all_indices)} total)")

    # Criar pasta de saída dentro do BASE_DIR
    out_dir = BASE_DIR / "combined_generations"
    out_dir.mkdir(exist_ok=True)

    # Processar cada geração numérica
    for gen_idx in sorted(all_indices):
        out_path = out_dir / f"combined_gen_{gen_idx:04d}.csv"
        if out_path.exists():
            continue  # já existe, pular
        df = combine_generation(gen_idx, ticker_dirs)
        if df is not None:
            df.to_csv(out_path, index=False)
        if gen_idx % 100 == 0:
            print(f"  Processado geração {gen_idx}...")

    print(f"Gerações numéricas: {len(all_indices)} arquivos criados em {out_dir}")

    # Processar arquivo especial causal_sde
    df_causal = combine_special("causal_sde", ticker_dirs)
    if df_causal is not None:
        out_causal = out_dir / "combined_causal_sde.csv"
        df_causal.to_csv(out_causal, index=False)
        print(f"Arquivo especial: combined_causal_sde.csv criado ({len(df_causal)} linhas)")

    print("Concluído!")


if __name__ == "__main__":
    main()
