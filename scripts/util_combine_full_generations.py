"""
Combina os dados sintéticos de todos os tickers da pasta djia_2019_2020_bear_amplified_full
(geração com período completo 2019-01-01 → 2020-12-31, 50 samples).

Para cada índice de geração, gera um único CSV com todos os tickers juntos,
salvando na pasta combined_generations/. Duplicatas são removidas.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "results" / "djia_2019_2020_bear_amplified_full"

NON_TICKER_DIRS = {"finrl_output", "plots_finrl", "postprocess", "combined_generations"}


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


def combine_files(name: str, ticker_dirs: list[Path]) -> pd.DataFrame | None:
    frames = []
    for tdir in ticker_dirs:
        fpath = tdir / f"synthetic_data_{name}.csv"
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "tic"], keep="first")
    after = len(combined)
    if before != after:
        print(f"  {name}: removidas {before - after} duplicatas")
    return combined


def main():
    ticker_dirs = get_ticker_dirs(BASE_DIR)
    print(f"Tickers encontrados ({len(ticker_dirs)}): {[d.name for d in ticker_dirs]}")

    if not ticker_dirs:
        print(f"ERRO: nenhum ticker encontrado em {BASE_DIR}")
        return

    all_indices = get_generation_indices(ticker_dirs[0])
    print(f"Gerações numéricas: {sorted(all_indices)} ({len(all_indices)} total)")

    out_dir = BASE_DIR / "combined_generations"
    out_dir.mkdir(exist_ok=True)

    for gen_idx in sorted(all_indices):
        out_path = out_dir / f"combined_gen_{gen_idx:04d}.csv"
        if out_path.exists():
            continue
        df = combine_files(str(gen_idx), ticker_dirs)
        if df is not None:
            df = df.sort_values(["date", "tic"]).reset_index(drop=True)
            df.to_csv(out_path, index=False)
            print(f"  gen {gen_idx}: {len(df)} linhas, {df['tic'].nunique()} tickers, "
                  f"datas {df['date'].min()} → {df['date'].max()}")

    df_causal = combine_files("causal_sde", ticker_dirs)
    if df_causal is not None:
        df_causal = df_causal.sort_values(["date", "tic"]).reset_index(drop=True) \
            if "date" in df_causal.columns else df_causal
        out_causal = out_dir / "combined_causal_sde.csv"
        df_causal.to_csv(out_causal, index=False)
        print(f"causal_sde: {len(df_causal)} linhas")

    print(f"\nConcluído. {len(all_indices)} arquivos em {out_dir}")


if __name__ == "__main__":
    main()
