#!/usr/bin/env python3
"""
Genera un dataset histórico multi-temporada de la NBA, reutilizando el flujo
existente de extracción y enriquecimiento de datos.

Funciones clave:
- Descarga partidos por temporada (Regular y opcionalmente Playoffs)
- Obtiene estadísticas avanzadas por equipo por temporada
- Une, calcula L10 por equipo, días de descanso, B2B y distancias de viaje
- Guarda un Parquet consolidado en ML/data/raw/
- (Opcional) llama al procesador final para generar el dataset con features diferenciales

Uso:
  python -m ML.preprocessing.build_historical_dataset \
    --seasons 2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 \
    --include-playoffs \
    --output nba_games_raw_historical.parquet \
    --also-generate-final

Si no se especifican temporadas, se usa la constante SEASONS de fetch_nba_data.py
"""
import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd

# Asegurar imports relativos desde el paquete ML
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Importar utilidades del proyecto
from .fetch_nba_data import NBADataFetcher, SEASONS  # noqa: E402

# Procesador final (opcional)
try:
    from .process_final_dataset import NBAFinalProcessor  # noqa: E402
    HAS_FINAL_PROCESSOR = True
except Exception:  # pragma: no cover
    HAS_FINAL_PROCESSOR = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un dataset histórico multi-temporada de la NBA"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=",".join(SEASONS),
        help="Lista de temporadas separadas por coma en formato YYYY-YY. Por defecto: las definidas en SEASONS",
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Incluye partidos de Playoffs además de Regular Season",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nba_games_raw_historical.parquet",
        help="Nombre del archivo Parquet de salida en ML/data/raw/",
    )
    parser.add_argument(
        "--also-generate-final",
        action="store_true",
        help="Tras generar el crudo histórico, ejecuta el procesador final para crear el dataset con features diferenciales",
    )
    return parser.parse_args()


def build_historical_dataset(seasons: List[str], include_playoffs: bool) -> pd.DataFrame:
    fetcher = NBADataFetcher()

    all_games = []
    for season in seasons:
        # 1) Juegos temporada regular
        games_regular = fetcher.get_season_games(season=season, season_type="Regular Season")
        team_stats_regular = fetcher.get_team_stats(season=season, season_type="Regular Season")

        if games_regular is not None and not games_regular.empty:
            processed_regular = fetcher.process_games_data(games_regular, team_stats_regular)
            processed_regular = fetcher.calculate_rest_days_and_b2b(processed_regular)
            processed_regular = fetcher.calculate_travel_distances(processed_regular)
            all_games.append(processed_regular)

        # 2) Playoffs (opcional)
        if include_playoffs:
            games_playoffs = fetcher.get_season_games(season=season, season_type="Playoffs")
            team_stats_playoffs = fetcher.get_team_stats(season=season, season_type="Playoffs")

            if games_playoffs is not None and not games_playoffs.empty:
                processed_playoffs = fetcher.process_games_data(games_playoffs, team_stats_playoffs)
                processed_playoffs = fetcher.calculate_rest_days_and_b2b(processed_playoffs)
                processed_playoffs = fetcher.calculate_travel_distances(processed_playoffs)
                all_games.append(processed_playoffs)

    if not all_games:
        print("❌ No se obtuvieron datos para las temporadas solicitadas")
        return pd.DataFrame()

    historical_df = pd.concat(all_games, ignore_index=True)

    # Orden final por fecha
    if "GAME_DATE" in historical_df.columns:
        historical_df = historical_df.sort_values("GAME_DATE").reset_index(drop=True)

    # Guardar CSV adicional para inspección rápida
    csv_path = RAW_DATA_DIR / "nba_games_raw_historical.csv"
    try:
        historical_df.to_csv(csv_path, index=False)
        print(f"📄 CSV histórico guardado en {csv_path}")
    except Exception as e:
        print(f"⚠️  No se pudo guardar el CSV histórico: {e}")

    return historical_df


def main():
    args = parse_args()

    # Parseo de temporadas
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print("=" * 80)
    print("Construyendo dataset histórico de la NBA")
    print("Temporadas:", seasons)
    print("Incluir Playoffs:", args.include_playoffs)
    print("=" * 80)

    df_hist = build_historical_dataset(seasons, args.include_playoffs)
    if df_hist.empty:
        print("❌ No se generó dataset histórico")
        sys.exit(1)

    # Guardar Parquet en raw/
    out_parquet = RAW_DATA_DIR / args.output
    try:
        df_hist.to_parquet(out_parquet, index=False)
        print(f"✅ Dataset histórico guardado en {out_parquet}")
    except Exception as e:
        print(f"❌ Error guardando Parquet: {e}")
        sys.exit(1)

    # Opcional: generar dataset final con features diferenciales
    if args.also_generate_final:
        if not HAS_FINAL_PROCESSOR:
            print("⚠️  El procesador final no está disponible. Omite --also-generate-final o asegúrate de que process_final_dataset.py existe.")
            sys.exit(0)
        try:
            processor = NBAFinalProcessor(input_path=str(out_parquet), output_dir=str(PROCESSED_DATA_DIR))
            processor.load_data()
            processor.calculate_differential_features()
            final_path = processor.generate_final_dataset()
            print(f"🏁 Dataset final generado en {final_path}")
        except Exception as e:
            print(f"⚠️  Error generando el dataset final: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
