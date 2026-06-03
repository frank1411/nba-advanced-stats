"""
Módulo de base de datos SQLite para datos de la NBA.

Proporciona una interfaz centralizada para leer y escribir datos
de juegos de la NBA, reemplazando los archivos .parquet.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Ruta por defecto: ML/data/nba.db
BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = str(DB_DIR / "nba.db")

# Columnas conocidas del dataset raw (50 columnas)
RAW_COLUMNS = [
    "GAME_ID", "GAME_DATE", "SEASON", "SEASON_STR", "SEASON_TYPE",
    "TEAM_ID_HOME", "TEAM_ABBREVIATION_HOME", "PTS_HOME", "WL_HOME",
    "OFF_RATING_HOME", "DEF_RATING_HOME", "NET_RATING_HOME", "PACE_HOME",
    "W_PCT_HOME", "GP_HOME", "W_HOME", "L_HOME", "HOME_POINT_DIFF",
    "TEAM_ID_AWAY", "TEAM_ABBREVIATION_AWAY", "PTS_AWAY", "WL_AWAY",
    "OFF_RATING_AWAY", "DEF_RATING_AWAY", "NET_RATING_AWAY", "PACE_AWAY",
    "W_PCT_AWAY", "GP_AWAY", "W_AWAY", "L_AWAY", "AWAY_POINT_DIFF",
    "POINT_DIFF", "PTS_L10_AWAY", "PTS_L10_HOME", "OFF_RATING_L10_AWAY",
    "OFF_RATING_L10_HOME", "DEF_RATING_L10_AWAY", "DEF_RATING_L10_HOME",
    "NET_RATING_L10_AWAY", "NET_RATING_L10_HOME", "PACE_L10_AWAY",
    "PACE_L10_HOME", "POINT_DIFF_L10_AWAY", "POINT_DIFF_L10_HOME",
    "DAYS_REST_HOME", "IS_B2B_HOME", "DAYS_REST_AWAY", "IS_B2B_AWAY",
    "TRAVEL_DISTANCE_HOME", "TRAVEL_DISTANCE_AWAY",
]

# Columnas adicionales del dataset final (sobre las raw)
FINAL_EXTRA_COLUMNS = [
    "PTS_DIFF", "PTS_PCT_DIFF", "OFF_RATING_DIFF", "OFF_RATING_PCT_DIFF",
    "DEF_RATING_DIFF", "DEF_RATING_PCT_DIFF", "NET_RATING_DIFF",
    "NET_RATING_PCT_DIFF", "PACE_DIFF", "PACE_PCT_DIFF", "W_PCT_DIFF",
    "W_PCT_PCT_DIFF", "GP_DIFF", "W_DIFF", "L_DIFF", "PTS_L10_DIFF",
    "PTS_L10_PCT_DIFF", "OFF_RATING_L10_DIFF", "OFF_RATING_L10_PCT_DIFF",
    "DEF_RATING_L10_DIFF", "DEF_RATING_L10_PCT_DIFF", "NET_RATING_L10_DIFF",
    "NET_RATING_L10_PCT_DIFF", "PACE_L10_DIFF", "PACE_L10_PCT_DIFF",
    "POINT_DIFF_L10_DIFF", "POINT_DIFF_L10_PCT_DIFF", "DAYS_REST_DIFF",
    "B2B_ADVANTAGE", "TRAVEL_DISTANCE_DIFF", "OFF_RATING_HOME_TREND",
    "OFF_RATING_AWAY_TREND", "OFF_RATING_TREND_DIFF", "DEF_RATING_HOME_TREND",
    "DEF_RATING_AWAY_TREND", "DEF_RATING_TREND_DIFF", "NET_RATING_HOME_TREND",
    "NET_RATING_AWAY_TREND", "NET_RATING_TREND_DIFF", "PACE_HOME_TREND",
    "PACE_AWAY_TREND", "PACE_TREND_DIFF", "WIN_STREAK_DIFF",
    "TARGET", "POINT_DIFF_TARGET",
    # Betting tracking
    "LINE_OU_TOTAL", "LINE_OU_HOME", "LINE_OU_AWAY",
    "PREDICTION_OU_TOTAL", "PREDICTION_OU_HOME", "PREDICTION_OU_AWAY",
    "RECOMMENDATION_OU_TOTAL", "RECOMMENDATION_OU_HOME", "RECOMMENDATION_OU_AWAY",
    "RESULT_OU_TOTAL", "RESULT_OU_HOME", "RESULT_OU_AWAY",
    "CONFIDENCE_OU_TOTAL", "CONFIDENCE_OU_HOME", "CONFIDENCE_OU_AWAY",
]

FINAL_COLUMNS = RAW_COLUMNS + FINAL_EXTRA_COLUMNS


def _infer_sqlite_type(dtype) -> str:
    """Infiere el tipo de SQLite a partir del dtype de pandas."""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TEXT"
    else:
        return "TEXT"


class NBADatabase:
    """
    Gestor de base de datos SQLite para datos de la NBA.

    Uso:
        with NBADatabase() as db:
            df = db.get_final_games()
            # o insertar:
            db.insert_raw_games(df)
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._conn = None
        self._open()
        self._setup_connection()

    def _open(self):
        """Abre la conexión a la base de datos."""
        self._conn = sqlite3.connect(
            self._db_path,
            timeout=30,
            isolation_level=None,  # autocommit mode, we manage transactions manually
        )
        # Habilitar WAL mode para mejor concurrencia
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"Conexión abierta: {self._db_path}")

    def _setup_connection(self):
        """Configura la conexión (WAL, foreign keys)."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-2000")  # 2MB cache

    def close(self):
        """Cierra la conexión."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Conexión cerrada")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def exists(self) -> bool:
        return Path(self._db_path).exists()

    # ----------------------------------------------------------------
    # Schema Management
    # ----------------------------------------------------------------

    def create_tables(self):
        """Crea las tablas e índices si no existen."""
        self._create_raw_table()
        self._create_final_table()
        self._create_migration_log_table()
        self._create_indexes()
        logger.info("Tablas e índices creados/verificados")

    def _create_raw_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS nba_games_raw (
            GAME_ID TEXT PRIMARY KEY,
            GAME_DATE TEXT,
            SEASON INTEGER,
            SEASON_STR TEXT,
            SEASON_TYPE TEXT,
            TEAM_ID_HOME INTEGER,
            TEAM_ABBREVIATION_HOME TEXT,
            PTS_HOME REAL,
            WL_HOME TEXT,
            OFF_RATING_HOME REAL,
            DEF_RATING_HOME REAL,
            NET_RATING_HOME REAL,
            PACE_HOME REAL,
            W_PCT_HOME REAL,
            GP_HOME REAL,
            W_HOME REAL,
            L_HOME REAL,
            HOME_POINT_DIFF REAL,
            TEAM_ID_AWAY INTEGER,
            TEAM_ABBREVIATION_AWAY TEXT,
            PTS_AWAY REAL,
            WL_AWAY TEXT,
            OFF_RATING_AWAY REAL,
            DEF_RATING_AWAY REAL,
            NET_RATING_AWAY REAL,
            PACE_AWAY REAL,
            W_PCT_AWAY REAL,
            GP_AWAY REAL,
            W_AWAY REAL,
            L_AWAY REAL,
            AWAY_POINT_DIFF REAL,
            POINT_DIFF REAL,
            PTS_L10_AWAY REAL,
            PTS_L10_HOME REAL,
            OFF_RATING_L10_AWAY REAL,
            OFF_RATING_L10_HOME REAL,
            DEF_RATING_L10_AWAY REAL,
            DEF_RATING_L10_HOME REAL,
            NET_RATING_L10_AWAY REAL,
            NET_RATING_L10_HOME REAL,
            PACE_L10_AWAY REAL,
            PACE_L10_HOME REAL,
            POINT_DIFF_L10_AWAY REAL,
            POINT_DIFF_L10_HOME REAL,
            DAYS_REST_HOME REAL,
            IS_B2B_HOME INTEGER,
            DAYS_REST_AWAY REAL,
            IS_B2B_AWAY INTEGER,
            TRAVEL_DISTANCE_HOME REAL,
            TRAVEL_DISTANCE_AWAY REAL
        )
        """
        self._conn.execute(sql)

    def _create_final_table(self):
        """Crea la tabla nba_games_final con todas las columnas raw + diferenciales."""
        cols = [
            f"{col} REAL" if col not in ("GAME_ID",) else f"{col} TEXT PRIMARY KEY"
            for col in FINAL_COLUMNS
        ]
        # Fix specific column types
        cols_fixed = []
        for col in FINAL_COLUMNS:
            if col == "GAME_ID":
                cols_fixed.append("GAME_ID TEXT PRIMARY KEY")
            elif col in (
                "GAME_DATE", "SEASON_STR", "SEASON_TYPE",
                "TEAM_ABBREVIATION_HOME", "TEAM_ABBREVIATION_AWAY",
                "WL_HOME", "WL_AWAY",
                "RESULT_OU_TOTAL", "RESULT_OU_HOME", "RESULT_OU_AWAY",
            ):
                cols_fixed.append(f"{col} TEXT")
            elif col in ("SEASON", "TEAM_ID_HOME", "TEAM_ID_AWAY", "IS_B2B_HOME",
                         "IS_B2B_AWAY", "B2B_ADVANTAGE", "TARGET",
                         "POINT_DIFF_TARGET"):
                cols_fixed.append(f"{col} INTEGER")
            else:
                cols_fixed.append(f"{col} REAL")

        cols_sql = ",\n            ".join(cols_fixed)
        sql = f"""
        CREATE TABLE IF NOT EXISTS nba_games_final (
            {cols_sql}
        )
        """
        self._conn.execute(sql)

    def _create_migration_log_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            source TEXT,
            row_count INTEGER,
            status TEXT DEFAULT 'pending',
            started_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            error_msg TEXT
        )
        """
        self._conn.execute(sql)

    def _create_indexes(self):
        indexes = [
            ("idx_raw_game_date", "nba_games_raw", "GAME_DATE"),
            ("idx_raw_season", "nba_games_raw", "SEASON"),
            ("idx_raw_home_team", "nba_games_raw", "TEAM_ABBREVIATION_HOME"),
            ("idx_raw_away_team", "nba_games_raw", "TEAM_ABBREVIATION_AWAY"),
            ("idx_raw_season_type", "nba_games_raw", "SEASON_TYPE"),
            ("idx_raw_date_season", "nba_games_raw", "GAME_DATE, SEASON"),
            ("idx_final_game_date", "nba_games_final", "GAME_DATE"),
            ("idx_final_season", "nba_games_final", "SEASON"),
            ("idx_final_home_team", "nba_games_final", "TEAM_ABBREVIATION_HOME"),
            ("idx_final_away_team", "nba_games_final", "TEAM_ABBREVIATION_AWAY"),
            ("idx_final_target", "nba_games_final", "TARGET"),
            ("idx_final_date_season", "nba_games_final", "GAME_DATE, SEASON"),
        ]
        for idx_name, table, col in indexes:
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col})"
            )

    def drop_all_tables(self):
        """Elimina todas las tablas (para testing/re-migración)."""
        self._conn.execute("DROP TABLE IF EXISTS nba_games_raw")
        self._conn.execute("DROP TABLE IF EXISTS nba_games_final")
        self._conn.execute("DROP TABLE IF EXISTS migration_log")
        logger.warning("Todas las tablas eliminadas")

    # ----------------------------------------------------------------
    # Data Ingestion
    # ----------------------------------------------------------------

    def insert_raw_games(self, df: pd.DataFrame, if_exists: str = "replace"):
        """
        Inserta juegos raw en la base de datos.

        Args:
            df: DataFrame con los datos de juegos.
            if_exists: 'replace' borra y reinserta; 'append' hace upsert.
        """
        if df.empty:
            logger.warning("DataFrame vacío, no se insertan datos")
            return 0

        df_clean = df.copy()
        # Convertir datetime a string para SQLite
        if "GAME_DATE" in df_clean.columns:
            df_clean["GAME_DATE"] = pd.to_datetime(df_clean["GAME_DATE"]).dt.strftime(
                "%Y-%m-%d"
            )
        # Convertir bool a int
        for col in ["IS_B2B_HOME", "IS_B2B_AWAY"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(float).fillna(0).astype(int)

        if if_exists == "replace":
            self._conn.execute("DELETE FROM nba_games_raw")
            logger.info("Tabla nba_games_raw limpiada para reemplazo")

        # Filtrar solo columnas que existen en la tabla
        available_cols = [c for c in RAW_COLUMNS if c in df_clean.columns]
        df_insert = df_clean[available_cols]

        # Insertar en chunks para evitar "too many SQL variables"
        chunk_size = 500
        total = 0
        for start in range(0, len(df_insert), chunk_size):
            chunk = df_insert.iloc[start:start + chunk_size]
            chunk.to_sql(
                "nba_games_raw", self._conn, if_exists="append", index=False,
            )
            total += len(chunk)
        count = total
        logger.info(f"Insertados {count} juegos en nba_games_raw ({if_exists})")
        return count

    def insert_final_games(self, df: pd.DataFrame, if_exists: str = "replace"):
        """
        Inserta juegos procesados (final) en la base de datos.

        Args:
            df: DataFrame con los datos procesados.
            if_exists: 'replace' borra y reinserta; 'append' hace upsert.
        """
        if df.empty:
            logger.warning("DataFrame vacío, no se insertan datos")
            return 0

        df_clean = df.copy()
        if "GAME_DATE" in df_clean.columns:
            df_clean["GAME_DATE"] = pd.to_datetime(df_clean["GAME_DATE"]).dt.strftime(
                "%Y-%m-%d"
            )

        if if_exists == "replace":
            self._conn.execute("DELETE FROM nba_games_final")
            logger.info("Tabla nba_games_final limpiada para reemplazo")

        available_cols = [c for c in FINAL_COLUMNS if c in df_clean.columns]
        df_insert = df_clean[available_cols]

        # Insertar en chunks para evitar "too many SQL variables"
        chunk_size = 500
        total = 0
        for start in range(0, len(df_insert), chunk_size):
            chunk = df_insert.iloc[start:start + chunk_size]
            chunk.to_sql(
                "nba_games_final", self._conn, if_exists="append", index=False,
            )
            total += len(chunk)
        count = total
        logger.info(f"Insertados {count} juegos en nba_games_final ({if_exists})")
        return count

    def bulk_upsert(self, table: str, df: pd.DataFrame, key_column: str = "GAME_ID"):
        """
        Inserta o reemplaza filas usando INSERT OR REPLACE.

        Args:
            table: Nombre de la tabla ('nba_games_raw' o 'nba_games_final').
            df: DataFrame con los datos.
            key_column: Columna primaria (default GAME_ID).
        """
        if df.empty:
            return 0

        df_clean = df.copy()
        if "GAME_DATE" in df_clean.columns:
            df_clean["GAME_DATE"] = pd.to_datetime(df_clean["GAME_DATE"]).dt.strftime(
                "%Y-%m-%d"
            )

        df_clean.to_sql(table, self._conn, if_exists="append", index=False,
                        method="multi")
        count = len(df_clean)
        logger.info(f"Insertados {count} registros en {table}")
        return count

    # ----------------------------------------------------------------
    # Query Methods
    # ----------------------------------------------------------------

    def get_raw_games(self, where: str = "", params: tuple = ()) -> pd.DataFrame:
        """
        Obtiene todos los juegos raw.

        Args:
            where: Cláusula WHERE opcional (sin la palabra WHERE).
            params: Parámetros para la consulta.
        """
        sql = "SELECT * FROM nba_games_raw"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY GAME_DATE"
        df = pd.read_sql_query(sql, self._conn, params=params)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    def get_final_games(self, where: str = "", params: tuple = ()) -> pd.DataFrame:
        """
        Obtiene todos los juegos procesados (final).

        Args:
            where: Cláusula WHERE opcional.
            params: Parámetros para la consulta.
        """
        sql = "SELECT * FROM nba_games_final"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY GAME_DATE"
        df = pd.read_sql_query(sql, self._conn, params=params)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    def get_games_by_date_range(
        self, start_date: str, end_date: str, table: str = "final"
    ) -> pd.DataFrame:
        """Obtiene juegos en un rango de fechas."""
        where = "GAME_DATE >= ? AND GAME_DATE <= ?"
        params = (start_date, end_date)
        if table == "raw":
            return self.get_raw_games(where, params)
        return self.get_final_games(where, params)

    def get_games_by_season(
        self, season: int, table: str = "final"
    ) -> pd.DataFrame:
        """Obtiene juegos de una temporada específica."""
        where = "SEASON = ?"
        if table == "raw":
            return self.get_raw_games(where, (season,))
        return self.get_final_games(where, (season,))

    def get_games_by_team(
        self, team_abbrev: str, table: str = "final"
    ) -> pd.DataFrame:
        """Obtiene todos los juegos de un equipo (local o visitante)."""
        where = "TEAM_ABBREVIATION_HOME = ? OR TEAM_ABBREVIATION_AWAY = ?"
        if table == "raw":
            return self.get_raw_games(where, (team_abbrev, team_abbrev))
        return self.get_final_games(where, (team_abbrev, team_abbrev))

    def get_latest_game_date(self, table: str = "raw") -> Optional[str]:
        """Obtiene la fecha del último juego registrado."""
        tbl = "nba_games_raw" if table == "raw" else "nba_games_final"
        cursor = self._conn.execute(f"SELECT MAX(GAME_DATE) FROM {tbl}")
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]
        return None

    def get_game_count(self, table: str = "final") -> int:
        """Obtiene el número total de juegos en la tabla."""
        tbl = "nba_games_raw" if table == "raw" else "nba_games_final"
        cursor = self._conn.execute(f"SELECT COUNT(*) FROM {tbl}")
        return cursor.fetchone()[0]

    def get_seasons(self, table: str = "final") -> List[int]:
        """Obtiene la lista de temporadas disponibles."""
        tbl = "nba_games_raw" if table == "raw" else "nba_games_final"
        cursor = self._conn.execute(
            f"SELECT DISTINCT SEASON FROM {tbl} ORDER BY SEASON"
        )
        return [row[0] for row in cursor.fetchall()]

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Ejecuta una consulta SQL arbitraria."""
        df = pd.read_sql_query(sql, self._conn, params=params)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    def execute(self, sql: str, params: tuple = ()):
        """Ejecuta una sentencia SQL que no retorna datos."""
        self._conn.execute(sql, params)

    def add_column(self, table: str, column: str, col_type: str, default=None):
        """
        Agrega una columna a una tabla existente.

        Args:
            table: Nombre de la tabla ('nba_games_raw' o 'nba_games_final').
            column: Nombre de la nueva columna.
            col_type: Tipo SQL ('REAL', 'INTEGER', 'TEXT').
            default: Valor por defecto opcional.
        """
        tbl = "nba_games_raw" if table == "raw" else "nba_games_final"

        # Verificar si la columna ya existe
        cursor = self._conn.execute(f"PRAGMA table_info({tbl})")
        existing = [row[1] for row in cursor.fetchall()]

        if column in existing:
            logger.info(f"Columna {column} ya existe en {tbl}")
            return False

        sql = f"ALTER TABLE {tbl} ADD COLUMN {column} {col_type}"
        if default is not None:
            if isinstance(default, str):
                sql += f" DEFAULT '{default}'"
            else:
                sql += f" DEFAULT {default}"

        self._conn.execute(sql)
        logger.info(f"Columna {column} ({col_type}) agregada a {tbl}")
        return True

    def add_columns(self, table: str, columns: list[dict]):
        """
        Agrega múltiples columnas a una tabla.

        Args:
            table: Nombre de la tabla.
            columns: Lista de dicts con {'name': str, 'type': str, 'default': value}.
        """
        added = 0
        for col in columns:
            if self.add_column(table, col["name"], col["type"], col.get("default")):
                added += 1
        logger.info(f"{added} columnas agregadas a nba_games_{table}")
        return added

    # ----------------------------------------------------------------
    # Migration
    # ----------------------------------------------------------------

    def migrate_parquet_to_sqlite(
        self, parquet_path: str, table: str, if_exists: str = "replace"
    ) -> int:
        """
        Migra un archivo parquet a una tabla SQLite.

        Args:
            parquet_path: Ruta al archivo .parquet.
            table: 'raw' o 'final'.
            if_exists: 'replace' o 'append'.
        """
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {parquet_path}")

        logger.info(f"Migrando {parquet_path} -> nba_games_{table}")
        df = pd.read_parquet(path)

        # Log the migration
        self._conn.execute(
            "INSERT INTO migration_log (operation, source, row_count, status) "
            "VALUES (?, ?, ?, 'in_progress')",
            ("migrate", path.name, len(df)),
        )

        try:
            if table == "raw":
                count = self.insert_raw_games(df, if_exists)
            else:
                count = self.insert_final_games(df, if_exists)

            self._conn.execute(
                "UPDATE migration_log SET status = 'completed', "
                "completed_at = datetime('now'), row_count = ? "
                "WHERE status = 'in_progress' AND source = ?",
                (count, path.name),
            )
            logger.info(f"Migración completada: {count} filas")
            return count
        except Exception as e:
            self._conn.execute(
                "UPDATE migration_log SET status = 'failed', "
                "error_msg = ? WHERE status = 'in_progress' AND source = ?",
                (str(e), path.name),
            )
            logger.error(f"Error en migración: {e}")
            raise

    # ----------------------------------------------------------------
    # Utility / Inspection
    # ----------------------------------------------------------------

    def table_info(self, table: str = "raw") -> dict:
        """Información sobre una tabla."""
        tbl = "nba_games_raw" if table == "raw" else "nba_games_final"
        cursor = self._conn.execute(f"PRAGMA table_info({tbl})")
        columns = cursor.fetchall()
        count = self.get_game_count(table)
        return {
            "table": tbl,
            "row_count": count,
            "column_count": len(columns),
            "columns": [(c[1], c[2]) for c in columns],
        }

    def list_tables(self) -> List[str]:
        """Lista todas las tablas en la base de datos."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]
