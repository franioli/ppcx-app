import os
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text


class DICdb:
    """Database connection and query interface for DIC data."""

    def __init__(
        self,
        connection_string: str | None = None,
        host: str | None = None,
        port: str | int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        **kwargs,
    ):
        """Initialize database connection.

        Args:
            connection_string: Complete PostgreSQL connection string. If provided,
                other parameters are ignored.
            host: Database host (default: 'localhost')
            port: Database port (default: 5432)
            database: Database name
            user: Database username
            password: Database password
            **kwargs: Additional SQLAlchemy engine parameters

        Examples:
            # Using connection string
            db = DICDatabase("postgresql://user:pass@host:port/db")

            # Using individual parameters
            db = DICDatabase(host="localhost", database="mydb", user="user", password="pass")

            # Using environment variables for missing parameters
            db = DICDatabase(database="mydb")  # Other params from env vars
        """
        if connection_string:
            self.engine: Engine = create_engine(connection_string, **kwargs)
        else:
            # Use individual parameters or environment variables as fallback
            host = host or os.environ.get("DB_HOST", "localhost")
            port = port or os.environ.get("DB_PORT", "5432")
            database = database or os.environ.get("DB_NAME")
            user = user or os.environ.get("DB_USER")
            password = password or os.environ.get("DB_PASSWORD", "")

            if not database:
                raise ValueError(
                    "Database name must be provided either as parameter or DB_NAME environment variable"
                )
            if not user:
                raise ValueError(
                    "Username must be provided either as parameter or DB_USER environment variable"
                )

            connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )
            self.engine: Engine = create_engine(connection_string, **kwargs)

    def execute_query(
        self,
        query: str,
        params: list | tuple | dict | None = None,
        return_dataframe: bool = True,
    ) -> pd.DataFrame | Any:
        """Execute a custom SQL query.

        Args:
            query: SQL query string
            params: Query parameters (list, tuple, or dict)
            return_dataframe: If True, return pandas DataFrame. If False, return raw result

        Returns:
            DataFrame if return_dataframe=True, otherwise raw SQLAlchemy result

        Examples:
            # Simple query
            df = db.execute_query("SELECT * FROM my_table LIMIT 10")

            # Query with parameters (list/tuple)
            df = db.execute_query("SELECT * FROM my_table WHERE id = %s", [123])

            # Query with named parameters (dict)
            df = db.execute_query("SELECT * FROM my_table WHERE id = %(id)s", {'id': 123})
        """
        if return_dataframe:
            return pd.read_sql(query, self.engine, params=params)
        else:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return result

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get column information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            DataFrame with column information
        """
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))

    def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of table names
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        df = self.execute_query(query)
        return df["table_name"].tolist()

    def get_dic_data(self, target_date: str | date) -> pd.DataFrame:
        """Get DIC displacement data for a specific date.

        Args:
            target_date: Target date as string (YYYY-MM-DD) or date object

        Returns:
            DataFrame with displacement data for the specified date
        """
        query = """
        SELECT 
            A.id as analysis_id,
            A.master_image_id,
            A.master_timestamp,
            A.slave_image_id,
            A.slave_timestamp,
            A.time_difference_hours,
            R.seed_x_px, 
            R.seed_y_px, 
            R.displacement_x_px, 
            R.displacement_y_px,
            R.displacement_magnitude_px
        FROM glacier_monitoring_app_dicresult R
        JOIN glacier_monitoring_app_dicanalysis A ON R.analysis_id = A.id
        WHERE DATE(A.master_timestamp) = %s
        ORDER BY R.seed_x_px, R.seed_y_px
        """

        return self.execute_query(query, (str(target_date),))

    def get_dic_dates(self) -> tuple[list[str], list[str]]:
        """Get all unique dates (master and slave images) for which DIC analyses are available.

        Returns:
            Tuple of lists containing master and slave image dates
        """
        query = """
        SELECT DISTINCT DATE(master_timestamp), DATE(slave_timestamp)
        FROM glacier_monitoring_app_dicanalysis
        ORDER BY master_timestamp
        """

        df = self.execute_query(query)
        return (df["m_date"].tolist(), df["s_date"].tolist())

    def get_analysis_summary(self) -> pd.DataFrame:
        """Get summary statistics for all analyses.

        Returns:
            DataFrame with summary statistics per analysis date
        """
        query = """
        SELECT 
            DATE(A.master_timestamp) as analysis_date,
            COUNT(R.id) as num_points,
            AVG(R.displacement_magnitude_px) as avg_magnitude,
            MAX(R.displacement_magnitude_px) as max_magnitude,
            MIN(R.displacement_magnitude_px) as min_magnitude,
            STDDEV(R.displacement_magnitude_px) as std_magnitude
        FROM glacier_monitoring_app_dicanalysis A
        LEFT JOIN glacier_monitoring_app_dicresult R ON A.id = R.analysis_id
        GROUP BY DATE(A.master_timestamp)
        ORDER BY analysis_date
        """

        return self.execute_query(query)
