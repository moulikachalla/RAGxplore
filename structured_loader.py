import os
import duckdb
import pandas as pd
from pathlib import Path

class StructuredDataLoader:
    def __init__(self, db_path="data/structured/allyin.db"):
        """
        Initialize DuckDB connection.
        If db_path is set to a file (like 'allyin.db'), data will persist on disk.
        """
        self.conn = duckdb.connect(db_path)
        self.loaded_tables = []

    def load_csv_to_table(self, csv_path, table_name=None):
        """
        Load a single CSV file into DuckDB as a table.
        Auto-generates table name from file name if not provided.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if table_name is None:
            table_name = Path(csv_path).stem

        df = pd.read_csv(csv_path)
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        self.loaded_tables.append(table_name)
        print(f"Loaded '{csv_path}' into table '{table_name}' with {len(df)} rows")

    def load_all_csvs_from_directory(self, directory_path):
        """
        Load all CSV files from a given directory into DuckDB.
        Each CSV will become a separate table.
        """
        csv_files = list(Path(directory_path).glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {directory_path}")
            return

        for csv_file in csv_files:
            self.load_csv_to_table(str(csv_file))

    def show_tables(self):
        """
        Return a DataFrame of all tables currently loaded in DuckDB.
        """
        return self.conn.execute("SHOW TABLES").df()

    def preview_table(self, table_name, limit=10):
        """
        Preview the first few rows of a specific table.
        """
        return self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").df()

if __name__ == "__main__":
    print("Loading structured data into DuckDB")

    # Use persistent DB file to allow SQLRetriever to access the same data
    loader = StructuredDataLoader(db_path="data/structured/allyin.db")
    loader.load_all_csvs_from_directory("data/structured")

    print("\nTables loaded:")
    print(loader.show_tables())

    for table in loader.loaded_tables:
        print(f"\nPreview of '{table}' table:")
        print(loader.preview_table(table))
