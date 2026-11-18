import zipfile
import pandas as pd


def inspect_path(file_path: str):
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path) as zf:
            # Filter out hidden macOS metadata files (like __MACOSX) if present
            csv_files = sorted([
                f
                for f in zf.namelist()
                if f.endswith(".csv") and not f.startswith("__")
            ])
            print(f"Found {len(csv_files)} CSVs inside {file_path}: {csv_files}")

            for name in csv_files:
                print(f"\n--- {name} ---")
                with zf.open(name) as f:
                    df = pd.read_csv(f, nrows=5)
                    print(df.dtypes)
    else:
        df = pd.read_csv(file_path, nrows=5)
        print(df.columns)
        print(df.head())
        print(df.dtypes)

print("----------------------------data for 2025-------------------------------")
inspect_path("data/raw/202501-citibike-tripdata.zip")
print("----------------------------data for 2026-------------------------------")
inspect_path("data/raw/202609-citibike-tripdata.zip")
