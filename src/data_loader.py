import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: No data found in the file.")
        raise
    except pd.errors.ParserError:
        print("Error: Error parsing the file.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    file_path = "data/raw/hotel_bookings.csv"
    data = load_data(file_path)
    data.to_pickle("data/processed/processed_data.pkl")
