# import os
import pandas as pd

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the absolute path to the CSV file
# csv_path = os.path.join(script_dir, '../data/raw/hotel_bookings.csv')

# # Read the CSV file
# df = pd.read_csv(csv_path)
# print(script_dir)





from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent

def fun(filepath:str):
    df1 = pd.read_csv(filepath)
    print('1',type(filepath))

# Example usage
# df1 = pd.read_csv(base_dir / 'data/raw/hotel_bookings.csv')
fun(base_dir / 'data/raw/hotel_bookings.csv')

print(type(base_dir / 'data/raw/hotel_bookings.csv'))


if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent

    file_path = 'data/raw/hotel_bookings.csv'

    df1 = pd.read_csv(file_path)


