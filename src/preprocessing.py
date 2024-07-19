import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib



num_features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                "babies", "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "agent", "company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel", "arrival_date_month", "meal", "market_segment",
                "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]

def preprocess_data() -> ColumnTransformer:
    num_transformer = SimpleImputer(strategy="constant")

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])

    return preprocessor

if __name__ == "__main__":
    try:
        data = pd.read_pickle("data/processed/processed_data.pkl")
        preprocessor = preprocess_data()
        X = data[num_features + cat_features]
        y = data["is_canceled"]
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        X.to_pickle("data/processed/X.pkl")
        y.to_pickle("data/processed/y.pkl")
    except FileNotFoundError:
        print("Error: The processed_data.pkl file was not found.")
        raise
    except KeyError as e:
        print(f"Error: Missing key in the dataframe: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
