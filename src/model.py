import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def get_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=160,
                                  max_features=0.4,
                                  min_samples_split=2,
                                  n_jobs=-1,
                                  random_state=0)

def train_model(X: pd.DataFrame, y: pd.Series, preprocessor, model: RandomForestClassifier):
    kfolds = 4
    split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    model_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    cv_results = cross_val_score(model_pipe, X, y, cv=split, scoring="accuracy", n_jobs=-1)
    return cv_results, model_pipe

if __name__ == "__main__":
    try:
        X = pd.read_pickle("X.pkl")
        y = pd.read_pickle("y.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        model = get_model()
        cv_results, model_pipe = train_model(X, y, preprocessor, model)
        np.save("cv_results.npy", cv_results)
        joblib.dump(model_pipe, "model_pipe.pkl")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

