import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import mlflow


import dagshub
dagshub.init(repo_owner='SameerRanjanMansingh', repo_name='DSMP_fun', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/SameerRanjanMansingh/DSMP_fun.mlflow")

def evaluate_model(cv_results: np.ndarray) -> dict:
    metrics = {
        "mean_score": round(np.mean(cv_results), 4),
        "std_dev": round(np.std(cv_results), 4),
        "min_score": round(min(cv_results), 4),
        "max_score": round(max(cv_results), 4)
    }
    return metrics

def evaluate_additional_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

if __name__ == "__main__":
    try:
        cv_results = np.load("models/cv_results.npy")

        X = pd.read_pickle("data/processed/X.pkl")
        y = pd.read_pickle("data/processed/y.pkl")

        model_pipe = joblib.load("models/model_pipe.pkl")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        model_pipe.fit(X_train, y_train)
        
        mlflow.set_experiment('iris_dt')
        with mlflow.start_run(run_name='run1'):
            mlflow.log_param("n_estimators", 160)
            mlflow.log_param("max_features", 0.4)
            mlflow.log_param("min_samples_split", 2)
            mlflow.log_param("n_jobs", -1)
            mlflow.log_param("random_state", 0)
            
            # Log metrics
            metrics = evaluate_model(cv_results)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            additional_metrics = evaluate_additional_metrics(model_pipe, X_test, y_test)
            for key, value in additional_metrics.items():
                mlflow.log_metric(key, value)
            
            # Log the model
            mlflow.sklearn.log_model(model_pipe, "model")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
