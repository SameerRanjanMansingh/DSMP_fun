import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import json



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
        basic_metrics = evaluate_model(cv_results)
        print(f"Cross-validation accuracy: {basic_metrics['mean_score']} +/- {basic_metrics['std_dev']} (std) min: {basic_metrics['min_score']}, max: {basic_metrics['max_score']}")

        X = pd.read_pickle("data/processed/X.pkl")
        y = pd.read_pickle("data/processed/y.pkl")
        model_pipe = joblib.load("models/model_pipe.pkl")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        model_pipe.fit(X_train, y_train)
        additional_metrics = evaluate_additional_metrics(model_pipe, X_test, y_test)

        metrics = {**basic_metrics, **additional_metrics}
        print(f"Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1 Score: {metrics['f1']}")

        with open("results/metrics.json", "w") as f:
            json.dump(metrics, f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
