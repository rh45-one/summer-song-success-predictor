# src/train_model.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data
from src.preprocessing import build_preprocessor

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

def train_and_evaluate(
    data_path: str = 'data/songs.csv',
    model_output: str = 'models/song_success_model.pkl'
):
     # 1) Carga datos
    df = load_data(data_path)

    # 1.1) Definir variable objetivo 'exito'
    # Ejemplo: popularidad ≥ 70 → éxito (1), si no → 0
    df['exito'] = (df['popularity'] >= 70).astype(int)

    # 2) Preprocesador
    preprocessor, X, y = build_preprocessor(df)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Pipeline + GridSearch
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    param_grid = { 'clf__n_estimators': [100,200], 'clf__max_depth': [None,10,20] }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    # 5) Evaluation (prints to console)
    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:,1]
    print("Test Accuracy :", accuracy_score(y_test, y_pred))
    print("Test ROC AUC :", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    # 6) Save model
    joblib.dump(grid.best_estimator_, model_output)
    print(f"Model serialized to {model_output}")

    # 7) Return grid for further use
    return grid

if __name__ == "__main__":
    # When invoked as a script, train and print everything
    train_and_evaluate(
        data_path='data/songs.csv',
        model_output='models/song_success_model.pkl'
    )