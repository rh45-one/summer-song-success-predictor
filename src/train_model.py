import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.data_loader import load_data
from src.preprocessing import build_preprocessor

def train_and_evaluate(
    data_path: str = 'data/songs.csv',
    model_output: str = 'models/song_success_model.pkl'
):
    """
    Flujo completo de entrenamiento y evaluación:
      1. Carga datos.
      2. Construye preprocesador.
      3. Divide en train/test.
      4. Ajusta GridSearchCV sobre RandomForest.
      5. Muestra métricas.
      6. Guarda el modelo óptimo.
    Parámetros:
      - data_path: ruta al CSV.
      - model_output: ruta de salida del pickle.
    """
    # 1. Carga y preprocesado
    df = load_data(data_path)
    preprocessor, X, y = build_preprocessor(df)

    # 2. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Pipeline + GridSearchCV
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # 4. Resultados
    print('\n=== Mejor configuración CV ===')
    print(f'ROC AUC (CV): {grid.best_score_:.4f}')
    print('Parámetros:', grid.best_params_)

    # 5. Evaluación en test
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:,1]
    print('\n=== Evaluación en Test ===')
    print(classification_report(y_test, y_pred))
    print(f'ROC AUC (test): {roc_auc_score(y_test, y_proba):.4f}')
    print(f'Accuracy (test): {accuracy_score(y_test, y_pred):.4f}')

    # 6. Guardado del modelo
    joblib.dump(grid.best_estimator_, model_output)
    print(f'\nModelo guardado en: {model_output}')

if __name__ == '__main__':
    train_and_evaluate()
