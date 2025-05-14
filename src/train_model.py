import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from src.data_loader import load_data
from src.preprocessing import build_preprocessor

def train_and_evaluate(data_path='data/songs.csv',
                       model_output='models/song_success_model.pkl'):
    # 1) Load
    df = load_data(data_path)
    # 2) Preprocess
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
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )
    print("🔍 Starting GridSearchCV…")
    grid.fit(X_train, y_train)
    print("✅ GridSearchCV done.")
    # 5) Evaluate
    y_pred  = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test ROC AUC  : {auc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    # 6) Save
    joblib.dump(grid.best_estimator_, model_output)
    print(f"💾 Model saved to {model_output}")
    return grid

if __name__ == "__main__":
    # When invoked as a script, train and print everything
    train_and_evaluate(
        data_path='data/songs.csv',
        model_output='models/song_success_model.pkl'
    )
