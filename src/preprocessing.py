import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(df: pd.DataFrame):
    """
    Construye un transformador de columna completo para:
      - Escalar numéricos con StandardScaler.
      - Codificar categóricos con OneHotEncoder (sin sparsity).
    Parámetros:
      - df: DataFrame que incluye la columna objetivo 'exito'.
    Devuelve:
      - preprocessor: ColumnTransformer listo para transformar X.
      - X: DataFrame de características (sin la columna 'exito').
      - y: Serie con la etiqueta binaria 'exito'.
    """
    X = df.drop('exito', axis=1)
    y = df['exito']

    # Detectar columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # Pipelines para cada tipo
    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor, X, y
