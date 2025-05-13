import pandas as pd

def load_data(path: str = 'data/songs.csv') -> pd.DataFrame:
    """
    Carga el dataset de canciones desde un CSV.
    Parámetros:
      - path: ruta al CSV con los datos.
    Devuelve:
      - df: DataFrame con los datos originales.
    """
    df = pd.read_csv(path)
    return df
