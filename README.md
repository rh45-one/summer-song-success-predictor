# Reto nº6 – Challenge The Summer Song

## Descripción / Description
**Español:**
Proyecto para el Reto nº6 del hackathon *Challenge The Summer Song* en el programa *Enseña by Oracle Spain*. Entrena un modelo de ML que predice qué características acústicas y de popularidad hacen que una canción sea la “Summer Song” exitosa.

**English:**
Project for Challenge The Summer Song Hackathon, Task #6, under the Enseña by Oracle Spain program. Trains an ML model to predict which acoustic features and popularity metrics make a track a successful “Summer Song.”

## Estructura / Structure
```
summer-song-success-predictor/
├── data/               # Datasets
│   └── summer_songs.csv
├── notebooks/          # Jupyter notebooks
│   └── summer_songs_model.ipynb
├── models/             # Modelos serializados
│   └── song_success_model.pkl
├── src/                # Código fuente modular (opcional)
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train_model.py
└── README.md           # Documentación (esta misma)
```

## Requisitos / Requirements
- Python 3.8+
- pip install pandas numpy scikit-learn matplotlib seaborn joblib

## Uso / Usage
```bash
git clone https://github.com/tu-usuario/summer-song-success-predictor.git
cd summer-song-success-predictor
# Coloca canciones_verano.csv en data/
jupyter notebook notebooks/modelo_cancion_verano.ipynb
```

## Licencia / License
MIT © 2025
