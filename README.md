# Reto nº6 – Challenge The Summer Song

## Descripción / Description

**Español:**
Proyecto para el Reto nº6 del hackathon *Challenge The Summer Song* en el programa *Enseña by Oracle Spain*. Entrena un modelo de ML que predice qué características acústicas y de popularidad hacen que una canción sea la “Summer Song” exitosa.

**English:**
Project for Challenge The Summer Song Hackathon, Task #6, under the Enseña by Oracle Spain program. Trains an ML model to predict which acoustic features and popularity metrics make a track a successful “Summer Song.”

## Enunciado / Statement

Entrena un modelo predictivo con herramientas de Machine Learning que prediga qué características hacen exitosa una canción del verano.

Se permite generar el modelo directamente en Jupyter Notebook o usar Oracle Machine Learning (Database Actions) y trabajar desde el propio cloud:

* Exportar datos desde la app en APEX
* Crear un notebook en Jupyter
* Usar AutoML o **scikit-learn** para entrenar el modelo
* Subir el notebook a GitHub y compartir enlace

## Implementación / Implementation

Este proyecto utiliza **scikit-learn 1.6.1** para el desarrollo del modelo predictivo, sin depender de soluciones AutoML. Se aprovecha el API de Pipeline de scikit-learn para crear un flujo de trabajo completo, combinando preprocesamiento de datos y entrenamiento de un modelo RandomForest con búsqueda de hiperparámetros mediante GridSearchCV.

## Estructura / Structure

```
summer-song-success-predictor/
├── data/               # Datasets
│   └── songs.csv       # Datos brutos de canciones (track_name, artists, popularidad, features...)
├── notebooks/          # Jupyter notebooks
│   └── summer_song_model.ipynb  # Notebook con EDA, entrenamiento y scoring
├── models/             # Modelos serializados (Git LFS)
│   └── song_success_model.pkl
├── src/                # Código fuente modular
│   ├── data_loader.py      # Funciones para cargar datos
│   ├── preprocessing.py    # Funciones de preprocesado y pipeline
│   └── train_model.py      # Script para entrenamiento completo y serialización
└── README.md           # Documentación del proyecto
```

## Requisitos / Requirements

* Python 3.8+ (64-bit)
* scikit-learn==1.6.1
* pandas
* numpy
* matplotlib
* seaborn
* joblib

Puedes instalar todo de una vez con:

```bash
pip install scikit-learn==1.6.1 pandas numpy matplotlib seaborn joblib
```

## Datos / Data

El archivo `songs.csv` debe contener las siguientes columnas (sin variable objetivo explícita):

* `row_id`, `track_id`, `artists`, `album_name`, `track_name`  # Identificadores y metadatos (se descartan en el pipeline)
* `popularity`      # Métrica de popularidad (0–100)
* `duration_ms`     # Duración en milisegundos
* `explicit`        # Booleano (True/False)
* `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`  # Características acústicas numéricas
* `track_genre`     # Género musical (categórica)

## Uso / Usage

```bash
git clone https://github.com/rh45-one/summer-song-success-predictor.git
cd summer-song-success-predictor
pip install scikit-learn==1.6.1 pandas numpy matplotlib seaborn joblib
jupyter notebook notebooks/summer_song_model.ipynb
```

## Parámetros configurables / Configurable Parameters

En el notebook existe un parámetro:

* **threshold** (celda de scoring): Por defecto `0.7`. Determina el umbral de probabilidad para marcar una canción como "hit de verano". Ajusta este valor para modificar precisión vs. cobertura.

## Resultados / Results

* **Precisión (Accuracy) en test**: 99.97%
* **ROC AUC en test**: 0.999999

Las características más determinantes incluyen `danceability`, `energy` y `valence`.

## Autor / Author

Creado por el equipo **Kermit Panic** para el Programa Enseña by Oracle Spain.

## Licencia / License

MIT © 2025
