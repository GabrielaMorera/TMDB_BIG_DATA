import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
import json
import requests
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import traceback
import xgboost as xgb

# Configuración de la página
st.set_page_config(
    page_title="TMDB Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de PostgreSQL
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Villeta-11"
POSTGRES_HOST = "movie_postgres"
POSTGRES_PORT = "5432"

# Configuración de TMDB API
TMDB_API_KEY = "e8e1dae84a0345bd3ec23e3030905258"
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlOGUxZGFlODRhMDM0NWJkM2VjMjNlMzAzMDkwNTI1OCIsIm5iZiI6MTc0MTkxMzEzNi4xMjEsInN1YiI6IjY3ZDM3YzMwYmY0ODE4ODU0YzY0ZTVmNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tdq930_qLqYbYAqwVLl3Tdw84HdEsZtM41CX_9-lJNU"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Ruta fija al modelo
MODEL_PATH_FILE = "/opt/airflow/data/model_location.txt"


# Función para cargar el modelo más reciente
@st.cache_resource
def load_latest_model():
    try:
        # Leer la ruta del modelo desde el archivo
        with open(MODEL_PATH_FILE, 'r') as path_file:
            model_path = path_file.read().strip()

        # Cargar modelo
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.success(f"✅ Modelo cargado desde: {model_path}")

            # Cargar métricas (si están en el mismo directorio)
            metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                st.success(f"✅ Métricas cargadas desde: {metrics_path}")

                # Mostrar R^2 y MSE en el sidebar
                if 'r2' in metrics:
                    st.sidebar.metric("R² Score del Modelo", f"{metrics['r2']:.3f}")
                if 'mse' in metrics:
                    st.sidebar.metric("Error Cuadrático Medio", f"{metrics['mse']:.3f}")

            # Cargar feature importance (si está en el mismo directorio)
            feature_path = os.path.join(os.path.dirname(model_path), "feature_importance.csv")
            feature_names = ["vote_average", "vote_count", "runtime", "budget", "revenue"]  # Default
            if os.path.exists(feature_path):
                try:
                    feature_df = pd.read_csv(feature_path)
                    feature_names = feature_df['feature'].tolist()
                    st.success(f"✅ Características cargadas desde: {feature_path}")
                except Exception as e:
                    st.warning(f"Error al cargar características desde CSV: {e}")
            elif 'feature_importance' in metrics and isinstance(metrics['feature_importance'], list):
                importance_dict = {item['feature']: item['importance'] for item in metrics['feature_importance']}
                metrics['feature_importance'] = importance_dict
                feature_names = list(importance_dict.keys())

            return model, metrics, feature_names

        else:
            st.warning(f"No se encontró el archivo del modelo en {model_path}. Utilizando un modelo simulado.")
            # Crear un modelo dummy para simular si no hay uno
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.coef_ = np.array([0.5, 0.3, 0.2, 0.1, 0.2])  # Coeficientes simulados
            model.intercept_ = 5.0
            metrics = {
                "r2": 0.75,
                "mse": 2.5,
                "feature_importance": {
                    "vote_average": 0.5,
                    "vote_count": 0.3,
                    "runtime": 0.2,
                    "budget": 0.1,
                    "revenue": 0.2
                }
            }
            return model, metrics, ["vote_average", "vote_count", "runtime", "budget", "revenue"]

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        # Crear un modelo dummy para simulación
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.coef_ = np.array([0.5, 0.3, 0.2, 0.1, 0.2])
        model.intercept_ = 5.0
        metrics = {"r2": 0.75, "mse": 2.5}
        return model, metrics, ["vote_average", "vote_count", "runtime", "budget", "revenue"]


# Función para obtener conexión a PostgreSQL
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        return conn
    except Exception as e:
        st.error(f"Error de conexión a PostgreSQL: {e}")
        return None


# Función para buscar películas por título
def search_movie_by_title(title, limit=5):
    results = []

    # Primero buscar en la base de datos
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT tmdb_id, title, release_date, popularity, vote_average FROM movies WHERE title ILIKE %s LIMIT %s",
                (f"%{title}%", limit))
            movie_rows = cur.fetchall()

            if movie_rows:
                for row in movie_rows:
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "release_date": row[2],
                        "popularity": row[3],
                        "vote_average": row[4],
                        "source": "Database"
                    })

                cur.close()
                conn.close()
                return results

            cur.close()
            conn.close()
    except Exception as e:
        st.warning(f"Error al buscar en la base de datos: {e}")

    # Si no hay resultados en la base de datos, consultar API de TMDB
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        headers = {
            "Authorization": f"Bearer {TMDB_TOKEN}",
            "Content-Type": "application/json;charset=utf-8"
        }
        params = {
            "query": title,
            "language": "es-ES",
            "page": 1
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            movies = response.json().get("results", [])

            for movie in movies[:limit]:
                results.append({
                    "id": movie.get("id"),
                    "title": movie.get("title"),
                    "release_date": movie.get("release_date"),
                    "popularity": movie.get("popularity"),
                    "vote_average": movie.get("vote_average"),
                    "poster_path": movie.get("poster_path"),
                    "source": "TMDB API"
                })
    except Exception as e:
        st.error(f"Error al consultar TMDB API: {e}")

    return results

# Función para obtener detalles de película por ID
def get_movie_details(movie_id):
    """
    Obtiene los detalles de una película buscando primero en la base de datos
    (warehouse y luego tabla 'movies') y finalmente en la API de TMDB.
    """
    movie = None # Inicializar movie a None
    # No es necesario buscar overview específicamente si lo obtenemos de 'movies'

    st.info(f"Buscando detalles para película ID: {movie_id}")

    # 1. Intentar buscar datos básicos en movie_data_warehouse
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            st.info("Consultando movie_data_warehouse...")
            cur.execute(
                """
                SELECT
                    tmdb_id, title, release_date,
                    popularity, vote_average, vote_count, budget, revenue,
                    runtime, genre, director, roi, popularity_level, rating_level, release_year
                FROM movie_data_warehouse
                WHERE tmdb_id = %s
                LIMIT 1
                """, (movie_id,))
            row = cur.fetchone()

            if row:
                movie = {
                    "id": row[0], "title": row[1], "release_date": row[2],
                    "popularity": float(row[3]) if row[3] is not None else 0.0,
                    "vote_average": float(row[4]) if row[4] is not None else 0.0,
                    "vote_count": int(row[5]) if row[5] is not None else 0,
                    "budget": int(row[6]) if row[6] is not None else 0,
                    "revenue": int(row[7]) if row[7] is not None else 0,
                    "runtime": int(row[8]) if row[8] is not None else 0,
                    "genre": row[9], "directors": [row[10]] if row[10] else [],
                    "genres": [row[9]] if row[9] else [], # Placeholder
                    "roi": float(row[11]) if row[11] is not None else 0.0,
                    "popularity_category": row[12], "rating_category": row[13],
                    "release_year": int(row[14]) if row[14] is not None else None,
                    "source": "Database (Warehouse)"
                }
                st.info(f"-> Datos básicos encontrados en Warehouse.")
            else:
                 st.info(f"-> Película no encontrada en Warehouse.")
            cur.close()
            conn.close()
        else:
            st.warning("No se pudo establecer conexión DB para buscar en Warehouse.")
    except Exception as e:
        st.warning(f"Error al buscar en movie_data_warehouse: {e}")
        st.warning(traceback.format_exc())

    # 2. Consultar SIEMPRE la tabla 'movies' para obtener overview y listas completas (si existen)
    #    Esto también sirve si la película no estaba en el warehouse.
    st.info("Consultando tabla 'movies' para detalles adicionales...")
    try:
         conn = get_db_connection()
         if conn:
             cur = conn.cursor()
             # CORRECCIÓN: Seleccionar solo columnas existentes en 'movies' según el DAG
             cur.execute(
                 """
                 SELECT
                     overview, genres_str, directors_str, cast_str, companies_str,
                     title, release_date, popularity, vote_average, vote_count,
                     budget, revenue, runtime, roi, popularity_category, rating_category
                     -- NO SE INCLUYEN poster_path, backdrop_path PORQUE NO ESTÁN EN LA TABLA
                 FROM movies
                 WHERE tmdb_id = %s
                 LIMIT 1
                 """, (movie_id,)
             )
             details_row = cur.fetchone()
             if details_row:
                 st.info(f"-> Detalles adicionales encontrados en tabla 'movies'.")
                 # Si no teníamos datos del warehouse, creamos el objeto movie ahora
                 if movie is None:
                     movie = {"id": movie_id, "source": "Database (Movies Table Only)"}
                     # Llenar con datos básicos de la tabla 'movies'
                     movie["title"] = details_row[5]
                     movie["release_date"] = details_row[6]
                     movie["popularity"] = float(details_row[7]) if details_row[7] is not None else 0.0
                     movie["vote_average"] = float(details_row[8]) if details_row[8] is not None else 0.0
                     movie["vote_count"] = int(details_row[9]) if details_row[9] is not None else 0
                     movie["budget"] = int(details_row[10]) if details_row[10] is not None else 0
                     movie["revenue"] = int(details_row[11]) if details_row[11] is not None else 0
                     movie["runtime"] = int(details_row[12]) if details_row[12] is not None else 0
                     movie["roi"] = float(details_row[13]) if details_row[13] is not None else 0.0
                     movie["popularity_category"] = details_row[14]
                     movie["rating_category"] = details_row[15]
                     # Calcular año si no estaba
                     if movie.get("release_date") and len(str(movie.get("release_date"))) >= 4:
                         try: movie["release_year"] = int(str(movie["release_date"])[:4])
                         except: movie["release_year"] = None

                 # Actualizar/Añadir detalles de la tabla 'movies' al objeto 'movie'
                 movie["overview"] = details_row[0] if details_row[0] else movie.get("overview", "Sinopsis no disponible.")
                 movie["genres"] = details_row[1].split(',') if details_row[1] else movie.get("genres", [])
                 movie["directors"] = details_row[2].split(',') if details_row[2] else movie.get("directors", [])
                 movie["cast"] = details_row[3].split(',') if details_row[3] else [] # Obtener reparto completo
                 movie["production_companies"] = details_row[4].split(',') if details_row[4] else [] # Obtener compañías

                 # Limpiar listas vacías o con strings vacíos
                 movie["genres"] = [g.strip() for g in movie["genres"] if g.strip()]
                 movie["directors"] = [d.strip() for d in movie["directors"] if d.strip()]
                 movie["cast"] = [c.strip() for c in movie["cast"] if c.strip()]
                 movie["production_companies"] = [p.strip() for p in movie["production_companies"] if p.strip()]

                 # Actualizar fuente si venía del warehouse
                 if "Warehouse" in movie.get("source", ""):
                     movie["source"] = "Database (Warehouse + Movies Table)"
                 elif "Movies Table Only" not in movie.get("source", ""):
                      movie["source"] = "Database (Movies Table)"

                 # ---- IMPORTANTE: poster_path y backdrop_path NO están disponibles desde esta consulta ----
                 # Se obtendrán de la API si es necesario más adelante

             else:
                # No se encontraron detalles adicionales en 'movies'
                 st.info(f"-> No se encontraron detalles adicionales en tabla 'movies'.")
                 if movie is None: # Si tampoco estaba en el warehouse
                     st.info("Película no encontrada en ninguna tabla de la BD.")

             cur.close()
             conn.close()
         else:
             st.warning("No se pudo establecer conexión DB para buscar en tabla 'movies'.")
    except Exception as e:
         st.warning(f"Error al buscar detalles adicionales en la tabla 'movies': {e}")
         st.warning(traceback.format_exc())

    # 3. Si aún no se encontró nada en la base de datos O si faltan datos clave como el poster,
    #    consultar API TMDB como último recurso o para complementar.
    if movie is None or not movie.get("poster_path"): # Consultar API si no hay datos o falta el poster
         st.info(f"Consultando API TMDB...")
         try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            headers = {
                 "Authorization": f"Bearer {TMDB_TOKEN}",
                 "Content-Type": "application/json;charset=utf-8"
            }
            params = {
                 "language": "es-ES",
                 "append_to_response": "credits" # Para obtener directores y reparto
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                 data = response.json()
                 st.success(f"-> Detalles obtenidos de TMDB API.")

                 # Si no teníamos película de la BD, la creamos desde la API
                 if movie is None:
                      movie = {"id": data.get("id"), "source": "TMDB API"}

                 # Sobrescribimos o añadimos datos de la API (más frescos/completos)
                 movie["title"] = data.get("title", movie.get("title"))
                 movie["original_title"] = data.get("original_title")
                 movie["overview"] = data.get("overview", movie.get("overview"))
                 movie["release_date"] = data.get("release_date", movie.get("release_date"))
                 movie["popularity"] = data.get("popularity", movie.get("popularity"))
                 movie["vote_average"] = data.get("vote_average", movie.get("vote_average"))
                 movie["vote_count"] = data.get("vote_count", movie.get("vote_count"))
                 movie["budget"] = data.get("budget", movie.get("budget"))
                 movie["revenue"] = data.get("revenue", movie.get("revenue"))
                 movie["runtime"] = data.get("runtime", movie.get("runtime"))
                 movie["genres"] = [genre.get("name").strip() for genre in data.get("genres", []) if genre.get("name")] or movie.get("genres", [])
                 movie["directors"] = [person.get("name").strip() for person in data.get("credits", {}).get("crew", []) if person.get("job") == "Director" and person.get("name")] or movie.get("directors", [])
                 movie["cast"] = [person.get("name").strip() for person in data.get("credits", {}).get("cast", [])[:10] if person.get("name")] or movie.get("cast", [])
                 movie["production_companies"] = [company.get("name").strip() for company in data.get("production_companies", []) if company.get("name")] or movie.get("production_companies", [])
                 movie["poster_path"] = data.get("poster_path") # Obtenido de la API
                 movie["backdrop_path"] = data.get("backdrop_path") # Obtenido de la API

                 # Recalcular/Actualizar ROI
                 if movie.get("budget") and movie.get("revenue") and movie["budget"] > 0:
                    movie["roi"] = (movie["revenue"] - movie["budget"]) / movie["budget"]
                 elif "roi" not in movie: # Si no existía antes
                    movie["roi"] = None

                 # Recalcular/Actualizar año de lanzamiento
                 if movie.get("release_date") and len(str(movie.get("release_date"))) >= 4:
                     try: movie["release_year"] = int(str(movie["release_date"])[:4])
                     except: movie["release_year"] = movie.get("release_year") # Mantener si ya existía
                 elif "release_year" not in movie:
                     movie["release_year"] = None

                 # Recalcular/Actualizar categorías si es necesario (o mantener las de la BD si existen)
                 if "popularity_category" not in movie or movie["popularity_category"] == "Desconocida":
                    # (Tu lógica para calcular categorías aquí, si quieres)
                    movie["popularity_category"] = "Desconocida (API)"
                 if "rating_category" not in movie or movie["rating_category"] == "Desconocida":
                    # (Tu lógica para calcular categorías aquí, si quieres)
                    movie["rating_category"] = "Desconocida (API)"

                 # Actualizar la fuente si se complementó la BD
                 if "Database" in movie.get("source", ""):
                     movie["source"] += " + TMDB API"
                 else:
                     movie["source"] = "TMDB API" # Era solo de la API

            else:
                st.error(f"Error {response.status_code} al consultar TMDB API.")
                # No actualizamos 'movie' si la API falla y ya teníamos datos de la BD

         except Exception as e:
             st.error(f"Error general al consultar TMDB API: {e}")
             st.error(traceback.format_exc())
             # No actualizamos 'movie' si la API falla y ya teníamos datos de la BD

    # 4. Devolver el objeto movie (puede ser None si falló todo)
    if movie is None:
        st.error(f"No se pudieron obtener detalles para la película ID {movie_id} desde ninguna fuente.")
    else:
        # Asegurar tipos de datos numéricos correctos antes de devolver
        for key in ['popularity', 'vote_average', 'roi']:
            if key in movie and movie[key] is not None:
                try: movie[key] = float(movie[key])
                except (ValueError, TypeError): movie[key] = 0.0
        for key in ['id', 'vote_count', 'budget', 'revenue', 'runtime', 'release_year']:
             if key in movie and movie[key] is not None:
                 try: movie[key] = int(movie[key])
                 except (ValueError, TypeError): movie[key] = 0 # O None si prefieres

    return movie

# Función para predecir el éxito de una película
# En tmdb_streamlit_app.py
def predict_movie_success(movie, model, feature_names):
    try:
        # Crear características pre-estreno relevantes
        
        # Extraer o procesar el género principal
        main_genre = 'Unknown'
        if movie.get('genre'):
            main_genre = movie.get('genre')
        elif movie.get('genres') and isinstance(movie.get('genres'), list) and len(movie.get('genres')) > 0:
            main_genre = movie.get('genres')[0]
        elif movie.get('genres') and isinstance(movie.get('genres'), str):
            genres = movie.get('genres').split(',')
            if genres and genres[0].strip():
                main_genre = genres[0].strip()
                
        # Presupuesto en millones
        budget_million = float(movie.get('budget', 0)) / 1000000
        
        # Categoría de presupuesto
        budget_category = 'Medio'
        if budget_million <= 10:
            budget_category = 'Muy bajo'
        elif budget_million <= 30:
            budget_category = 'Bajo'
        elif budget_million <= 70:
            budget_category = 'Medio'
        elif budget_million <= 100:
            budget_category = 'Alto'
        else:
            budget_category = 'Blockbuster'
            
        # Determinar la temporada de estreno
        release_season = 4  # Default (otoño)
        if movie.get('release_date'):
            try:
                import datetime
                if isinstance(movie.get('release_date'), str) and len(movie.get('release_date')) >= 10:
                    month = int(movie.get('release_date')[5:7])
                    season_map = {
                        1: 4,  # Enero
                        2: 3, 3: 3, 4: 3,  # Primavera
                        5: 1, 6: 1, 7: 1, 8: 1,  # Verano
                        9: 4, 10: 4,  # Otoño
                        11: 2, 12: 2  # Navidad/premios
                    }
                    release_season = season_map.get(month, 4)
            except:
                pass
                
        # Duración óptima
        runtime = float(movie.get('runtime', 120))
        optimal_runtime = 1 if (runtime >= 90 and runtime <= 150) else 0
        
        # Ratio duración/presupuesto
        runtime_budget_ratio = runtime / (budget_million + 1)
        
        # Crear un diccionario con todas las características necesarias
        mapped_features = {
            'budget_million': budget_million,
            'runtime': runtime,
            'main_genre': main_genre,
            'vote_average': float(movie.get('vote_average', 7.0)),  # Valor razonable predeterminado
            'optimal_runtime': optimal_runtime,
            'runtime_budget_ratio': runtime_budget_ratio,
            'release_season': release_season,
            'release_year': int(movie.get('release_year', datetime.datetime.now().year)),
            'budget_category': budget_category
        }
        
        # Crear DataFrame con las características
        X_pred = pd.DataFrame([mapped_features])
        
        # Hacer predicción
        try:
            # Intenta predecir probabilidades (para modelos de clasificación)
            proba = model.predict_proba(X_pred)[0][1]
            is_classification = True
        except:
            # Si falla, asume que es un modelo de regresión
            proba = model.predict(X_pred)[0] / 100  # Normaliza a rango 0-1
            is_classification = False
        
        # Convertir probabilidad a una estimación de ROI
        # Una probabilidad alta (>0.7) sugiere un ROI>3
        estimated_roi = proba * 5  # Escala aproximada para estimar ROI
        
        # Categorizar resultado
        if proba >= 0.7:
            success_level = "Muy Alta"
            success_color = "#1E8449"  # Verde oscuro
        elif proba >= 0.5:
            success_level = "Alta"
            success_color = "#58D68D"  # Verde claro
        elif proba >= 0.3:
            success_level = "Media"
            success_color = "#F4D03F"  # Amarillo
        else:
            success_level = "Baja"
            success_color = "#E74C3C"  # Rojo

        # Resultado
        prediction_result = {
            "probability": float(proba * 100),  # Convertir a porcentaje
            "estimated_roi": float(estimated_roi),
            "success_level": success_level,
            "success_color": success_color,
            "is_successful": proba >= 0.5,  # 50%+ de probabilidad se considera éxito
            "model_type": "Clasificación" if is_classification else "Regresión"
        }

        return prediction_result
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        import traceback
        st.error(traceback.format_exc())
        return {
            "probability": 0,
            "estimated_roi": 0,
            "success_level": "Error",
            "success_color": "#E74C3C",
            "is_successful": False,
            "model_type": "Error"
        }

# Función para mostrar la predicción
def display_prediction(prediction, movie, metrics):
    st.markdown("## Predicción de Éxito")

    col1, col2, col3 = st.columns(3)

    with col1:
        score = prediction["popularity_score"]
        st.markdown(f"### Puntuación de Popularidad")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{score:.2f}</h1>",
                    unsafe_allow_html=True)

    with col2:
        success_level = prediction["success_level"]
        st.markdown(f"### Nivel de Éxito")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{success_level}</h1>",
                    unsafe_allow_html=True)

    with col3:
        st.markdown(f"### Probabilidad de Éxito")
        is_successful = "Alta" if prediction["is_successful"] else "Baja"
        color = "#1E8449" if prediction["is_successful"] else "#E74C3C"
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{is_successful}</h1>",
                    unsafe_allow_html=True)

    # Factores que influyeron
    st.markdown("### Factores que Influyeron en la Predicción")

    factors_data = []
    if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
        importance = metrics["feature_importance"]
        for feature, value in importance.items():
            # Mapear características a valores de la película
            feature_value = None
            if feature == 'budget' or feature == 'budget_million':
                feature_value = float(movie.get('budget', 0)) / 1000000 if movie.get('budget') else 0  # En millones
            elif feature == 'revenue' or feature == 'revenue_million':
                feature_value = float(movie.get('revenue', 0)) / 1000000 if movie.get('revenue') else 0  # En millones
            elif feature == 'release_year' and movie.get('release_date'):
                feature_value = movie.get('release_year', 2000)  # Usar el valor extraído o el valor por defecto
            elif feature == 'main_genre' and movie.get('genres'):
                feature_value = movie.get('genre', 'Unknown')
            elif feature in movie:
                feature_value = float(movie.get(feature, 0))  # Convertir a float para evitar problemas con Decimal

            if feature_value is not None:
                factors_data.append({
                    "Factor": feature,
                    "Importancia": abs(value),
                    "Valor en la Película": feature_value
                })
    else:
        # Si no hay importancia de características, usar valores por defecto
        default_features = ["vote_average", "vote_count", "runtime", "budget", "revenue"]
        for feature in default_features:
            if feature in movie:
                factors_data.append({
                    "Factor": feature,
                    "Importancia": 0.2,  # Valor por defecto equitativo
                    "Valor en la Película": float(movie.get(feature, 0))  # Convertir a float
                })

    # Mostrar tabla de factores
    df_factors = pd.DataFrame(factors_data)
    if not df_factors.empty:
        # Traducir nombres de factores
        factor_translations = {
            "vote_average": "Calificación Promedio",
            "vote_count": "Número de Votos",
            "runtime": "Duración (minutos)",
            "budget": "Presupuesto (millones $)",
            "budget_million": "Presupuesto (millones $)",
            "revenue": "Ingresos (millones $)",
            "revenue_million": "Ingresos (millones $)",
            "release_year": "Año de Lanzamiento",
            "main_genre": "Género Principal"
        }
        df_factors["Factor"] = df_factors["Factor"].map(factor_translations).fillna(df_factors["Factor"])
        
        # Formatear valores específicos
        for i, row in df_factors.iterrows():
            if "Presupuesto" in row["Factor"] or "Ingresos" in row["Factor"]:
                df_factors.at[i, "Valor en la Película"] = f"${row['Valor en la Película']:,.2f}M"
            elif "Calificación" in row["Factor"]:
                df_factors.at[i, "Valor en la Película"] = f"{row['Valor en la Película']:.1f}/10"

        st.table(df_factors)

    # Gráfico de importancia de factores
    if not df_factors.empty:
        fig = px.bar(
            df_factors,
            x="Importancia",
            y="Factor",
            orientation='h',
            color="Importancia",
            color_continuous_scale=["#E74C3C", "#F4D03F", "#1E8449"],
            title="Importancia de los Factores en la Predicción"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparación con otras películas
    st.markdown("### Comparación con la Media")
    
    try:
        # Obtener promedios de la base de datos
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    AVG(vote_average) as avg_rating,
                    AVG(vote_count) as avg_votes,
                    AVG(runtime) as avg_runtime,
                    AVG(budget) as avg_budget,
                    AVG(revenue) as avg_revenue,
                    AVG(popularity) as avg_popularity
                FROM movie_data_warehouse
                WHERE vote_count > 0
            """)
            avg_row = cur.fetchone()

            if avg_row:
                # Convertir todos los valores a floats para evitar problemas con Decimal
                avg_data = {
                    "Calificación": [float(movie.get("vote_average", 0)), float(avg_row[0] or 0)],
                    "Votos": [float(movie.get("vote_count", 0)), float(avg_row[1] or 0)],
                    "Duración": [float(movie.get("runtime", 0)), float(avg_row[2] or 0)],
                    "Presupuesto": [float(movie.get("budget", 0)) / 1000000, float(avg_row[3] or 0) / 1000000],
                    "Ingresos": [float(movie.get("revenue", 0)) / 1000000, float(avg_row[4] or 0) / 1000000],
                    "Popularidad": [float(movie.get("popularity", 0)), float(avg_row[5] or 0)]
                }

                # Crear DataFrame para comparación
                df_comp = pd.DataFrame({
                    "Métrica": list(avg_data.keys()),
                    "Esta Película": [avg_data[k][0] for k in avg_data.keys()],
                    "Promedio": [avg_data[k][1] for k in avg_data.keys()]
                })

                # Crear gráfico de comparación
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_comp["Métrica"],
                    y=df_comp["Esta Película"],
                    name="Esta Película",
                    marker_color="#3498DB"
                ))
                fig.add_trace(go.Bar(
                    x=df_comp["Métrica"],
                    y=df_comp["Promedio"],
                    name="Promedio",
                    marker_color="#F4D03F"
                ))

                fig.update_layout(
                    title="Comparación con Películas Promedio",
                    barmode="group",
                    xaxis_title="Métrica",
                    yaxis_title="Valor",
                    legend_title="Categoría"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Radar chart para comparación visual
                categories = list(avg_data.keys())

                fig = go.Figure()

                # Normalizar datos para radar chart
                max_values = {k: max(max(avg_data[k][0], avg_data[k][1]), 0.001) * 1.2 for k in avg_data.keys()}
                movie_values = [avg_data[k][0] / max_values[k] for k in avg_data.keys()]
                avg_values = [avg_data[k][1] / max_values[k] for k in avg_data.keys()]

                fig.add_trace(go.Scatterpolar(
                    r=movie_values,
                    theta=categories,
                    fill='toself',
                    name='Esta Película',
                    line_color='#3498DB'
                ))

                fig.add_trace(go.Scatterpolar(
                    r=avg_values,
                    theta=categories,
                    fill='toself',
                    name='Promedio',
                    line_color='#F4D03F'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Perfil de la Película vs Promedio"
                )
                st.plotly_chart(fig, use_container_width=True)

            cur.close()
            conn.close()
    except Exception as e:
        st.warning(f"No se pudo generar la comparación: {e}")
        import traceback
        st.warning(traceback.format_exc())

# Función para obtener estadísticas de la base de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_db_stats():
    try:
        conn = get_db_connection()
        if not conn:
            return None

        stats = {}

        # Número total de películas
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM movie_data_warehouse")
        stats["total_movies"] = cur.fetchone()[0]

        # Promedio de popularidad
        cur.execute("SELECT AVG(popularity) FROM movie_data_warehouse")
        stats["avg_popularity"] = cur.fetchone()[0]

        # Promedio de calificación
        cur.execute("SELECT AVG(vote_average) FROM movie_data_warehouse")
        stats["avg_rating"] = cur.fetchone()[0]

        # Distribución de géneros
        cur.execute("""
            SELECT genre, COUNT(*) as count
            FROM movie_data_warehouse
            WHERE genre IS NOT NULL AND genre != ''
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 10
        """)
        genres = cur.fetchall()
        stats["genres"] = [(g[0], g[1]) for g in genres]

        # Películas más populares
        cur.execute("""
            SELECT title, popularity, vote_average
            FROM movie_data_warehouse
            ORDER BY popularity DESC
            LIMIT 5
        """)
        top_movies = cur.fetchall()
        stats["top_movies"] = [(m[0], m[1], m[2]) for m in top_movies]

        # Películas con mejor ROI
        cur.execute("""
            SELECT title, roi, budget, revenue
            FROM movie_data_warehouse
            WHERE budget > 0
            ORDER BY roi DESC
            LIMIT 5
        """)
        best_roi = cur.fetchall()
        stats["best_roi"] = [(r[0], r[1], r[2], r[3]) for r in best_roi]

        # Consultas avanzadas

        # Correlación presupuesto-ingresos
        cur.execute("""
            SELECT
                CORR(budget, revenue) as budget_revenue_corr,
                CORR(budget, popularity) as budget_popularity_corr,
                CORR(revenue, popularity) as revenue_popularity_corr,
                CORR(vote_average, popularity) as rating_popularity_corr
            FROM movie_data_warehouse
            WHERE budget > 0 AND revenue > 0
        """)
        corr_result = cur.fetchone()
        if corr_result:
            stats["correlations"] = {
                "budget_revenue": float(corr_result[0] or 0),
                "budget_popularity": float(corr_result[1] or 0),
                "revenue_popularity": float(corr_result[2] or 0),
                "rating_popularity": float(corr_result[3] or 0)
            }

        # Análisis por año
        cur.execute("""
            SELECT
                release_year as year,
                COUNT(*) as count,
                AVG(popularity) as avg_popularity
            FROM movie_data_warehouse
            WHERE release_year IS NOT NULL
            GROUP BY release_year
            ORDER BY count DESC
            LIMIT 10
        """)
        years_data = cur.fetchall()
        stats["years"] = [(int(y[0]), y[1], float(y[2])) for y in years_data]

        cur.close()
        conn.close()

        return stats
    except Exception as e:
        st.error(f"Error al obtener estadísticas: {e}")
        return None
    
# Función para mostrar información de la película
def display_movie_card(movie):
    col1, col2 = st.columns([1, 2])

    with col1:
        if movie.get("poster_path"):
            st.image(f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}", width=200)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)

    with col2:
        st.subheader(movie["title"])

        # Información básica
        release_date = movie.get('release_date', 'Desconocida')
        if release_date and len(release_date) >= 4:
            year = release_date[:4]
            st.markdown(f"**Año de lanzamiento:** {year}")
        else:
            st.markdown(f"**Fecha de lanzamiento:** {release_date}")

        st.markdown(f"**Calificación:** {movie.get('vote_average', 0)}/10 ({movie.get('vote_count', 0)} votos)")

        if movie.get("runtime"):
            hours = movie["runtime"] // 60
            minutes = movie["runtime"] % 60
            st.markdown(f"**Duración:** {hours}h {minutes}min")

        # Géneros
        if movie.get("genres"):
            st.markdown("**Géneros:** " + ", ".join(movie["genres"]))

        # Presupuesto y recaudación
        if movie.get("budget"):
            st.markdown(f"**Presupuesto:** ${movie['budget']:,}")
        if movie.get("revenue"):
            st.markdown(f"**Recaudación:** ${movie['revenue']:,}")

        # ROI
        if movie.get("roi") is not None:
            roi_percentage = movie["roi"] * 100
            st.markdown(f"**ROI:** {roi_percentage:.2f}%")

        # Directores
        if movie.get("directors"):
            st.markdown("**Director(es):** " + ", ".join(movie["directors"]))

        # Reparto principal
        if movie.get("cast"):
            st.markdown("**Reparto principal:** " + ", ".join(movie["cast"]))

    # Sinopsis
    if movie.get("overview"):
        st.markdown("### Sinopsis")
        st.markdown(movie["overview"])


# Función para mostrar la predicción
# En tmdb_streamlit_app.py
def display_prediction(prediction, movie, metrics):
    st.markdown("## Predicción de Éxito")

    col1, col2, col3 = st.columns(3)

    with col1:
        prob = prediction["probability"]
        st.markdown(f"### Probabilidad de Éxito")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{prob:.1f}%</h1>",
                    unsafe_allow_html=True)

    with col2:
        success_level = prediction["success_level"]
        st.markdown(f"### Nivel de Éxito Proyectado")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{success_level}</h1>",
                    unsafe_allow_html=True)

    with col3:
        roi = prediction["estimated_roi"]
        st.markdown(f"### ROI Estimado")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{roi:.1f}x</h1>",
                    unsafe_allow_html=True)

    # Explicación de la predicción
    st.markdown("### Interpretación")
    
    # Crear explicación basada en el ROI estimado
    if roi >= 3:
        st.success("""
        **¡Alto potencial de éxito!** El modelo predice que esta película podría generar más de 3 veces su presupuesto,
        lo que la califica como un éxito financiero significativo. Las características de esta película están alineadas
        con las de películas que históricamente han tenido un alto retorno de inversión.
        """)
    elif roi >= 2:
        st.info("""
        **Buen potencial.** El modelo predice que esta película podría generar entre 2 y 3 veces su presupuesto.
        Aunque no alcanza el umbral estándar de "éxito extraordinario" (3x), sigue siendo una inversión potencialmente rentable.
        """)
    elif roi >= 1:
        st.warning("""
        **Potencial moderado.** El modelo predice que esta película podría recuperar su presupuesto, pero 
        probablemente no alcanzará el umbral de 3x considerado como un gran éxito en la industria. Podría ser rentable, 
        pero con un margen limitado.
        """)
    else:
        st.error("""
        **Potencial bajo.** El modelo predice que esta película podría tener dificultades para recuperar su presupuesto.
        Algunas características de la película no se alinean bien con patrones de películas exitosas pasadas.
        """)
    
    # Recordatorio sobre predicciones
    st.info("""
    **Nota:** Esta predicción se basa en características conocidas antes del estreno y tendencias históricas.
    Factores imprevistos como la recepción crítica, el marketing, eventos mundiales o la competencia en
    la fecha de estreno pueden alterar significativamente el desempeño real.
    """)

    # Factores que influyeron
    st.markdown("### Factores que Influyeron en la Predicción")

    factors_data = []
    if "feature_importance" in metrics:
        # Determinar si es lista o diccionario
        if isinstance(metrics["feature_importance"], list):
            # Convertir la lista de diccionarios a un solo diccionario
            importance_dict = {item['feature']: item['importance'] for item in metrics["feature_importance"]}
        elif isinstance(metrics["feature_importance"], dict):
            importance_dict = metrics["feature_importance"]
        else:
            importance_dict = {}
            
        for feature, value in importance_dict.items():
            # Mapear características a valores de la película
            feature_value = None
            if feature == 'budget_million':
                feature_value = float(movie.get('budget', 0)) / 1000000
            elif feature == 'runtime':
                feature_value = float(movie.get('runtime', 0))
            elif feature == 'main_genre' and (movie.get('genre') or movie.get('genres')):
                feature_value = movie.get('genre', movie.get('genres', ['Unknown'])[0] if isinstance(movie.get('genres', []), list) else 'Unknown')
            elif feature == 'vote_average':
                feature_value = float(movie.get('vote_average', 0))
            elif feature == 'optimal_runtime':
                runtime = float(movie.get('runtime', 0))
                feature_value = "Sí" if 90 <= runtime <= 150 else "No"
            elif feature == 'runtime_budget_ratio':
                budget_million = float(movie.get('budget', 0)) / 1000000
                runtime = float(movie.get('runtime', 0))
                feature_value = runtime / (budget_million + 1) if budget_million > 0 else 0
            elif feature == 'release_season':
                # Mapear valores numéricos a temporadas
                season_names = {1: "Verano", 2: "Navidad/Premios", 3: "Primavera", 4: "Otoño"}
                feature_value = season_names.get(movie.get('release_season', 4), "Desconocida")
            elif feature == 'release_year':
                feature_value = movie.get('release_year', "Desconocido")
            elif feature == 'budget_category':
                feature_value = movie.get('budget_category', "Desconocido")
            elif feature in movie:
                feature_value = movie.get(feature)

            if feature_value is not None and abs(value) > 0.001:  # Filtrar características de baja importancia
                factors_data.append({
                    "Factor": feature,
                    "Importancia": abs(value),
                    "Valor en la Película": feature_value
                })
    
    # Si no hay factores, usar algunos predeterminados
    if not factors_data:
        default_features = [
            {"Factor": "Presupuesto (millones $)", "Importancia": 0.3, "Valor en la Película": float(movie.get('budget', 0)) / 1000000},
            {"Factor": "Duración (minutos)", "Importancia": 0.2, "Valor en la Película": float(movie.get('runtime', 0))},
            {"Factor": "Género Principal", "Importancia": 0.25, "Valor en la Película": movie.get('genre', movie.get('genres', ['Unknown'])[0] if isinstance(movie.get('genres', []), list) else 'Unknown')},
            {"Factor": "Temporada de Estreno", "Importancia": 0.15, "Valor en la Película": "Verano" if movie.get('release_season', 4) == 1 else "Navidad/Premios" if movie.get('release_season', 4) == 2 else "Primavera" if movie.get('release_season', 4) == 3 else "Otoño"},
            {"Factor": "Año de Estreno", "Importancia": 0.1, "Valor en la Película": movie.get('release_year', "Desconocido")}
        ]
        factors_data = default_features

    # Mostrar tabla de factores
    df_factors = pd.DataFrame(factors_data)
    if not df_factors.empty:
        # Traducir nombres de factores
        factor_translations = {
            "budget_million": "Presupuesto (millones $)",
            "runtime": "Duración (minutos)",
            "main_genre": "Género Principal",
            "vote_average": "Calificación Promedio",
            "optimal_runtime": "Duración Óptima (90-150 min)",
            "runtime_budget_ratio": "Ratio Duración/Presupuesto",
            "release_season": "Temporada de Estreno",
            "release_year": "Año de Estreno",
            "budget_category": "Categoría de Presupuesto"
        }
        df_factors["Factor"] = df_factors["Factor"].map(lambda x: factor_translations.get(x, x))
        
        # Formatear valores específicos
        for i, row in df_factors.iterrows():
            if "Presupuesto" in row["Factor"]:
                df_factors.at[i, "Valor en la Película"] = f"${row['Valor en la Película']:,.2f}M"
            elif "Ratio" in row["Factor"] and isinstance(row['Valor en la Película'], (int, float)):
                df_factors.at[i, "Valor en la Película"] = f"{row['Valor en la Película']:.2f}"
            elif "Calificación" in row["Factor"] and isinstance(row['Valor en la Película'], (int, float)):
                df_factors.at[i, "Valor en la Película"] = f"{row['Valor en la Película']:.1f}/10"

        # Ordenar por importancia
        df_factors = df_factors.sort_values("Importancia", ascending=False)
        
        # Mostrar tabla
        st.table(df_factors)

    # Gráfico de importancia de factores
    if not df_factors.empty:
        fig = px.bar(
            df_factors,
            x="Importancia",
            y="Factor",
            orientation='h',
            color="Importancia",
            color_continuous_scale=["#E74C3C", "#F4D03F", "#1E8449"],
            title="Importancia de los Factores en la Predicción"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparación con otras películas
    st.markdown("### Comparación con la Media de la Industria")
    
    try:
        # Obtener promedios de la base de datos
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    AVG(vote_average) as avg_rating,
                    AVG(vote_count) as avg_votes,
                    AVG(runtime) as avg_runtime,
                    AVG(budget) as avg_budget,
                    AVG(revenue) as avg_revenue,
                    AVG(roi) as avg_roi
                FROM movie_data_warehouse
                WHERE vote_count > 10 AND budget > 0
            """)
            avg_row = cur.fetchone()

            if avg_row:
                # Obtener valores actuales de la película
                movie_budget = float(movie.get('budget', 0))
                movie_runtime = float(movie.get('runtime', 0))
                movie_vote_avg = float(movie.get('vote_average', 0))
                
                # Y valores promedio
                avg_budget = float(avg_row[3] or 0)
                avg_runtime = float(avg_row[2] or 0)
                avg_vote = float(avg_row[0] or 0)
                avg_roi = float(avg_row[5] or 0)

                # Crear tabla comparativa
                movie_data = {
                    "Métrica": ["Presupuesto (M$)", "Duración (min)", "Calificación", "ROI Esperado"],
                    "Esta Película": [
                        movie_budget / 1000000, 
                        movie_runtime, 
                        movie_vote_avg,
                        prediction["estimated_roi"]
                    ],
                    "Promedio Industria": [
                        avg_budget / 1000000, 
                        avg_runtime, 
                        avg_vote,
                        avg_roi
                    ]
                }
                
                # Crear DataFrame y mostrar
                df_comp = pd.DataFrame(movie_data)
                
                # Formatear para mejor visualización
                for i, row in df_comp.iterrows():
                    if row["Métrica"] == "Presupuesto (M$)":
                        df_comp.at[i, "Esta Película"] = f"${row['Esta Película']:.2f}M"
                        df_comp.at[i, "Promedio Industria"] = f"${row['Promedio Industria']:.2f}M"
                    elif row["Métrica"] == "Calificación":
                        df_comp.at[i, "Esta Película"] = f"{row['Esta Película']:.1f}/10"
                        df_comp.at[i, "Promedio Industria"] = f"{row['Promedio Industria']:.1f}/10"
                    elif row["Métrica"] == "ROI Esperado":
                        df_comp.at[i, "Esta Película"] = f"{row['Esta Película']:.2f}x"
                        df_comp.at[i, "Promedio Industria"] = f"{row['Promedio Industria']:.2f}x"
                
                st.table(df_comp)

                # Crear gráfico de comparación
                df_comp_plot = pd.DataFrame({
                    "Métrica": ["Presupuesto", "Duración", "Calificación", "ROI Esperado"],
                    "Esta Película": [
                        movie_budget / 1000000, 
                        movie_runtime / avg_runtime,  # Normalizado al promedio
                        movie_vote_avg / avg_vote,    # Normalizado al promedio
                        prediction["estimated_roi"] / max(avg_roi, 0.1)  # Normalizado al promedio
                    ],
                    "Promedio Industria": [
                        avg_budget / 1000000,
                        1.0,  # Normalizado
                        1.0,  # Normalizado
                        1.0   # Normalizado
                    ]
                })
                
                # Radar chart para comparación visual
                fig = go.Figure()

                # Añadir película actual
                fig.add_trace(go.Scatterpolar(
                    r=df_comp_plot["Esta Película"],
                    theta=df_comp_plot["Métrica"],
                    fill='toself',
                    name='Esta Película',
                    line_color='#3498DB'
                ))

                # Añadir promedio de la industria
                fig.add_trace(go.Scatterpolar(
                    r=df_comp_plot["Promedio Industria"],
                    theta=df_comp_plot["Métrica"],
                    fill='toself',
                    name='Promedio Industria',
                    line_color='#F4D03F'
                ))

                # Configuración del gráfico
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(df_comp_plot["Esta Película"].max(), df_comp_plot["Promedio Industria"].max()) * 1.1]
                        )),
                    showlegend=True,
                    title="Comparación con el Promedio de la Industria (Normalizado)"
                )
                st.plotly_chart(fig, use_container_width=True)

            cur.close()
            conn.close()
    except Exception as e:
        st.warning(f"No se pudo generar la comparación con la industria: {e}")
        
    # Recomendaciones para mejorar el ROI potencial
    st.markdown("### Recomendaciones para Mejorar el Potencial de Éxito")
    
    recommendations = []
    
    # Recomendaciones basadas en características específicas
    budget_million = float(movie.get('budget', 0)) / 1000000
    runtime = float(movie.get('runtime', 0))
    
    # Presupuesto
    if budget_million > 150 and prediction["estimated_roi"] < 2:
        recommendations.append("**Ajustar el presupuesto:** Películas con presupuestos muy altos necesitan recaudaciones enormes para ser exitosas. Considere reducir el presupuesto para mejorar el ROI potencial.")
    elif budget_million < 10 and prediction["estimated_roi"] < 2:
        recommendations.append("**Marketing y distribución:** Con presupuestos pequeños, es crucial maximizar la exposición. Considere estrategias de marketing innovadoras y de bajo costo para aumentar la visibilidad.")
    
    # Duración
    if runtime < 90 or runtime > 180:
        recommendations.append("**Ajustar la duración:** Las películas entre 90-150 minutos suelen tener mejor recepción en taquilla. Su duración actual podría afectar la experiencia del espectador o la cantidad de proyecciones diarias.")
    
    # Temporada de estreno
    release_season = movie.get('release_season', 4)
    if release_season == 4 and prediction["estimated_roi"] < 2:  # Otoño
        recommendations.append("**Considerar otra temporada de estreno:** El otoño es típicamente una temporada menos favorable para estrenos de alto ROI, salvo para películas específicas orientadas a premios.")
    
    # Si no hay suficientes recomendaciones específicas, añadir algunas generales
    if len(recommendations) < 2:
        if prediction["estimated_roi"] < 3:
            recommendations.append("**Enfoque en el marketing:** Las campañas de marketing efectivas pueden aumentar significativamente el rendimiento en taquilla, especialmente si se dirigen al público correcto.")
            recommendations.append("**Considera la presencia de estrellas:** La inclusión de actores reconocibles puede mejorar las posibilidades de éxito en taquilla.")
    
    # Siempre añadir esta recomendación general
    recommendations.append("**Análisis de competencia:** Investigue qué otras películas se estrenarán en fechas cercanas. La competencia directa, especialmente de películas similares, puede afectar significativamente el rendimiento.")
    
    # Mostrar recomendaciones
    for rec in recommendations:
        st.markdown(rec)

# Función para mostrar estadísticas del sistema
def display_system_stats():
    stats = get_db_stats()

    if not stats:
        st.warning("No se pudieron obtener estadísticas de la base de datos.")
        return

    st.markdown("## Estadísticas del Sistema")

    # Métricas principales
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de Películas", f"{stats['total_movies']:,}")

    with col2:
        st.metric("Popularidad Promedio", f"{stats['avg_popularity']:.2f}")

    with col3:
        st.metric("Calificación Promedio", f"{stats['avg_rating']:.1f}/10")

    # Correlaciones si están disponibles
    if 'correlations' in stats:
        st.markdown("### Correlaciones entre Variables")

        corr_data = []
        for key, value in stats['correlations'].items():
            if value is not None:
                name_map = {
                    "budget_revenue": "Presupuesto vs Ingresos",
                    "budget_popularity": "Presupuesto vs Popularidad",
                    "revenue_popularity": "Ingresos vs Popularidad",
                    "rating_popularity": "Calificación vs Popularidad"
                }
                corr_data.append({
                    "Variables": name_map.get(key, key),
                    "Correlación": value
                })

        if corr_data:
            df_corr = pd.DataFrame(corr_data)

            # Crear barras horizontales coloridas
            fig = px.bar(
                df_corr,
                x="Correlación",
                y="Variables",
                orientation='h',
                color="Correlación",
                color_continuous_scale="RdBu",
                range_color=[-1, 1],
                title="Correlación entre Variables",
                labels={"Correlación": "Coeficiente de Correlación (-1 a 1)"}
            )

            st.plotly_chart(fig, use_container_width=True)

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Distribución de géneros
        st.markdown("### Géneros más Populares")
        if stats["genres"]:
            genres_df = pd.DataFrame(stats["genres"], columns=["Género", "Cantidad"])
            fig = px.bar(
                genres_df,
                x="Cantidad",
                y="Género",
                orientation="h",
                color="Cantidad",
                color_continuous_scale="Viridis",
                title="Distribución de Géneros"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Nube de palabras de géneros
            try:
                genre_freq = {g[0]: g[1] for g in stats["genres"]}

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate_from_frequencies(genre_freq)

                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(plt)
            except Exception as e:
                st.warning(f"No se pudo generar la nube de palabras: {e}")

    with col2:
        # Películas más populares
        st.markdown("### Películas más Populares")
        if stats["top_movies"]:
            top_df = pd.DataFrame(stats["top_movies"], columns=["Título", "Popularidad", "Calificación"])
            fig = px.bar(
                top_df,
                x="Popularidad",
                y="Título",
                orientation="h",
                color="Calificación",
                color_continuous_scale="RdYlGn",
                title="Top 5 Películas por Popularidad"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Películas con mejor ROI
    st.markdown("### Películas con Mejor ROI")
    if stats["best_roi"]:
        roi_df = pd.DataFrame(stats["best_roi"], columns=["Título", "ROI", "Presupuesto", "Ingresos"])
        roi_df["ROI %"] = roi_df["ROI"] * 100
        roi_df["Presupuesto (M$)"] = roi_df["Presupuesto"] / 1000000
        roi_df["Ingresos (M$)"] = roi_df["Ingresos"] / 1000000

        fig = px.bar(
            roi_df,
            x="ROI %",
            y="Título",
            orientation="h",
            color="ROI %",
            color_continuous_scale="Viridis",
            title="Top 5 Películas por ROI",
            labels={"ROI %": "Retorno de Inversión (%)"}
        )

        st.plotly_chart(fig, use_container_width=True)


# Interfaz principal de la aplicación
def main():
    # Configuración de la barra lateral
    st.sidebar.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg", width=150)
    st.sidebar.title("TMDB Movie Success Predictor")

    # Cargar modelo
    model, metrics, feature_names = load_latest_model()

    # Menú de navegación
    menu = st.sidebar.selectbox(
        "Menú",
        ["Inicio", "Buscar Película", "Analizar Nueva Película", "Estadísticas"]
    )

    if menu == "Inicio":
        st.title("🎬 Predictor de Éxito de Películas")
        st.markdown("""
        Esta aplicación te permite analizar películas y predecir su éxito potencial utilizando
        modelos de Machine Learning entrenados con datos reales de TMDB.
        
        ### Características principales:
        - **Búsqueda de películas** por título y análisis de su potencial de éxito
        - **Predicción personalizada** para películas nuevas o hipotéticas
        - **Estadísticas** sobre tendencias, géneros y factores de éxito
        
        ### Cómo usar:
        1. Utiliza el menú de la izquierda para navegar entre las diferentes secciones
        2. En "Buscar Película", puedes encontrar películas existentes y ver su análisis
        3. En "Analizar Nueva Película", puedes crear una película hipotética y predecir su éxito
        4. En "Estadísticas", puedes explorar tendencias y patrones del mundo cinematográfico
        
        Este sistema utiliza un modelo de Machine Learning que considera factores como presupuesto,
        duración, género, votos y otros indicadores para predecir la popularidad y éxito potencial.
        """)

        # Mostrar algunas estadísticas en la página principal
        try:
            stats = get_db_stats()
            if stats:
                st.markdown("---")
                st.markdown("### Estadísticas Generales")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Películas Analizadas", f"{stats['total_movies']:,}")
                with col2:
                    st.metric("Popularidad Promedio", f"{stats['avg_popularity']:.2f}")
                with col3:
                    st.metric("Calificación Promedio", f"{stats['avg_rating']:.1f}/10")
        except Exception as e:
            st.error(f"Error al cargar estadísticas: {e}")

    elif menu == "Buscar Película":
        st.title("🔍 Buscar y Analizar Película")

        # Buscador
        search_query = st.text_input("Buscar película por título", "")

        if search_query:
            results = search_movie_by_title(search_query)

            if results:
                st.markdown(f"### Resultados para: '{search_query}'")

                for i, movie in enumerate(results):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{movie['title']}**")
                    with col2:
                        st.write(f"Popularidad: {movie.get('popularity', 'N/A')}")
                    with col3:
                        if st.button(f"Analizar", key=f"btn_analyze_{i}"):
                            st.session_state.selected_movie_id = movie['id']
                            st.rerun()

            else:
                st.warning("No se encontraron resultados.")

        # Si hay una película seleccionada, mostrar sus detalles
        if 'selected_movie_id' in st.session_state:
            movie_details = get_movie_details(st.session_state.selected_movie_id)

            if movie_details:
                # Mostrar tarjeta de película
                display_movie_card(movie_details)

                # Predecir éxito
                prediction = predict_movie_success(movie_details, model, feature_names)

                # Mostrar predicción
                display_prediction(prediction, movie_details, metrics)

                # Botón para limpiar selección
                if st.button("← Volver a resultados"):
                    del st.session_state.selected_movie_id
                    st.rerun()
            else:
                st.error("No se pudieron obtener detalles de la película seleccionada.")

    elif menu == "Analizar Nueva Película":
        st.title("🎯 Analizar Nueva Película")

        st.markdown("""
        En esta sección puedes crear una película hipotética y predecir su éxito potencial.
        Completa los siguientes campos con la información de tu película:
        """)

        # Formulario para datos de nueva película
        with st.form("new_movie_form"):
            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input("Título", "Mi Película")
                genres = st.multiselect(
                    "Géneros",
                    ["Acción", "Aventura", "Animación", "Comedia", "Crimen", "Documental",
                     "Drama", "Familia", "Fantasía", "Historia", "Terror", "Música",
                     "Misterio", "Romance", "Ciencia ficción", "Película de TV", "Suspense",
                     "Bélica", "Western"]
                )
                runtime = st.slider("Duración (minutos)", 60, 240, 120)
                release_year = st.slider("Año de lanzamiento", 2000, 2030, 2023)

            with col2:
                budget = st.number_input("Presupuesto (millones $)", 0.1, 500.0, 100.0, step=0.1)
                vote_average = st.slider("Calificación esperada (0-10)", 0.0, 10.0, 7.0, 0.1)
                vote_count = st.slider("Número esperado de votos", 0, 10000, 1000)
                director = st.text_input("Director", "Director")

            submit_button = st.form_submit_button("Analizar")

            if submit_button:
                # Crear objeto de película simulada
                new_movie = {
                    "title": title,
                    "genres": genres,
                    "runtime": runtime,
                    "release_date": f"{release_year}-01-01",
                    "budget": budget * 1000000,  # Convertir a unidades
                    "revenue": 0,  # No conocemos los ingresos todavía
                    "vote_average": vote_average,
                    "vote_count": vote_count,
                    "directors": [director],
                    "popularity": 0,  # No conocemos la popularidad todavía
                    "release_year": release_year
                }

                # Guardar en session state
                st.session_state.new_movie = new_movie
                st.rerun()

        # Si hay una película creada, mostrar predicción
        if 'new_movie' in st.session_state:
            new_movie = st.session_state.new_movie

            st.markdown("---")
            st.markdown("### Película a Analizar")

            # Mostrar datos básicos
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Título:** {new_movie['title']}")
                st.markdown(f"**Géneros:** {', '.join(new_movie['genres'])}")
                st.markdown(f"**Director:** {', '.join(new_movie['directors'])}")
                st.markdown(f"**Año:** {new_movie['release_date'][:4]}")

            with col2:
                st.markdown(f"**Duración:** {new_movie['runtime']} minutos")
                st.markdown(f"**Presupuesto:** ${new_movie['budget']:,.2f}")
                st.markdown(f"**Calificación esperada:** {new_movie['vote_average']}/10")
                st.markdown(f"**Votos esperados:** {new_movie['vote_count']}")

            # Predecir éxito
            prediction = predict_movie_success(new_movie, model, feature_names)

            # Mostrar predicción
            display_prediction(prediction, new_movie, metrics)

            # Mostrar recomendaciones basadas en la predicción
            st.markdown("### Recomendaciones para Mejorar")

            if prediction["is_successful"]:
                st.success("¡Tu película tiene buenas posibilidades de éxito! Aquí hay algunas recomendaciones para maximizar su potencial:")
            else:
                st.warning("Tu película podría enfrentar desafíos para alcanzar el éxito. Considera estos ajustes:")

            # Recomendaciones específicas basadas en factores
            recommend_cols = st.columns(2)

            with recommend_cols[0]:
                if new_movie["budget"] < 50000000 and prediction["popularity_score"] < 15:
                    st.markdown("- **Aumenta el presupuesto**: Un mayor presupuesto podría mejorar la calidad de producción")

                if new_movie["runtime"] < 90 or new_movie["runtime"] > 180:
                    st.markdown("- **Ajusta la duración**: Las películas entre 90-150 minutos suelen tener mejor recepción")

                if not new_movie["genres"] or len(new_movie["genres"]) < 2:
                    st.markdown("- **Diversifica géneros**: Considera combinar géneros para ampliar la audiencia")

            with recommend_cols[1]:
                if prediction["popularity_score"] < 10:
                    st.markdown("- **Marketing**: Invierte más en campañas de marketing para aumentar la visibilidad")

                if new_movie["vote_average"] < 7:
                    st.markdown("- **Calidad del guión**: Considera mejorar la historia para aumentar la calificación esperada")

                # Recomendación general
                st.markdown("- **Análisis competitivo**: Estudia películas similares exitosas para inspiración")

            # Botón para limpiar y crear nueva película
            if st.button("Crear Nueva Película"):
                del st.session_state.new_movie
                st.rerun()

    elif menu == "Estadísticas":
        st.title("📊 Estadísticas y Análisis")

        # Mostrar estadísticas del sistema
        display_system_stats()


# Ejecutar aplicación
if __name__ == "__main__":
    main()