from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

import json
import time
import os
import requests
import pandas as pd
import numpy as np
import sys
import logging
import traceback
import subprocess

# Add this new function to store the trained model
def store_ml_model(**kwargs):
    """Store the trained ML model in a specific location for Streamlit"""
    import shutil
    import os
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    ti = kwargs['ti']
    model_dir = ti.xcom_pull(key='model_dir', task_ids='train_ml_model')
    model_metrics = ti.xcom_pull(key='model_metrics', task_ids='train_ml_model')
    
    if not model_dir or not os.path.exists(model_dir):
        logger.error("No model directory found to store")
        return None
    
    # Define the permanent storage location
    permanent_model_dir = "/opt/airflow/data/movie_analytics/ml_models/latest_model"
    os.makedirs(permanent_model_dir, exist_ok=True)
    
    try:
        # Copy model files to permanent location
        for filename in os.listdir(model_dir):
            src_file = os.path.join(model_dir, filename)
            dst_file = os.path.join(permanent_model_dir, filename)
            shutil.copy2(src_file, dst_file)
        
        logger.info(f"Model stored permanently in {permanent_model_dir}")
        
        # Optional: Create a symlink for easy access
        latest_symlink = "/opt/airflow/data/movie_analytics/latest_ml_model"
        if os.path.exists(latest_symlink):
            os.unlink(latest_symlink)
        os.symlink(permanent_model_dir, latest_symlink)
        
        return permanent_model_dir
    
    except Exception as e:
        logger.error(f"Error storing model: {e}")
        return None

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuraciones
TMDB_API_KEY = "e8e1dae84a0345bd3ec23e3030905258"
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlOGUxZGFlODRhMDM0NWJkM2VjMjNlMzAzMDkwNTI1OCIsIm5iZiI6MTc0MTkxMzEzNi4xMjEsInN1YiI6IjY3ZDM3YzMwYmY0ODE4ODU0YzY0ZTVmNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tdq930_qLqYbYAqwVLl3Tdw84HdEsZtM41CX_9-lJNU"
KAFKA_TOPIC = "tmdb_data"
KAFKA_BROKER = "kafka:9093"  # Conexión dentro del contenedor de Docker
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Villeta-11"
POSTGRES_HOST = "movie_postgres"  # Nombre del servicio en Docker
POSTGRES_PORT = "5433"  # Puerto dentro de Docker
OUTPUT_DIR = "/opt/airflow/data/movie_analytics"
STREAMLIT_PATH = "/opt/airflow/data/tmdb_streamlit_app.py"

# Crear directorio para datos
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clase personalizada para ejecutar Streamlit
class StreamlitOperator(BaseOperator):
    """
    Operador personalizado para iniciar una aplicación Streamlit
    como parte de un DAG de Airflow.
    """
    
    @apply_defaults
    def __init__(
        self,
        script_path,
        port=8501,
        host="localhost",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.script_path = script_path
        self.port = port
        self.host = host
    
    def execute(self, context):
        """
        Inicia la aplicación Streamlit.
        Nota: Este operador no mantiene el proceso de Streamlit ejecutándose
        más allá de la duración de la tarea. Para uso en producción,
        considera un enfoque con systemd, supervisor, o un contenedor dedicado.
        """
        script_dir = os.path.dirname(os.path.abspath(self.script_path))
        script_name = os.path.basename(self.script_path)
        
        self.log.info(f"Iniciando aplicación Streamlit: {self.script_path} en puerto {self.port}")
        
        # Verificar que el script existe
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"No se encontró el script de Streamlit: {self.script_path}")
        
        # Iniciar Streamlit como un proceso detached
        cmd = [
            "streamlit", "run", script_name,
            "--server.port", str(self.port),
            "--server.address", self.host,
            "--server.headless", "true",
            "--browser.serverAddress", self.host,
            "--browser.gatherUsageStats", "false"
        ]
        
        try:
            # Para uso en producción, considerar usar nohup o similar
            # Aquí usamos subprocess para iniciar el proceso como demonio
            with open(os.path.join(script_dir, "streamlit.log"), "w") as log_file:
                subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=script_dir,
                    start_new_session=True
                )
            
            self.log.info(f"Aplicación Streamlit iniciada en http://{self.host}:{self.port}")
            return f"Aplicación Streamlit disponible en http://{self.host}:{self.port}"
            
        except Exception as e:
            self.log.error(f"Error al iniciar Streamlit: {e}")
            raise

# Función para crear el contenido de la app Streamlit
def create_streamlit_app_content():
    return '''
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
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5433"

# Configuración de TMDB API
TMDB_API_KEY = "e8e1dae84a0345bd3ec23e3030905258"
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlOGUxZGFlODRhMDM0NWJkM2VjMjNlMzAzMDkwNTI1OCIsIm5iZiI6MTc0MTkxMzEzNi4xMjEsInN1YiI6IjY3ZDM3YzMwYmY0ODE4ODU0YzY0ZTVmNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tdq930_qLqYbYAqwVLl3Tdw84HdEsZtM41CX_9-lJNU"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Función para cargar el modelo más reciente
@st.cache_resource
def load_latest_model():
    try:
        # Buscar la carpeta de modelo más reciente
        model_dirs = glob.glob("data/movie_analytics/ml_model_v2_*") + glob.glob("data/movie_analytics/batch_*/movie_success_model.pkl")
        
        if not model_dirs:
            st.warning("No se encontraron modelos entrenados. Utilizando un modelo simulado.")
            # Crear un modelo dummy para simular si no hay uno
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.coef_ = np.array([0.5, 0.3, 0.2, 0.1])  # Coeficientes simulados
            model.intercept_ = 5.0
            metrics = {
                "r2": 0.75,
                "mse": 2.5,
                "feature_importance": {
                    "vote_average": 0.5,
                    "vote_count": 0.3, 
                    "runtime": 0.2,
                    "budget": 0.1
                }
            }
            return model, metrics, ["vote_average", "vote_count", "runtime", "budget"]
        
        # Encontrar el directorio más reciente
        latest_model_dir = max(model_dirs)
        
        # Determinar si es un directorio o un archivo directamente
        if latest_model_dir.endswith('.pkl'):
            model_path = latest_model_dir
            metrics_path = os.path.join(os.path.dirname(latest_model_dir), "feature_importance.csv")
            directory = os.path.dirname(latest_model_dir)
        else:
            model_path = os.path.join(latest_model_dir, "model.pkl")
            metrics_path = os.path.join(latest_model_dir, "metrics.json")
            directory = latest_model_dir
        
        # Cargar modelo
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            st.warning(f"No se encontró el archivo del modelo en {model_path}. Utilizando un modelo simulado.")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.coef_ = np.array([0.5, 0.3, 0.2, 0.1])
            model.intercept_ = 5.0
        
        # Cargar métricas
        metrics = {}
        feature_names = ["vote_average", "vote_count", "runtime", "budget"]
        
        if os.path.exists(metrics_path):
            if metrics_path.endswith('.json'):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                if 'feature_importance' in metrics and isinstance(metrics['feature_importance'], dict):
                    feature_names = list(metrics['feature_importance'].keys())
            elif metrics_path.endswith('.csv'):
                try:
                    feat_importance = pd.read_csv(metrics_path)
                    if 'feature' in feat_importance.columns and 'importance' in feat_importance.columns:
                        metrics['feature_importance'] = dict(zip(feat_importance['feature'], feat_importance['importance']))
                        feature_names = feat_importance['feature'].tolist()
                except Exception as e:
                    st.warning(f"Error al cargar métricas desde CSV: {e}")
        
        st.success(f"Modelo cargado correctamente desde {directory}")
        return model, metrics, feature_names
    
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        # Crear un modelo dummy para simulación
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.coef_ = np.array([0.5, 0.3, 0.2, 0.1])
        model.intercept_ = 5.0
        metrics = {"r2": 0.75, "mse": 2.5}
        return model, metrics, ["vote_average", "vote_count", "runtime", "budget"]

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
            cur.execute("SELECT tmdb_id, title, release_date, popularity, vote_average FROM movies WHERE title ILIKE %s LIMIT %s", 
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
    # Primero buscar en la base de datos
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT tmdb_id, title, original_title, overview, release_date, 
                       popularity, vote_average, vote_count, budget, revenue, 
                       runtime, genres_str, directors_str, cast_str, companies_str,
                       roi, popularity_category, rating_category
                FROM movies WHERE tmdb_id = %s
            """, (movie_id,))
            row = cur.fetchone()
            
            if row:
                movie = {
                    "id": row[0],
                    "title": row[1],
                    "original_title": row[2],
                    "overview": row[3],
                    "release_date": row[4],
                    "popularity": row[5],
                    "vote_average": row[6],
                    "vote_count": row[7],
                    "budget": row[8],
                    "revenue": row[9],
                    "runtime": row[10],
                    "genres": row[11].split(',') if row[11] else [],
                    "directors": row[12].split(',') if row[12] else [],
                    "cast": row[13].split(',') if row[13] else [],
                    "production_companies": row[14].split(',') if row[14] else [],
                    "roi": row[15],
                    "popularity_category": row[16],
                    "rating_category": row[17],
                    "source": "Database"
                }
                
                cur.close()
                conn.close()
                return movie
            
            cur.close()
            conn.close()
    except Exception as e:
        st.warning(f"Error al buscar en la base de datos: {e}")
    
    # Si no está en la base de datos, consultar API de TMDB
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        headers = {
            "Authorization": f"Bearer {TMDB_TOKEN}",
            "Content-Type": "application/json;charset=utf-8"
        }
        params = {
            "language": "es-ES",
            "append_to_response": "credits"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            
            # Crear diccionario con los datos de la API
            movie = {
                "id": data.get("id"),
                "title": data.get("title"),
                "original_title": data.get("original_title"),
                "overview": data.get("overview"),
                "release_date": data.get("release_date"),
                "popularity": data.get("popularity"),
                "vote_average": data.get("vote_average"),
                "vote_count": data.get("vote_count"),
                "budget": data.get("budget"),
                "revenue": data.get("revenue"),
                "runtime": data.get("runtime"),
                "genres": [genre.get("name") for genre in data.get("genres", [])],
                "directors": [person.get("name") for person in data.get("credits", {}).get("crew", []) if person.get("job") == "Director"],
                "cast": [person.get("name") for person in data.get("credits", {}).get("cast", [])[:5]],
                "production_companies": [company.get("name") for company in data.get("production_companies", [])],
                "poster_path": data.get("poster_path"),
                "backdrop_path": data.get("backdrop_path"),
                "source": "TMDB API"
            }
            
            # Calcular ROI si tenemos budget y revenue
            if movie["budget"] and movie["revenue"] and movie["budget"] > 0:
                movie["roi"] = (movie["revenue"] - movie["budget"]) / movie["budget"]
            else:
                movie["roi"] = None
            
            # Calcular categorías
            if movie["popularity"] is not None:
                if movie["popularity"] > 20:
                    movie["popularity_category"] = "Muy Alta"
                elif movie["popularity"] > 10:
                    movie["popularity_category"] = "Alta"
                elif movie["popularity"] > 5:
                    movie["popularity_category"] = "Media"
                else:
                    movie["popularity_category"] = "Baja"
            else:
                movie["popularity_category"] = "Desconocida"
            
            if movie["vote_average"] is not None:
                if movie["vote_average"] >= 8:
                    movie["rating_category"] = "Excelente"
                elif movie["vote_average"] >= 6:
                    movie["rating_category"] = "Buena"
                elif movie["vote_average"] >= 4:
                    movie["rating_category"] = "Regular"
                else:
                    movie["rating_category"] = "Mala"
            else:
                movie["rating_category"] = "Desconocida"
            
            return movie
    except Exception as e:
        st.error(f"Error al consultar TMDB API: {e}")
    
    return None

# Función para predecir el éxito de una película
def predict_movie_success(movie, model, feature_names):
    try:
        # Preparar features para predicción
        features = {}
        for feature in feature_names:
            if feature in movie and movie[feature] is not None:
                features[feature] = movie[feature]
            else:
                features[feature] = 0
        
        # Convertir a DataFrame
        X = pd.DataFrame([list(features.values())], columns=feature_names)
        
        # Hacer predicción
        popularity_prediction = model.predict(X)[0]
        
        # Categorizar resultado
        if popularity_prediction > 20:
            success_level = "Muy Alta"
            success_color = "#1E8449"  # Verde oscuro
        elif popularity_prediction > 10:
            success_level = "Alta"
            success_color = "#58D68D"  # Verde claro
        elif popularity_prediction > 5:
            success_level = "Media"
            success_color = "#F4D03F"  # Amarillo
        else:
            success_level = "Baja"
            success_color = "#E74C3C"  # Rojo
        
        # Resultado
        prediction_result = {
            "popularity_score": float(popularity_prediction),
            "success_level": success_level,
            "success_color": success_color,
            "is_successful": popularity_prediction > 10
        }
        
        return prediction_result
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return {
            "popularity_score": 0,
            "success_level": "Error",
            "success_color": "#E74C3C",
            "is_successful": False
        }

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
        cur.execute("SELECT COUNT(*) FROM movies")
        stats["total_movies"] = cur.fetchone()[0]
        
        # Promedio de popularidad
        cur.execute("SELECT AVG(popularity) FROM movies")
        stats["avg_popularity"] = cur.fetchone()[0]
        
        # Promedio de calificación
        cur.execute("SELECT AVG(vote_average) FROM movies")
        stats["avg_rating"] = cur.fetchone()[0]
        
        # Distribución de géneros
        cur.execute("""
            SELECT genre, COUNT(*) as count
            FROM movie_data_warehouse
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 10
        """)
        genres = cur.fetchall()
        stats["genres"] = [(g[0], g[1]) for g in genres]
        
        # Películas más populares
        cur.execute("""
            SELECT title, popularity, vote_average
            FROM movies
            ORDER BY popularity DESC
            LIMIT 5
        """)
        top_movies = cur.fetchall()
        stats["top_movies"] = [(m[0], m[1], m[2]) for m in top_movies]
        
        # Películas con mejor ROI
        cur.execute("""
            SELECT title, roi, budget, revenue
            FROM movies
            WHERE budget > 0
            ORDER BY roi DESC
            LIMIT 5
        """)
        best_roi = cur.fetchall()
        stats["best_roi"] = [(r[0], r[1], r[2], r[3]) for r in best_roi]
        
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
        st.markdown(f"**Fecha de lanzamiento:** {movie.get('release_date', 'Desconocida')}")
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
def display_prediction(prediction, movie, metrics):
    st.markdown("## Predicción de Éxito")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = prediction["popularity_score"]
        st.markdown(f"### Puntuación de Popularidad")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{score:.2f}</h1>", unsafe_allow_html=True)
    
    with col2:
        success_level = prediction["success_level"]
        st.markdown(f"### Nivel de Éxito")
        st.markdown(f"<h1 style='text-align: center; color: {prediction['success_color']};'>{success_level}</h1>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"### Probabilidad de Éxito")
        is_successful = "Alta" if prediction["is_successful"] else "Baja"
        color = "#1E8449" if prediction["is_successful"] else "#E74C3C"
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{is_successful}</h1>", unsafe_allow_html=True)
    
    # Factores que influyeron
    st.markdown("### Factores que Influyeron en la Predicción")
    
    factors_data = []
    if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
        importance = metrics["feature_importance"]
        for feature, value in importance.items():
            if feature in movie:
                factors_data.append({
                    "Factor": feature,
                    "Importancia": abs(value),
                    "Valor en la Película": movie.get(feature, 0)
                })
    else:
        # Si no hay importancia de características, usar valores por defecto
        for feature in ["vote_average", "vote_count", "runtime", "budget"]:
            if feature in movie:
                factors_data.append({
                    "Factor": feature,
                    "Importancia": 0.25,  # Valor por defecto
                    "Valor en la Película": movie.get(feature, 0)
                })
    
    # Mostrar tabla de factores
    df_factors = pd.DataFrame(factors_data)
    if not df_factors.empty:
        # Traducir nombres de factores
        factor_translations = {
            "vote_average": "Calificación Promedio", 
            "vote_count": "Número de Votos",
            "runtime": "Duración (minutos)",
            "budget": "Presupuesto ($)"
        }
        df_factors["Factor"] = df_factors["Factor"].map(factor_translations).fillna(df_factors["Factor"])
        
        # Formatear valores específicos
        for i, row in df_factors.iterrows():
            if "Presupuesto" in row["Factor"]:
                df_factors.at[i, "Valor en la Película"] = f"${row['Valor en la Película']:,.2f}"
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
                    AVG(popularity) as avg_popularity
                FROM movies
            """)
            avg_row = cur.fetchone()
            
            if avg_row:
                avg_data = {
                    "Calificación": [movie.get("vote_average", 0), avg_row[0]],
                    "Votos": [movie.get("vote_count", 0), avg_row[1]],
                    "Duración": [movie.get("runtime", 0), avg_row[2]],
                    "Presupuesto": [movie.get("budget", 0)/1000000, avg_row[3]/1000000],  # En millones
                    "Popularidad": [movie.get("popularity", 0), avg_row[4]]
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
            
            cur.close()
            conn.close()
    except Exception as e:
        st.warning(f"No se pudo generar la comparación: {e}")

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
        fig = px.bar(
            roi_df,
            x="ROI %",
            y="Título",
            orientation="h",
            color="ROI %",
            color_continuous_scale="Greens",
            title="Top 5 Películas por ROI (Retorno de Inversión)"
        )
        st.plotly_chart(fig, use_container_width=True)

# UI Principal
def main():
    # Cargar modelo
    model, metrics, feature_names = load_latest_model()
    
    # Título y descripción
    st.title("🎬 Predictor de Éxito de Películas TMDB")
    st.markdown("""
    Esta aplicación predice el éxito potencial de películas utilizando un modelo de machine learning
    entrenado con datos de la API de TMDB. El éxito se mide principalmente por la popularidad
    proyectada de la película.
    """)
    
    # Sidebar
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Ir a:", ["Predicción", "Estadísticas"])
    
    if page == "Predicción":
        st.header("Predicción de Éxito de Películas")
        
        # Opciones de búsqueda
        search_option = st.radio("Buscar película por:", ["Título", "ID de TMDB"])
        
        if search_option == "Título":
            movie_title = st.text_input("Ingrese el título de la película:")
            
            if movie_title:
                with st.spinner("Buscando películas..."):
                    movies = search_movie_by_title(movie_title)
                
                if not movies:
                    st.warning(f"No se encontraron películas con el título '{movie_title}'")
                else:
                    st.success(f"Se encontraron {len(movies)} resultados:")
                    
                    # Crear opciones para selección
                    movie_options = {f"{m['title']} ({m['release_date'][:4] if m.get('release_date') else 'N/A'})": m['id'] for m in movies}
                    selected_movie = st.selectbox("Seleccione una película:", list(movie_options.keys()))
                    
                    if selected_movie:
                        movie_id = movie_options[selected_movie]
                        with st.spinner("Obteniendo detalles de la película..."):
                            movie = get_movie_details(movie_id)
                        
                        if movie:
                            # Mostrar información de la película
                            st.markdown("---")
                            display_movie_card(movie)
                            
                            # Hacer predicción
                            with st.spinner("Realizando predicción..."):
                                prediction = predict_movie_success(movie, model, feature_names)
                            
                            # Mostrar resultados de predicción
                            st.markdown("---")
                            display_prediction(prediction, movie, metrics)
                        else:
                            st.error("No se pudieron obtener los detalles de la película seleccionada.")
        
        elif search_option == "ID de TMDB":
            movie_id = st.text_input("Ingrese el ID de TMDB:")
            
            if movie_id and movie_id.isdigit():
                with st.spinner("Obteniendo detalles de la película..."):
                    movie = get_movie_details(int(movie_id))
                
                if movie:
                    # Mostrar información de la película
                    st.markdown("---")
                    display_movie_card(movie)
                    
                    # Hacer predicción
                    with st.spinner("Realizando predicción..."):
                        prediction = predict_movie_success(movie, model, feature_names)
                    
                    # Mostrar resultados de predicción
                    st.markdown("---")
                    display_prediction(prediction, movie, metrics)
                else:
                    st.error(f"No se encontró ninguna película con el ID {movie_id}")
    
    elif page == "Estadísticas":
        st.header("Estadísticas del Sistema")
        
        with st.spinner("Cargando estadísticas..."):
            display_system_stats()
        
        # Información del modelo
        st.markdown("---")
        st.markdown("## Información del Modelo")
        
        if "r2" in metrics:
            st.metric("Coeficiente de Determinación (R²)", f"{metrics['r2']:.3f}")
        
        if "mse" in metrics:
            st.metric("Error Cuadrático Medio (MSE)", f"{metrics['mse']:.3f}")
        
        # Características usadas en el modelo
        st.markdown("### Características Utilizadas")
        
        feature_translations = {
            "vote_average": "Calificación Promedio", 
            "vote_count": "Número de Votos",
            "runtime": "Duración (minutos)",
            "budget": "Presupuesto ($)"
        }
        
        translated_features = [feature_translations.get(f, f) for f in feature_names]
        st.write(", ".join(translated_features))
        
        # Importancia de características
        if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
            st.markdown("### Importancia de Características")
            
            # Crear DataFrame para visualización
            importance_data = []
            for feature, value in metrics["feature_importance"].items():
                importance_data.append({
                    "Característica": feature_translations.get(feature, feature),
                    "Importancia": abs(value)
                })
            
            df_importance = pd.DataFrame(importance_data)
            if not df_importance.empty:
                # Ordenar por importancia
                df_importance = df_importance.sort_values("Importancia", ascending=False)
                
                # Crear gráfico
                fig = px.bar(
                    df_importance,
                    x="Importancia",
                    y="Característica",
                    orientation="h",
                    color="Importancia",
                    color_continuous_scale="RdYlGn",
                    title="Importancia Relativa de Cada Característica"
                )
                st.plotly_chart(fig, use_container_width=True)

# Ejecutar aplicación
if __name__ == "__main__":
    main()
'''

# Función para verificar y crear las tablas en PostgreSQL
def create_postgres_tables():
    """Verifica y crea las tablas necesarias en PostgreSQL"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        
        cur = conn.cursor()
        
        # Crear esquema si no existe
        cur.execute("CREATE SCHEMA IF NOT EXISTS public;")
        conn.commit()
        
        # Crear tabla movies si no existe
        cur.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id SERIAL PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            title VARCHAR(255),
            original_title VARCHAR(255),
            overview TEXT,
            release_date VARCHAR(50),
            popularity FLOAT,
            vote_average FLOAT,
            vote_count INTEGER,
            budget BIGINT,
            revenue BIGINT,
            runtime INTEGER,
            adult BOOLEAN,
            genres_str TEXT,
            directors_str TEXT,
            cast_str TEXT,
            companies_str TEXT,
            roi FLOAT,
            popularity_category VARCHAR(50),
            rating_category VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Crear tabla de géneros
        cur.execute("""
        CREATE TABLE IF NOT EXISTS genres (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE
        );
        """)
        
        # Crear tabla de directores
        cur.execute("""
        CREATE TABLE IF NOT EXISTS directors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE
        );
        """)
        
        # Crear tabla de actores
        cur.execute("""
        CREATE TABLE IF NOT EXISTS actors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE
        );
        """)
        
        # Crear tabla de data warehouse
        cur.execute("""
        CREATE TABLE IF NOT EXISTS movie_data_warehouse (
            id SERIAL PRIMARY KEY,
            tmdb_id INTEGER,
            title VARCHAR(255),
            release_date VARCHAR(50),
            release_year INTEGER,
            genre VARCHAR(100),
            budget BIGINT,
            revenue BIGINT,
            runtime INTEGER,
            popularity FLOAT,
            vote_average FLOAT,
            vote_count INTEGER,
            roi FLOAT,
            director VARCHAR(100),
            popularity_level VARCHAR(50),
            rating_level VARCHAR(50),
            is_profitable BOOLEAN,
            data_date DATE DEFAULT CURRENT_DATE
        );
        """)
        
        # Confirmar cambios
        conn.commit()
        logger.info("Tablas verificadas y creadas en PostgreSQL")
        
        # Listar las tablas existentes
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        logger.info("Tablas existentes:")
        for table in tables:
            logger.info(f"- {table[0]}")
        
        # Cerrar conexión
        cur.close()
        conn.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error al crear tablas en PostgreSQL: {e}")
        logger.error(traceback.format_exc())
        return False

# Función preparación inicial
def setup_environment(**kwargs):
    """Prepara el entorno para el pipeline"""
    
    logger.info("Preparando entorno para el pipeline TMDB...")
    
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Directorio de salida creado: {OUTPUT_DIR}")
    
    # Verificar y crear tablas en PostgreSQL
    tables_created = create_postgres_tables()
    
    if tables_created:
        logger.info("Entorno preparado correctamente")
    else:
        logger.warning("Hubo problemas preparando el entorno, pero continuaremos")
    
    return "setup_completed"

# Función para preparar el script de Streamlit
def setup_streamlit_app(**kwargs):
    """Verifica o crea el archivo de la aplicación Streamlit"""
    
    # Verificar si existe el script de Streamlit
    if not os.path.exists(STREAMLIT_PATH):
        logger.info(f"Creando archivo de aplicación Streamlit en {STREAMLIT_PATH}")
        
        # Obtener el contenido del script
        script_content = create_streamlit_app_content()
        
        # Guardar el contenido en el archivo
        with open(STREAMLIT_PATH, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Archivo de Streamlit creado en {STREAMLIT_PATH}")
        
        # Instalar dependencias necesarias para Streamlit
        try:
            logger.info("Instalando dependencias necesarias para Streamlit...")
            
            subprocess.run([
                "pip", "install", "streamlit", "plotly", "matplotlib", "seaborn", 
                "psycopg2-binary", "scikit-learn"
            ], check=True)
            
            logger.info("Dependencias instaladas correctamente")
        except Exception as e:
            logger.warning(f"Error al instalar dependencias: {e}")
            logger.warning("Continúa el proceso, pero Streamlit podría no funcionar correctamente")
        
        return STREAMLIT_PATH
    else:
        logger.info(f"El archivo de Streamlit ya existe en {STREAMLIT_PATH}")
        return STREAMLIT_PATH

# Función para iniciar la aplicación Streamlit desde Airflow
def start_streamlit_app(**kwargs):
    """Inicia la aplicación Streamlit en un proceso separado"""
    logger.info("Iniciando aplicación Streamlit para predicciones de películas")
    
    streamlit_path = kwargs.get('ti').xcom_pull(task_ids='setup_streamlit_app')
    if not streamlit_path or not os.path.exists(streamlit_path):
        logger.error(f"No se encontró el archivo de Streamlit: {streamlit_path}")
        return "error_starting_streamlit"
    
    # Retornar la URL de la aplicación
    return f"Aplicación Streamlit disponible en http://localhost:8501"

# Definir el DAG
default_args = {
    'owner': 'gabriela',
    'start_date': days_ago(1),
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False
}

# Función 1: Obtener datos de la API TMDB y enviarlos a Kafka
def fetch_and_send_to_kafka(**kwargs):
    """Obtiene datos de películas de TMDB y los envía a Kafka"""
    import json
    import time
    from kafka import KafkaProducer
    
    # Crear el productor de Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            max_block_ms=30000  # 30 segundos máximo de bloqueo
        )
        logger.info(f"Productor Kafka conectado a {KAFKA_BROKER}")
    except Exception as e:
        logger.error(f"Error al conectar con Kafka: {e}")
        raise
    
    # Obtener películas populares
    url = f"https://api.themoviedb.org/3/movie/popular"
    headers = {
        "Authorization": f"Bearer {TMDB_TOKEN}",
        "Content-Type": "application/json;charset=utf-8"
    }
    params = {
        "language": "es-ES",
        "page": 1
    }
    
    try:
        # Obtener películas populares
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        movies = response.json().get("results", [])
        logger.info(f"Obtenidas {len(movies)} películas populares")
        
        # Procesar cada película
        processed_movies = []
        for movie in movies[:10]:  # Limitamos a 10 películas para evitar sobrecargar la API
            movie_id = movie.get("id")
            
            # Obtener detalles adicionales de la película
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            details_params = {
                "language": "es-ES",
                "append_to_response": "credits"
            }
            
            details_response = requests.get(details_url, headers=headers, params=details_params)
            if details_response.status_code == 200:
                details = details_response.json()
                
                # Enriquecer el objeto de película con detalles adicionales
                movie.update({
                    "budget": details.get("budget"),
                    "revenue": details.get("revenue"),
                    "runtime": details.get("runtime"),
                    "genres": [genre.get("name") for genre in details.get("genres", [])],
                    "production_companies": [company.get("name") for company in details.get("production_companies", [])],
                    "directors": [person.get("name") for person in details.get("credits", {}).get("crew", []) 
                                if person.get("job") == "Director"],
                    "cast": [person.get("name") for person in details.get("credits", {}).get("cast", [])[:5]]  # Primeros 5 actores
                })
                
                processed_movies.append(movie)
                logger.info(f"Procesada película: {movie.get('title')}")
                
                # Enviar a Kafka
                producer.send(KAFKA_TOPIC, value=movie)
                logger.info(f"Enviada a Kafka: {movie.get('title')}")
                
                # Pequeña pausa para no sobrecargar la API
                time.sleep(1)
        
        # Asegurar que todos los mensajes se envíen
        producer.flush()
        logger.info(f"Enviadas {len(processed_movies)} películas a Kafka")
        
        # Guardar también localmente para referencia
        output_file = f"{OUTPUT_DIR}/movies_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_movies, f, indent=4, ensure_ascii=False)
        
        # Pasar información a la siguiente tarea
        kwargs['ti'].xcom_push(key='movie_count', value=len(processed_movies))
        
        return len(processed_movies)
    
    except Exception as e:
        logger.error(f"Error al obtener o enviar datos: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if 'producer' in locals():
            producer.close()

# Función 2: Consumir datos de Kafka y procesarlos
def process_kafka_data(**kwargs):
    """Consume datos de Kafka y los procesa"""
    import json
    import pandas as pd
    from kafka import KafkaConsumer
    import os
    import time
    
    logger.info("Iniciando proceso de consumo de datos de Kafka")
    
    # Crear consumidor de Kafka
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='airflow_consumer_group_v3',
            consumer_timeout_ms=15000  # 15 segundos de timeout
        )
        logger.info(f"Consumidor conectado a Kafka, esperando mensajes en '{KAFKA_TOPIC}'...")
    except Exception as e:
        logger.error(f"Error al conectar con Kafka: {e}")
        logger.error(traceback.format_exc())
        
        # Usar datos de ejemplo en caso de error
        logger.warning("Usando datos de ejemplo debido al error de conexión")
        messages = [{
            "id": 123456,
            "title": "Película de ejemplo",
            "original_title": "Sample Movie",
            "overview": "Esta es una película de ejemplo para probar el DAG.",
            "release_date": "2025-04-01",
            "popularity": 8.5,
            "vote_average": 7.8,
            "vote_count": 1000,
            "budget": 1000000,
            "revenue": 5000000,
            "runtime": 120,
            "genres": ["Acción", "Comedia"],
            "directors": ["Director Ejemplo"],
            "cast": ["Actor 1", "Actor 2"]
        }]
        
        # Procesar con datos de ejemplo
        df = pd.DataFrame(messages)
        output_file = f"{OUTPUT_DIR}/processed_movies_v3.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Datos de ejemplo guardados en {output_file}")
        kwargs['ti'].xcom_push(key='processed_data_path', value=output_file)
        kwargs['ti'].xcom_push(key='movie_count', value=len(df))
        return len(df)
    
    # Procesar mensajes
    messages = []
    start_time = time.time()
    timeout = 30  # 30 segundos
    max_attempts = 3
    attempts = 0
    
    try:
        while time.time() - start_time < timeout and attempts < max_attempts:
            attempts += 1
            logger.info(f"Intento {attempts}/{max_attempts} de obtener mensajes")
            
            msg_pack = consumer.poll(timeout_ms=5000)
            if msg_pack:
                for _, records in msg_pack.items():
                    for record in records:
                        messages.append(record.value)
                        logger.info(f"Recibido mensaje: {record.value.get('title', 'Desconocido')}")
            
            if len(messages) >= 3:  # Si tenemos al menos 3 películas, procesamos
                logger.info(f"Recibidos {len(messages)} mensajes, procediendo a procesar")
                break
            
            logger.info(f"No se recibieron suficientes mensajes, esperando... ({len(messages)} hasta ahora)")
            time.sleep(2)  # Pequeña pausa entre intentos
        
        # Si no se recibieron mensajes, usar datos de ejemplo
        if not messages:
            logger.warning("No se recibieron mensajes de Kafka. Usando datos de ejemplo.")
            messages = [
                {
                    "id": 123456,
                    "title": "Película de ejemplo",
                    "original_title": "Sample Movie",
                    "overview": "Esta es una película de ejemplo para probar el DAG.",
                    "release_date": "2025-04-01",
                    "popularity": 8.5,
                    "vote_average": 7.8,
                    "vote_count": 1000,
                    "budget": 1000000,
                    "revenue": 5000000,
                    "runtime": 120,
                    "genres": ["Acción", "Comedia"],
                    "directors": ["Director Ejemplo"],
                    "cast": ["Actor 1", "Actor 2"]
                }
            ]
        
        # Procesar con Pandas
        logger.info(f"Procesando {len(messages)} mensajes con Pandas")
        
        # Convertir a DataFrame de pandas
        df = pd.DataFrame(messages)
        
        # Limpiar y procesar datos
        df = df.fillna({
            'budget': 0,
            'revenue': 0,
            'runtime': 0,
            'popularity': 0,
            'vote_average': 0,
            'vote_count': 0
        })
        
        # Convertir listas a strings para almacenamiento
        if 'genres' in df.columns and isinstance(df['genres'].iloc[0] if len(df) > 0 else None, list):
            df['genres_str'] = df['genres'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        if 'directors' in df.columns and isinstance(df['directors'].iloc[0] if len(df) > 0 else None, list):
            df['directors_str'] = df['directors'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        if 'cast' in df.columns and isinstance(df['cast'].iloc[0] if len(df) > 0 else None, list):
            df['cast_str'] = df['cast'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        if 'production_companies' in df.columns and isinstance(df['production_companies'].iloc[0] if len(df) > 0 else None, list):
            df['companies_str'] = df['production_companies'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        # Calcular métricas
        df['roi'] = df.apply(
            lambda row: (row['revenue'] - row['budget']) / row['budget'] if row['budget'] > 0 else 0, 
            axis=1
        )
        
        # Categorizar películas
        df['popularity_category'] = 'Baja'
        df.loc[df['popularity'] > 5, 'popularity_category'] = 'Media'
        df.loc[df['popularity'] > 10, 'popularity_category'] = 'Alta'
        df.loc[df['popularity'] > 20, 'popularity_category'] = 'Muy Alta'
        
        df['rating_category'] = 'Mala'
        df.loc[df['vote_average'] >= 4, 'rating_category'] = 'Regular'
        df.loc[df['vote_average'] >= 6, 'rating_category'] = 'Buena'
        df.loc[df['vote_average'] >= 8, 'rating_category'] = 'Excelente'
        
        # Guardar datos procesados
        output_file = f"{OUTPUT_DIR}/processed_movies_v3.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Datos procesados guardados en {output_file}")
        
        # Pasar a la siguiente tarea
        kwargs['ti'].xcom_push(key='processed_data_path', value=output_file)
        kwargs['ti'].xcom_push(key='movie_count', value=len(df))
        
        # Cerrar consumidor
        consumer.close()
        
        return len(df)
    
    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")
        logger.error(traceback.format_exc())
        
        # Asegurar que el consumidor se cierre incluso en caso de error
        if 'consumer' in locals():
            consumer.close()
        raise

# Función 3: Cargar datos en PostgreSQL
def load_to_postgres(**kwargs):
    """Carga los datos transformados en PostgreSQL"""
    import psycopg2
    
    ti = kwargs['ti']
    processed_data_path = ti.xcom_pull(key='processed_data_path', task_ids='process_kafka_data')
    
    if not processed_data_path or not os.path.exists(processed_data_path):
        logger.error(f"No se encontró el archivo de datos procesados: {processed_data_path}")
        # Intentar usar un archivo predeterminado
        processed_data_path = f"{OUTPUT_DIR}/processed_movies_v3.csv"
        if not os.path.exists(processed_data_path):
            logger.error("No hay datos procesados para cargar en PostgreSQL")
            return 0
    
    try:
        # Leer datos procesados
        df = pd.read_csv(processed_data_path)
        logger.info(f"Leyendo {len(df)} registros procesados para cargar en PostgreSQL")
        
        # Asegurar que las tablas existen
        create_postgres_tables()
        
        # Intentar conexión directa a PostgreSQL
        try:
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            
            cur = conn.cursor()
            logger.info("Conexión establecida con PostgreSQL")
            
            # Insertar datos en la tabla movies
            inserted_count = 0
            for _, row in df.iterrows():
                try:
                    # Verificar si la película ya existe
                    cur.execute("SELECT id FROM movies WHERE tmdb_id = %s", (row.get('id'),))
                    if cur.fetchone() is None:
                        # Insertar película
                        insert_query = """
                        INSERT INTO movies (
                            tmdb_id, title, original_title, overview, release_date,
                            popularity, vote_average, vote_count, budget, revenue,
                            runtime, adult, genres_str, directors_str, cast_str,
                            companies_str, roi, popularity_category, rating_category
                        ) ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        adult_value = row.get('adult', False)
                        if not isinstance(adult_value, bool):
                            adult_value = False
                        
                        cur.execute(insert_query, (
                            row.get('id'), 
                            row.get('title'), 
                            row.get('original_title'), 
                            row.get('overview'), 
                            row.get('release_date'),
                            row.get('popularity'), 
                            row.get('vote_average'), 
                            row.get('vote_count'), 
                            row.get('budget', 0), 
                            row.get('revenue', 0),
                            row.get('runtime', 0), 
                            adult_value, 
                            row.get('genres_str', ''), 
                            row.get('directors_str', ''), 
                            row.get('cast_str', ''),
                            row.get('companies_str', ''), 
                            row.get('roi', 0), 
                            row.get('popularity_category', 'Desconocida'), 
                            row.get('rating_category', 'Desconocida')
                        ))
                        
                        conn.commit()
                        inserted_count += 1
                        logger.info(f"Película insertada en movies: {row.get('title')}")
                        
                        # Insertar géneros individuales
                        if 'genres_str' in row and row['genres_str']:
                            genres = row['genres_str'].split(',')
                            for genre in genres:
                                if genre.strip():
                                    # Añadir género si no existe
                                    cur.execute("SELECT id FROM genres WHERE name = %s", (genre.strip(),))
                                    genre_id = cur.fetchone()
                                    if not genre_id:
                                        cur.execute("INSERT INTO genres (name) VALUES (%s) RETURNING id", (genre.strip(),))
                                        genre_id = cur.fetchone()[0]
                                        conn.commit()
                        
                        # Insertar directores individuales
                        if 'directors_str' in row and row['directors_str']:
                            directors = row['directors_str'].split(',')
                            for director in directors:
                                if director.strip():
                                    # Añadir director si no existe
                                    cur.execute("SELECT id FROM directors WHERE name = %s", (director.strip(),))
                                    director_id = cur.fetchone()
                                    if not director_id:
                                        cur.execute("INSERT INTO directors (name) VALUES (%s) RETURNING id", (director.strip(),))
                                        director_id = cur.fetchone()[0]
                                        conn.commit()
                        
                        # Insertar actores individuales
                        if 'cast_str' in row and row['cast_str']:
                            actors = row['cast_str'].split(',')
                            for actor in actors:
                                if actor.strip():
                                    # Añadir actor si no existe
                                    cur.execute("SELECT id FROM actors WHERE name = %s", (actor.strip(),))
                                    actor_id = cur.fetchone()
                                    if not actor_id:
                                        cur.execute("INSERT INTO actors (name) VALUES (%s) RETURNING id", (actor.strip(),))
                                        actor_id = cur.fetchone()[0]
                                        conn.commit()
                        
                        # Insertar en data warehouse
                        # Extraer año de lanzamiento
                        release_year = None
                        release_date = row.get('release_date', '')
                        if release_date and len(str(release_date)) >= 4:
                            try:
                                release_year = int(str(release_date)[:4])
                            except:
                                pass
                        
                        # Para cada género, crear una entrada en el warehouse
                        genres = row.get('genres_str', '').split(',')
                        if not genres or genres[0] == '':
                            genres = ['Unknown']
                        
                        directors = row.get('directors_str', '').split(',')
                        if not directors or directors[0] == '':
                            directors = ['Unknown']
                        
                        for genre in genres:
                            if not genre.strip():
                                continue
                            
                            for director in directors:
                                if not director.strip():
                                    continue
                                
                                # Insertar en warehouse
                                warehouse_query = """
                                INSERT INTO movie_data_warehouse (
                                    tmdb_id, title, release_date, release_year, genre,
                                    budget, revenue, runtime, popularity,
                                    vote_average, vote_count, roi,
                                    director, popularity_level, rating_level, is_profitable
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """
                                
                                cur.execute(warehouse_query, (
                                    row.get('id'),
                                    row.get('title'),
                                    release_date,
                                    release_year,
                                    genre.strip(),
                                    row.get('budget', 0),
                                    row.get('revenue', 0),
                                    row.get('runtime', 0),
                                    row.get('popularity', 0),
                                    row.get('vote_average', 0),
                                    row.get('vote_count', 0),
                                    row.get('roi', 0),
                                    director.strip(),
                                    row.get('popularity_category', 'Unknown'),
                                    row.get('rating_category', 'Unknown'),
                                    row.get('revenue', 0) > row.get('budget', 0)
                                ))
                                
                                conn.commit()
                except Exception as e:
                    logger.error(f"Error al insertar película {row.get('title')}: {e}")
                    conn.rollback()
            
            # Verificar cuántas películas hay en la base de datos
            cur.execute("SELECT COUNT(*) FROM movies")
            movie_count = cur.fetchone()[0]
            logger.info(f"Total de películas en la base de datos: {movie_count}")
            logger.info(f"Insertadas {inserted_count} nuevas películas en esta ejecución")
            
            # Cerrar conexión
            cur.close()
            conn.close()
            
            logger.info("Datos cargados correctamente en PostgreSQL")
            return len(df)
        
        except Exception as e:
            logger.error(f"Error al conectar o insertar en PostgreSQL: {e}")
            logger.error(traceback.format_exc())
            
            # Usamos un método alternativo si falla la conexión directa
            logger.info("Intentando método alternativo: guardar resultados como CSV")
            
            # Guardar resultados como CSV para análisis posterior
            results_file = f"{OUTPUT_DIR}/results_data_v3.csv"
            df.to_csv(results_file, index=False)
            logger.info(f"Resultados guardados en {results_file}")
            
            return 0
    
    except Exception as e:
        logger.error(f"Error en la carga de datos a PostgreSQL: {e}")
        logger.error(traceback.format_exc())
        raise

# Función 4: Generar visualizaciones
def generate_visualizations(**kwargs):
    """Genera visualizaciones a partir de los datos almacenados"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Intentar leer datos de PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            
            # Consultas analíticas
            queries = {
                "top_movies": """
                    SELECT title, popularity, vote_average, vote_count
                    FROM movies
                    ORDER BY popularity DESC
                    LIMIT 10;
                """,
                "genre_distribution": """
                    SELECT genre, COUNT(*) as count
                    FROM movie_data_warehouse
                    GROUP BY genre
                    ORDER BY count DESC
                    LIMIT 10;
                """
            }
            
            results = {}
            for name, query in queries.items():
                results[name] = pd.read_sql(query, conn)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error al leer datos de PostgreSQL: {e}")
            logger.error(traceback.format_exc())
            
            # Si falla, usar los datos procesados del CSV
            processed_data_path = kwargs['ti'].xcom_pull(key='processed_data_path', task_ids='process_kafka_data')
            if not processed_data_path or not os.path.exists(processed_data_path):
                processed_data_path = f"{OUTPUT_DIR}/processed_movies_v3.csv"
            
            if not os.path.exists(processed_data_path):
                logger.error("No se encontraron datos para visualizar")
                return None
            
            # Cargar datos desde CSV
            df = pd.read_csv(processed_data_path)
            
            # Crear resultados basados en CSV
            results = {
                "top_movies": df[['title', 'popularity', 'vote_average', 'vote_count']].sort_values('popularity', ascending=False).head(10),
                "genre_distribution": pd.DataFrame([])
            }
            
            # Si tenemos géneros, analizarlos
            if 'genres_str' in df.columns:
                # Extraer géneros
                all_genres = []
                for genres in df['genres_str']:
                    if isinstance(genres, str) and genres:
                        all_genres.extend([g.strip() for g in genres.split(',')])
                
                # Contar géneros
                genre_counts = {}
                for genre in all_genres:
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
                # Crear DataFrame
                results["genre_distribution"] = pd.DataFrame({
                    'genre': list(genre_counts.keys()),
                    'count': list(genre_counts.values())
                }).sort_values('count', ascending=False)
        
        # Generar visualizaciones
        visualizations_dir = f"{OUTPUT_DIR}/visualizations_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(visualizations_dir, exist_ok=True)
        
        for name, df in results.items():
            if df.empty:
                continue
                
            plt.figure(figsize=(12, 8))
            
            if name == "top_movies":
                sns.barplot(x='popularity', y='title', data=df)
                plt.title('Top 10 Películas por Popularidad')
            
            elif name == "genre_distribution":
                sns.barplot(x='count', y='genre', data=df)
                plt.title('Distribución de Géneros')
            
            plt.tight_layout()
            plt.savefig(f"{visualizations_dir}/{name}.png")
            plt.close()
        
        # Generar un reporte de texto
        report_path = f"{visualizations_dir}/report.txt"
        with open(report_path, 'w') as f:
            f.write("ANÁLISIS DE PELÍCULAS (Versión 3)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Informe generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Visualizaciones generadas:\n")
            for name in results.keys():
                if not results[name].empty:
                    f.write(f"- {name}.png\n")
        
        logger.info(f"Visualizaciones guardadas en {visualizations_dir}")
        
        # Pasar información a la siguiente tarea
        kwargs['ti'].xcom_push(key='visualizations_dir', value=visualizations_dir)
        
        return visualizations_dir
    
    except Exception as e:
        logger.error(f"Error al generar visualizaciones: {e}")
        logger.error(traceback.format_exc())
        raise

# Función 5: Entrenar modelo ML
def train_ml_model(**kwargs):
    """Entrena un modelo de Machine Learning básico"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Intentar leer datos de PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            
            # Consultar datos para el entrenamiento
            query = """
                SELECT popularity, vote_average, vote_count, budget, revenue, runtime
                FROM movies
                WHERE vote_count > 0
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
        except Exception as e:
            logger.error(f"Error al leer datos de PostgreSQL para ML: {e}")
            logger.error(traceback.format_exc())
            
            # Si falla, usar los datos procesados del CSV
            processed_data_path = kwargs['ti'].xcom_pull(key='processed_data_path', task_ids='process_kafka_data')
            if not processed_data_path or not os.path.exists(processed_data_path):
                processed_data_path = f"{OUTPUT_DIR}/processed_movies_v3.csv"
            
            if not os.path.exists(processed_data_path):
                logger.error("No se encontraron datos para entrenar modelo ML")
                return None
            
            # Cargar datos desde CSV
            df = pd.read_csv(processed_data_path)
        
        if len(df) < 5:
            logger.warning("No hay suficientes datos para entrenar un modelo ML")
            return None
        
        # Preparar datos
        df = df.fillna(0)
        
        # Target: popularidad
        df['target'] = df['popularity']
        
        # Features
        features = ['vote_average', 'vote_count', 'runtime']
        if 'budget' in df.columns:
            features.append('budget')
        
        X = df[features]
        y = df['target']
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardar métricas
        metrics = {
            'mse': float(mse),
            'r2': float(r2),
            'feature_importance': dict(zip(features, model.coef_))
        }
        
        # Guardar modelo y resultados
        model_dir = f"{OUTPUT_DIR}/ml_model_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f"{model_dir}/model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        with open(f"{model_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Visualizar importancia de características
        plt.figure(figsize=(10, 6))
        feat_importances = pd.DataFrame({
            'feature': features,
            'importance': model.coef_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feat_importances)
        plt.title('Importancia de Características')
        plt.tight_layout()
        plt.savefig(f"{model_dir}/feature_importance.png")
        plt.close()
        
        logger.info(f"Modelo ML entrenado y guardado en {model_dir}")
        logger.info(f"Métricas del modelo - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Pasar información a la siguiente tarea
        kwargs['ti'].xcom_push(key='model_dir', value=model_dir)
        kwargs['ti'].xcom_push(key='model_metrics', value=metrics)
        
        return model_dir
    
    except Exception as e:
        logger.error(f"Error al entrenar modelo ML: {e}")
        logger.error(traceback.format_exc())
        raise

with DAG(
    'tmdb_pipeline_streamlit',  # Nombre actualizado del DAG
    default_args=default_args,
    description='Pipeline de datos TMDB con interfaz Streamlit',
    schedule_interval=timedelta(hours=12),
    catchup=False
) as dag:
    
    # Tarea 0: Preparar entorno
    setup_task = PythonOperator(
        task_id='setup_environment',
        python_callable=setup_environment,
        provide_context=True
    )
    
    # Tarea 1: Iniciar el pipeline
    start_task = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Iniciando pipeline de datos de TMDB con Streamlit" && mkdir -p {{ params.output_dir }}',
        params={'output_dir': OUTPUT_DIR}
    )
    
    # Tarea 2: Obtener datos de TMDB y enviarlos a Kafka
    fetch_task = PythonOperator(
        task_id='fetch_and_send_to_kafka',
        python_callable=fetch_and_send_to_kafka,
        provide_context=True
    )
    
    # Tarea 3: Consumir y procesar datos de Kafka
    process_task = PythonOperator(
        task_id='process_kafka_data',
        python_callable=process_kafka_data,
        provide_context=True
    )
    
    # Tarea 4: Cargar datos en PostgreSQL
    load_task = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres,
        provide_context=True
    )
    
    # Tarea 5: Generar visualizaciones
    visualization_task = PythonOperator(
        task_id='generate_visualizations',
        python_callable=generate_visualizations,
        provide_context=True
    )
    
    # Tarea 6: Entrenar modelo ML
    ml_task = PythonOperator(
        task_id='train_ml_model',
        python_callable=train_ml_model,
        provide_context=True
    )
    
    # Tarea 7: Preparar la aplicación Streamlit
    setup_streamlit_task = PythonOperator(
        task_id='setup_streamlit_app',
        python_callable=setup_streamlit_app,
        provide_context=True
    )
    
    # Tarea 8: Instalar dependencias de Streamlit
    install_deps_task = BashOperator(
        task_id='install_streamlit_deps',
        bash_command='pip install streamlit plotly matplotlib seaborn psycopg2-binary scikit-learn kafka-python',
    )
    
    # Tarea 9: Iniciar la aplicación Streamlit
    streamlit_task = StreamlitOperator(
        task_id='start_streamlit_app',
        script_path=STREAMLIT_PATH,
        port=8502,  # Cambiado a 8502 para evitar conflictos con AirPlay en Mac
        host="0.0.0.0"
    )
    
    # Tarea 10: Finalizar el pipeline
    end_task = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "Pipeline de datos de TMDB con Streamlit completado con éxito a las $(date)"'
    )
    
    # At the end of the DAG definition, update the task dependencies
    store_model_task = PythonOperator(
    task_id='store_ml_model',
    python_callable=store_ml_model,
    provide_context=True
)

    # Definir el flujo de tareas
    setup_task >> start_task >> fetch_task >> process_task >> load_task >> visualization_task >> ml_task >> store_model_task >> setup_streamlit_task >> install_deps_task >> streamlit_task >> end_task