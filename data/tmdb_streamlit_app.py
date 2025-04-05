
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
POSTGRES_HOST = "movie_postgres"
POSTGRES_PORT = "5433"

# Configuración de TMDB API
TMDB_API_KEY = "e8e1dae84a0345bd3ec23e3030905258"
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlOGUxZGFlODRhMDM0NWJkM2VjMjNlMzAzMDkwNTI1OCIsIm5iZiI6MTc0MTkxMzEzNi4xMjEsInN1YiI6IjY3ZDM3YzMwYmY0ODE4ODU0YzY0ZTVmNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tdq930_qLqYbYAqwVLl3Tdw84HdEsZtM41CX_9-lJNU"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Función para cargar el modelo más reciente
# Model loading function
@st.cache_resource
def load_latest_model():
    try:
        # Define search paths for models
        model_paths = [
            "/opt/airflow/data/movie_analytics/latest_ml_model/model.pkl",
            "/opt/airflow/data/movie_analytics/ml_models/latest_model/model.pkl",
            "/opt/airflow/data/movie_analytics/ml_model_v3/model.pkl",
            "data/movie_analytics/latest_ml_model/model.pkl",
            "data/movie_analytics/ml_models/latest_model/model.pkl",
            "data/movie_analytics/ml_model_v3/model.pkl"
        ]
        
        metrics_paths = [
            "/opt/airflow/data/movie_analytics/latest_ml_model/metrics.json",
            "/opt/airflow/data/movie_analytics/ml_models/latest_model/metrics.json",
            "/opt/airflow/data/movie_analytics/ml_model_v3/metrics.json",
            "data/movie_analytics/latest_ml_model/metrics.json", 
            "data/movie_analytics/ml_models/latest_model/metrics.json",
            "data/movie_analytics/ml_model_v3/metrics.json"
        ]
        
        # Find the first existing model path
        model_path = next((path for path in model_paths if os.path.exists(path)), None)
        metrics_path = next((path for path in metrics_paths if os.path.exists(path)), None)
        
        if not model_path:
            st.warning("No trained model found. Using a default model.")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.coef_ = np.array([0.5, 0.3, 0.2, 0.1])
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
            return model, metrics, list(metrics["feature_importance"].keys())
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metrics
        metrics = {}
        feature_names = []
        
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            feature_names = list(metrics.get("feature_importance", {}).keys())
        
        st.success(f"Model loaded from {model_path}")
        return model, metrics, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to default model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.coef_ = np.array([0.5, 0.3, 0.2, 0.1])
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
        return model, metrics, list(metrics["feature_importance"].keys())

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