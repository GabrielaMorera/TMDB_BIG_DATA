import json
import time
import os
import logging
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
from kafka import KafkaConsumer
from dask.distributed import Client
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Table, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import warnings
import psycopg2
from datetime import datetime
import traceback
import glob
import pickle
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
KAFKA_TOPIC = "tmdb_data"
KAFKA_BROKER = "localhost:9092"  # Usa localhost si estás ejecutando fuera de Docker

# Configuración de PostgreSQL
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Villeta-11"  # Agregada la contraseña correcta
POSTGRES_HOST = "movie_postgres"
POSTGRES_PORT = "5432"
OUTPUT_DIR = "movie_analytics"

# Crear directorio para las visualizaciones
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir la base SQLAlchemy
Base = declarative_base()

# Definir modelos SQLAlchemy para las tablas
class Movie(Base):
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, unique=True)
    title = Column(String)
    original_title = Column(String)
    overview = Column(String)
    release_date = Column(String)
    popularity = Column(Float)
    vote_average = Column(Float)
    vote_count = Column(Integer)
    budget = Column(Integer)
    revenue = Column(Integer)
    runtime = Column(Integer)
    adult = Column(Boolean)
    poster_path = Column(String)
    backdrop_path = Column(String)
    
    # Relaciones
    genres = relationship("Genre", secondary="movie_genres", back_populates="movies")
    production_companies = relationship("ProductionCompany", secondary="movie_production_companies", back_populates="movies")
    directors = relationship("Director", secondary="movie_directors", back_populates="movies")
    cast_members = relationship("Actor", secondary="movie_actors", back_populates="movies")

class Genre(Base):
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    
    # Relaciones
    movies = relationship("Movie", secondary="movie_genres", back_populates="genres")

class ProductionCompany(Base):
    __tablename__ = 'production_companies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    
    # Relaciones
    movies = relationship("Movie", secondary="movie_production_companies", back_populates="production_companies")

class Director(Base):
    __tablename__ = 'directors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    
    # Relaciones
    movies = relationship("Movie", secondary="movie_directors", back_populates="directors")

class Actor(Base):
    __tablename__ = 'actors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    
    # Relaciones
    movies = relationship("Movie", secondary="movie_actors", back_populates="cast_members")

# Tablas de relación (muchos a muchos)
movie_genres = Table('movie_genres', Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.id'), primary_key=True)
)

movie_production_companies = Table('movie_production_companies', Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('company_id', Integer, ForeignKey('production_companies.id'), primary_key=True)
)

movie_directors = Table('movie_directors', Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('director_id', Integer, ForeignKey('directors.id'), primary_key=True)
)

movie_actors = Table('movie_actors', Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('actor_id', Integer, ForeignKey('actors.id'), primary_key=True)
)

# Tabla para data warehouse
class MovieDataWarehouse(Base):
    __tablename__ = 'movie_data_warehouse'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer)
    title = Column(String)
    release_date = Column(String)
    release_year = Column(Integer)
    genre = Column(String)
    budget = Column(Integer)
    revenue = Column(Integer)
    runtime = Column(Integer)
    popularity = Column(Float)
    vote_average = Column(Float)
    vote_count = Column(Integer)
    roi = Column(Float)
    director = Column(String)
    popularity_level = Column(String)
    rating_level = Column(String)
    is_profitable = Column(Boolean)
    data_date = Column(DateTime, default=datetime.now)

class MovieAnalyzer:
    def __init__(self):
        logger.info("Inicializando MovieAnalyzer...")
        
        # Crear conexión a Kafka
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='movie_analysis_group',
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )

        # Cargar el modelo más reciente
        logger.info("Buscando el modelo más reciente...")
        model_dir = self.get_most_recent_model_dir()
        if model_dir:
            model_path = os.path.join(model_dir, "model.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Modelo cargado desde: {model_path}")
            else:
                logger.warning("No se encontró el archivo 'model.pkl' en el directorio más reciente.")
                self.model = None
        else:
            logger.warning("No se encontró ningún directorio de modelo.")
            self.model = None
        
        # Probar conexión a PostgreSQL antes de continuar
        self.test_postgres_connection()
        
        # Crear conexión a PostgreSQL con SQLAlchemy
        try:
            # Primero creamos el esquema y tablas sin SQLAlchemy para asegurar que existan
            self.create_schema_and_tables_raw()
            
            # Luego usamos SQLAlchemy
            self.engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
            
            # Crear tablas en PostgreSQL si no existen
            Base.metadata.create_all(self.engine)
            logger.info("Tablas creadas/verificadas en PostgreSQL usando SQLAlchemy")
            
            # Crear sesión SQLAlchemy
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info("Iniciando cliente Dask...")
            self.client = Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')
            logger.info(f"Dashboard Dask: {self.client.dashboard_link}")
            
            logger.info("Inicialización completada")
        except Exception as e:
            logger.error(f"Error al inicializar conexiones: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_most_recent_model_dir(self):
        search_patterns = [
            'data/movie_analytics/ml_model_v3_*',
            'data/movie_analytics/ml_model_v2_*',
            'data/movie_analytics/ml_model_*',
            'data/movie_analytics/xgb_model_*',
            'data/movie_analytics/gb_model_*',
            'data/movie_analytics/ensemble_model_*',
        ]
        all_model_paths = []
        for pattern in search_patterns:
            all_model_paths.extend(glob.glob(pattern))

        if not all_model_paths:
            return None

        all_model_paths.sort(key=os.path.getmtime, reverse=True)
        return all_model_paths[0]

    def create_schema_and_tables_raw(self):
        """Crea el esquema y tablas usando psycopg2 directamente, que puede ser más confiable."""
        try:
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            conn.autocommit = False
            cur = conn.cursor()
            
            # Crear esquema public
            cur.execute("CREATE SCHEMA IF NOT EXISTS public;")
            
            # Tabla movies
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
                poster_path VARCHAR(255),
                backdrop_path VARCHAR(255)
            );
            """)
            
            # Tabla genres
            cur.execute("""
            CREATE TABLE IF NOT EXISTS genres (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE
            );
            """)
            
            # Tabla production_companies
            cur.execute("""
            CREATE TABLE IF NOT EXISTS production_companies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE
            );
            """)
            
            # Tabla directors
            cur.execute("""
            CREATE TABLE IF NOT EXISTS directors (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE
            );
            """)
            
            # Tabla actors
            cur.execute("""
            CREATE TABLE IF NOT EXISTS actors (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE
            );
            """)
            
            # Tabla de relación movie_genres
            cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_genres (
                movie_id INTEGER REFERENCES movies(id),
                genre_id INTEGER REFERENCES genres(id),
                PRIMARY KEY (movie_id, genre_id)
            );
            """)
            
            # Tabla de relación movie_production_companies
            cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_production_companies (
                movie_id INTEGER REFERENCES movies(id),
                company_id INTEGER REFERENCES production_companies(id),
                PRIMARY KEY (movie_id, company_id)
            );
            """)
            
            # Tabla de relación movie_directors
            cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_directors (
                movie_id INTEGER REFERENCES movies(id),
                director_id INTEGER REFERENCES directors(id),
                PRIMARY KEY (movie_id, director_id)
            );
            """)
            
            # Tabla de relación movie_actors
            cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_actors (
                movie_id INTEGER REFERENCES movies(id),
                actor_id INTEGER REFERENCES actors(id),
                PRIMARY KEY (movie_id, actor_id)
            );
            """)
            
            # Tabla para data warehouse
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
                director VARCHAR(255),
                popularity_level VARCHAR(50),
                rating_level VARCHAR(50),
                is_profitable BOOLEAN,
                data_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            conn.commit()
            logger.info("Tablas creadas directamente con psycopg2")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error al crear tablas directamente: {e}")
            logger.error(traceback.format_exc())
            if 'conn' in locals():
                conn.rollback()
                conn.close()
    
    def test_postgres_connection(self):
        """Prueba la conexión a PostgreSQL e imprime las tablas existentes."""
        try:
            # Conexión directa con psycopg2
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            
            # Crear un cursor
            cur = conn.cursor()
            
            # Verificar la versión de PostgreSQL
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
            logger.info(f"Conectado a PostgreSQL: {db_version[0]}")
            
            # Crear un esquema público si no existe
            cur.execute("CREATE SCHEMA IF NOT EXISTS public;")
            conn.commit()
            
            # Listar las tablas existentes
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = cur.fetchall()
            logger.info("Tablas existentes:")
            if tables:
                for table in tables:
                    logger.info(f"- {table[0]}")
            else:
                logger.info("No hay tablas en la base de datos.")
            
            # Cerrar la conexión
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error al probar la conexión a PostgreSQL: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def process_batch(self, batch):
        """Procesa un lote de datos con Dask."""
        logger.info(f"Procesando lote de {len(batch)} películas")
        
        # Convertir a DataFrame de pandas
        df = pd.DataFrame(batch)
        
        # Convertir a Dask DataFrame para procesamiento paralelo
        ddf = dd.from_pandas(df, npartitions=4)
        
        # Manejar valores nulos
        ddf = ddf.map_partitions(self.clean_dataframe)
        
        # Calcular métricas adicionales
        ddf = ddf.map_partitions(self.calculate_metrics)
        
        # Calcular el dataframe final
        return ddf.compute()
    
    def clean_dataframe(self, df):
        """Limpia y estandariza el DataFrame."""
        # Llenar valores nulos
        df = df.fillna({
            'budget': 0,
            'revenue': 0,
            'runtime': 0,
            'popularity': 0,
            'vote_average': 0,
            'vote_count': 0,
            'genres': [],
            'production_companies': [],
            'directors': [],
            'cast': []
        })
        
        # Asegurar que ciertas columnas estén presentes
        for col in ['adult', 'backdrop_path', 'poster_path']:
            if col not in df.columns:
                df[col] = None
        
        # Limpiar y estandarizar fechas de lanzamiento
        if 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
        
        # Convertir valores numéricos
        numerical_cols = ['budget', 'revenue', 'popularity', 'vote_count', 'vote_average', 'runtime']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def calculate_metrics(self, df):
        """Calcula métricas adicionales para análisis."""
        # Calcular retorno sobre inversión (ROI)
        df['roi'] = df.apply(
            lambda row: (row['revenue'] - row['budget']) / row['budget'] if row['budget'] > 0 else 0, 
            axis=1
        )
        
        # Calcular ratio ingresos/presupuesto
        df['revenue_budget_ratio'] = df.apply(
            lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0,
            axis=1
        )
        
        # Calcular score de impacto (combinación de popularidad y valoración)
        df['impact_score'] = (df['popularity'] * 0.6) + (df['vote_average'] * 10 * 0.4)
        
        # Calcular score comercial (combinación de ingresos y ROI)
        df['commercial_score'] = df.apply(
            lambda row: (np.log1p(row['revenue']) * 0.7) + (min(row['roi'], 10) * 30 * 0.3),
            axis=1
        )
        
        # Categorizar películas por popularidad usando cuartiles
        pop_quartiles = [0, 5, 10, 20, float('inf')]
        pop_labels = ['Muy Baja', 'Baja', 'Media', 'Alta']
        df['popularity_category'] = pd.cut(df['popularity'], bins=pop_quartiles, labels=pop_labels, right=False)
        
        # Categorizar por calificación
        rating_quartiles = [0, 4, 6, 8, 10]
        rating_labels = ['Mala', 'Regular', 'Buena', 'Excelente']
        df['rating_category'] = pd.cut(df['vote_average'], bins=rating_quartiles, labels=rating_labels, right=False)
        
        # Convertir categorías a string para compatibilidad con la base de datos
        df['popularity_category'] = df['popularity_category'].astype(str)
        df['rating_category'] = df['rating_category'].astype(str)
        
        # Calcular índice de rentabilidad
        df['is_profitable'] = df['revenue'] > df['budget']
        
        return df
    
    def save_to_database(self, df):
        """Guarda los datos procesados en PostgreSQL."""
        processed_count = 0
        
        for _, movie_data in df.iterrows():
            try:
                # Verificar si la película ya existe
                tmdb_id = movie_data.get('id')
                existing_movie = self.session.query(Movie).filter_by(tmdb_id=tmdb_id).first()
                
                if existing_movie:
                    logger.info(f"Película ya existe: {movie_data.get('title')}")
                    continue
                
                # Crear nueva película
                movie = Movie(
                    tmdb_id=tmdb_id,
                    title=movie_data.get('title'),
                    original_title=movie_data.get('original_title'),
                    overview=movie_data.get('overview'),
                    release_date=movie_data.get('release_date'),
                    popularity=movie_data.get('popularity'),
                    vote_average=movie_data.get('vote_average'),
                    vote_count=movie_data.get('vote_count'),
                    budget=movie_data.get('budget', 0),
                    revenue=movie_data.get('revenue', 0),
                    runtime=movie_data.get('runtime', 0),
                    adult=movie_data.get('adult', False),
                    poster_path=movie_data.get('poster_path'),
                    backdrop_path=movie_data.get('backdrop_path')
                )
                
                # Añadir géneros
                for genre_name in movie_data.get('genres', []):
                    genre = self.session.query(Genre).filter_by(name=genre_name).first()
                    if not genre:
                        genre = Genre(name=genre_name)
                        self.session.add(genre)
                    movie.genres.append(genre)
                
                # Añadir compañías productoras
                for company_name in movie_data.get('production_companies', []):
                    company = self.session.query(ProductionCompany).filter_by(name=company_name).first()
                    if not company:
                        company = ProductionCompany(name=company_name)
                        self.session.add(company)
                    movie.production_companies.append(company)
                
                # Añadir directores
                for director_name in movie_data.get('directors', []):
                    director = self.session.query(Director).filter_by(name=director_name).first()
                    if not director:
                        director = Director(name=director_name)
                        self.session.add(director)
                    movie.directors.append(director)
                
                # Añadir actores
                for actor_name in movie_data.get('cast', []):
                    actor = self.session.query(Actor).filter_by(name=actor_name).first()
                    if not actor:
                        actor = Actor(name=actor_name)
                        self.session.add(actor)
                    movie.cast_members.append(actor)
                
                # Guardar película
                self.session.add(movie)
                self.session.commit()
                
                # Añadir entradas al data warehouse
                self.populate_data_warehouse(movie_data)
                
                processed_count += 1
                logger.info(f"Película guardada: {movie_data.get('title')}")
                
            except Exception as e:
                self.session.rollback()
                logger.error(f"Error al guardar película {movie_data.get('title')}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Procesadas {processed_count} películas nuevas")
        return processed_count
    
    def populate_data_warehouse(self, movie_data):
        """Añade datos al data warehouse."""
        try:
            # Extraer año de lanzamiento
            release_year = None
            if movie_data.get('release_date') and len(str(movie_data.get('release_date'))) >= 4:
                try:
                    release_year = int(str(movie_data.get('release_date'))[:4])
                except:
                    pass
            
            # Para cada género, crear una entrada en el warehouse
            genres = movie_data.get('genres', ['Unknown'])
            directors = movie_data.get('directors', ['Unknown'])
            
            for genre in genres:
                if not genre:
                    continue
                
                for director in directors:
                    if not director:
                        continue
                    
                    # Crear entrada en el data warehouse
                    warehouse_entry = MovieDataWarehouse(
                        tmdb_id=movie_data.get('id'),
                        title=movie_data.get('title'),
                        release_date=movie_data.get('release_date'),
                        release_year=release_year,
                        genre=genre,
                        budget=movie_data.get('budget', 0),
                        revenue=movie_data.get('revenue', 0),
                        runtime=movie_data.get('runtime', 0),
                        popularity=movie_data.get('popularity', 0),
                        vote_average=movie_data.get('vote_average', 0),
                        vote_count=movie_data.get('vote_count', 0),
                        roi=movie_data.get('roi', 0),
                        director=director,
                        popularity_level=movie_data.get('popularity_category', 'Unknown'),
                        rating_level=movie_data.get('rating_category', 'Unknown'),
                        is_profitable=movie_data.get('revenue', 0) > movie_data.get('budget', 0)
                    )
                    
                    self.session.add(warehouse_entry)
            
            self.session.commit()
            logger.info(f"Datos agregados al warehouse para: {movie_data.get('title')}")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error al agregar datos al warehouse para {movie_data.get('title')}: {e}")
            logger.error(traceback.format_exc())
    
    def analyze_data(self):
        """Realiza análisis estadístico de los datos."""
        logger.info("Realizando análisis de datos...")
        
        # Consultas analíticas
        queries = {
            "Top 10 películas por popularidad": """
                SELECT title, popularity, vote_average, vote_count
                FROM movies
                ORDER BY popularity DESC
                LIMIT 10;
            """,
            "Películas con mejor ROI": """
                SELECT m.title, m.revenue, m.budget, 
                       CASE WHEN m.budget > 0 THEN m.revenue::float / m.budget ELSE 0 END AS roi
                FROM movies m
                WHERE m.budget > 0
                ORDER BY roi DESC
                LIMIT 10;
            """,
            "Géneros más populares": """
                SELECT g.name, COUNT(*) as movie_count, AVG(m.vote_average) as avg_rating
                FROM genres g
                JOIN movie_genres mg ON g.id = mg.genre_id
                JOIN movies m ON mg.movie_id = m.id
                GROUP BY g.name
                ORDER BY movie_count DESC;
            """,
            "Directores con más películas": """
                SELECT d.name, COUNT(*) as movie_count, AVG(m.vote_average) as avg_rating
                FROM directors d
                JOIN movie_directors md ON d.id = md.director_id
                JOIN movies m ON md.movie_id = m.id
                GROUP BY d.name
                ORDER BY movie_count DESC
                LIMIT 10;
            """,
            "Análisis de data warehouse": """
                SELECT genre, COUNT(*) as count, AVG(popularity) as avg_popularity,
                       AVG(vote_average) as avg_rating, SUM(CASE WHEN is_profitable THEN 1 ELSE 0 END) as profitable_count
                FROM movie_data_warehouse
                GROUP BY genre
                ORDER BY count DESC;
            """,
            "Correlación presupuesto-ingresos": """
                SELECT title, budget/1000000 as budget_millions, revenue/1000000 as revenue_millions, 
                       popularity, vote_average
                FROM movies
                WHERE budget > 0 AND revenue > 0
                ORDER BY budget DESC
                LIMIT 30;
            """,
            "Distribución por año": """
                SELECT 
                    EXTRACT(YEAR FROM TO_DATE(release_date, 'YYYY-MM-DD')) as year,
                    COUNT(*) as movie_count,
                    AVG(popularity) as avg_popularity,
                    AVG(vote_average) as avg_rating
                FROM movies
                WHERE release_date IS NOT NULL AND release_date != ''
                GROUP BY year
                ORDER BY year DESC;
            """,
            "Películas de mayor impacto": """
                SELECT title, (popularity * 0.6 + vote_average * 10 * 0.4) as impact_score,
                       popularity, vote_average, release_date
                FROM movies
                ORDER BY impact_score DESC
                LIMIT 15;
            """
        }
        
        results = {}
        with self.engine.connect() as conn:
            for name, query in queries.items():
                try:
                    result = pd.read_sql(query, conn)
                    results[name] = result
                    logger.info(f"Consulta '{name}' completada: {len(result)} filas")
                except Exception as e:
                    logger.error(f"Error en consulta '{name}': {e}")
        
        # Consulta adicional para obtener todos los datos para análisis avanzado
        try:
            all_movies_query = """
                SELECT m.tmdb_id, m.title, m.release_date, m.popularity, m.vote_average, 
                       m.vote_count, m.budget, m.revenue, m.runtime, 
                       string_agg(DISTINCT g.name, ',') as genres
                FROM movies m
                LEFT JOIN movie_genres mg ON m.id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                GROUP BY m.tmdb_id, m.title, m.release_date, m.popularity, m.vote_average, 
                         m.vote_count, m.budget, m.revenue, m.runtime
            """
            with self.engine.connect() as conn:
                all_movies = pd.read_sql(all_movies_query, conn)
                results["all_movies"] = all_movies
                logger.info(f"Datos completos obtenidos: {len(all_movies)} películas")
        except Exception as e:
            logger.error(f"Error al obtener datos completos: {e}")
        
        return results
    
    def generate_visualizations(self, results, batch_number):
        """Genera visualizaciones avanzadas a partir de los resultados del análisis."""
        logger.info("Generando visualizaciones...")
        output_path = f"{OUTPUT_DIR}/batch_{batch_number}"
        os.makedirs(output_path, exist_ok=True)
        
        # Visualización 1: Top 10 películas por popularidad
        if "Top 10 películas por popularidad" in results and not results["Top 10 películas por popularidad"].empty:
            plt.figure(figsize=(14, 8))
            df = results["Top 10 películas por popularidad"]
            
            # Crear paleta de colores basada en la calificación
            colors = plt.cm.viridis(df['vote_average'] / 10)
            
            ax = sns.barplot(x='popularity', y='title', data=df, palette=colors)
            
            # Añadir valores de popularidad y calificación
            for i, (pop, rating) in enumerate(zip(df['popularity'], df['vote_average'])):
                ax.text(pop + 0.5, i, f"Pop: {pop:.1f} | Rating: {rating:.1f}", va='center')
            
            plt.title('Top 10 Películas por Popularidad', fontsize=16, fontweight='bold')
            plt.xlabel('Índice de Popularidad', fontsize=12)
            plt.ylabel('Película', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_path}/top_peliculas_popularidad.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico 'Top 10 películas por popularidad' guardado")
            
            # Gráfico interactivo con Plotly
            try:
                fig = px.bar(
                    df, x='popularity', y='title', 
                    color='vote_average', color_continuous_scale='viridis',
                    hover_data=['vote_count'],
                    labels={'popularity': 'Índice de Popularidad', 'title': 'Película', 
                            'vote_average': 'Calificación', 'vote_count': 'Total de Votos'},
                    title='Top 10 Películas por Popularidad'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig.write_html(f"{output_path}/top_peliculas_popularidad_interactive.html")
            except Exception as e:
                logger.error(f"Error al crear gráfico interactivo: {e}")
        
        # Visualización 2: Géneros más populares
        if "Géneros más populares" in results and not results["Géneros más populares"].empty:
            plt.figure(figsize=(14, 8))
            df = results["Géneros más populares"]
            
            # Paleta de colores basada en la calificación media
            colors = plt.cm.plasma(df['avg_rating'] / 10)
            
            ax = sns.barplot(x='movie_count', y='name', data=df, palette=colors)
            
            # Añadir calificación media como texto
            for i, rating in enumerate(df['avg_rating']):
                ax.text(df['movie_count'].iloc[i] + 0.5, i, f"Rating: {rating:.1f}", va='center')
            
            plt.title('Géneros Cinematográficos por Popularidad', fontsize=16, fontweight='bold')
            plt.xlabel('Número de Películas', fontsize=12)
            plt.ylabel('Género', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_path}/generos_popularidad.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Gráfico de burbujas para relación entre cantidad y calificación
            plt.figure(figsize=(14, 10))
            
            # Calcular el tamaño de las burbujas (normalizado)
            sizes = df['movie_count'] / df['movie_count'].max() * 1000
            
            scatter = plt.scatter(df['movie_count'], df['avg_rating'], 
                                  s=sizes, c=range(len(df)), cmap='viridis', alpha=0.7)
            
            # Añadir etiquetas a cada burbuja
            for i, name in enumerate(df['name']):
                plt.annotate(name, (df['movie_count'].iloc[i], df['avg_rating'].iloc[i]),
                            ha='center', va='center', fontsize=9, fontweight='bold')
            
            plt.title('Relación entre Cantidad de Películas y Calificación Media por Género', fontsize=16, fontweight='bold')
            plt.xlabel('Cantidad de Películas', fontsize=12)
            plt.ylabel('Calificación Media', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_path}/generos_calificacion_burbujas.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráficos de géneros guardados")
            
            # Nube de palabras para géneros
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                      colormap='viridis', max_words=100)
                
                genre_freq = dict(zip(df['name'], df['movie_count']))
                wordcloud.generate_from_frequencies(genre_freq)
                
                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Nube de Palabras de Géneros Cinematográficos', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{output_path}/generos_nube.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.error(f"Error al crear nube de palabras: {e}")
        
        # Visualización 3: Correlación presupuesto-ingresos
        if "Correlación presupuesto-ingresos" in results and not results["Correlación presupuesto-ingresos"].empty:
            plt.figure(figsize=(14, 10))
            df = results["Correlación presupuesto-ingresos"]
            
            # Gráfico de dispersión con línea de tendencia
            sns.set_style("whitegrid")
            scatter = sns.regplot(x='budget_millions', y='revenue_millions', data=df, 
                                 scatter_kws={'s': df['popularity']*5, 'alpha': 0.7},
                                 line_kws={'color': 'red'})
            
            # Colorear puntos según calificación
            scatter_plot = plt.scatter(df['budget_millions'], df['revenue_millions'], 
                                      s=df['popularity']*5, c=df['vote_average'], 
                                      cmap='viridis', alpha=0.7)
            
            # Añadir barra de color
            cbar = plt.colorbar(scatter_plot)
            cbar.set_label('Calificación (0-10)', fontsize=10)
            
            # Marcar la línea de equilibrio (ingresos = presupuesto)
            max_val = max(df['budget_millions'].max(), df['revenue_millions'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Punto de equilibrio')
            
            # Anotar algunas películas destacadas
            for i, row in df.iterrows():
                if (row['revenue_millions'] > row['budget_millions']*3) or (row['budget_millions'] > 100):
                    plt.annotate(row['title'], (row['budget_millions'], row['revenue_millions']),
                                fontsize=8, ha='center', va='bottom', 
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            plt.title('Relación entre Presupuesto e Ingresos de Películas', fontsize=16, fontweight='bold')
            plt.xlabel('Presupuesto (Millones $)', fontsize=12)
            plt.ylabel('Ingresos (Millones $)', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_path}/presupuesto_ingresos.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de correlación presupuesto-ingresos guardado")
        
        # Visualización 4: Análisis de películas de mayor impacto
        if "Películas de mayor impacto" in results and not results["Películas de mayor impacto"].empty:
            plt.figure(figsize=(14, 10))
            df = results["Películas de mayor impacto"].head(10)  # Top 10
            
            # Crear gráfico horizontal con impacto desglosado
            plt.figure(figsize=(14, 10))
            
            # Preparar datos desglosados
            df = df.sort_values('impact_score', ascending=True)  # Para que la mayor esté arriba
            popularity_contribution = df['popularity'] * 0.6
            rating_contribution = df['vote_average'] * 10 * 0.4
            
            # Crear barras apiladas
            x_pos = range(len(df))
            plt.barh(x_pos, popularity_contribution, color='#1976D2', alpha=0.8, label='Contribución de Popularidad')
            plt.barh(x_pos, rating_contribution, left=popularity_contribution, color='#FFC107', alpha=0.8, 
                    label='Contribución de Calificación')
            
            # Añadir títulos de películas
            plt.yticks(x_pos, df['title'])
            
            # Añadir texto de valores
            for i, row in enumerate(df.itertuples()):
                # Texto para popularidad
                plt.text(popularity_contribution.iloc[i]/2, i, f"Pop: {row.popularity:.1f}", 
                        ha='center', va='center', fontsize=9, fontweight='bold', color='white')
                
                # Texto para calificación
                plt.text(popularity_contribution.iloc[i] + rating_contribution.iloc[i]/2, i, f"Rating: {row.vote_average:.1f}", 
                        ha='center', va='center', fontsize=9, fontweight='bold', color='black')
                
                # Texto para score total
                plt.text(row.impact_score + 1, i, f"Score: {row.impact_score:.1f}", 
                        ha='left', va='center', fontsize=10)
            
            plt.title('Top 10 Películas de Mayor Impacto', fontsize=16, fontweight='bold')
            plt.xlabel('Score de Impacto (Contribución de Popularidad + Calificación)', fontsize=12)
            plt.legend(loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_path}/peliculas_mayor_impacto.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de películas de mayor impacto guardado")
        
        # Visualización 5: Distribución por año
        if "Distribución por año" in results and not results["Distribución por año"].empty:
            plt.figure(figsize=(16, 10))
            df = results["Distribución por año"]
            
            # Filtrar años válidos y recientes
            df = df[df['year'] > 1950].sort_values('year')
            
            # Crear figura con dos subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
            
            # Gráfico 1: Cantidad de películas por año
            ax1.plot(df['year'], df['movie_count'], 'b-o', linewidth=2, markersize=6)
            ax1.set_ylabel('Número de Películas', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.set_title('Evolución del Cine a lo Largo del Tiempo', fontsize=16, fontweight='bold')
            
            # Gráfico 2: Popularidad y calificación media por año
            ax2.plot(df['year'], df['avg_popularity'], 'r-o', linewidth=2, markersize=6, label='Popularidad Media')
            
            # Añadir segundo eje Y para calificación
            ax3 = ax2.twinx()
            ax3.plot(df['year'], df['avg_rating'], 'g-o', linewidth=2, markersize=6, label='Calificación Media')
            
            # Configurar ejes
            ax2.set_xlabel('Año', fontsize=12)
            ax2.set_ylabel('Popularidad Media', fontsize=12, color='red')
            ax3.set_ylabel('Calificación Media', fontsize=12, color='green')
            
            ax2.tick_params(axis='y', labelcolor='red')
            ax3.tick_params(axis='y', labelcolor='green')
            
            # Añadir leyenda combinada
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            # Mejorar apariencia
            plt.tight_layout()
            fig.subplots_adjust(hspace=0.1)
            
            # Guardar figura
            plt.savefig(f"{output_path}/evolucion_cine.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de evolución del cine guardado")
        
        # Visualización 6: Panel de métricas generales si tenemos todos los datos
        if "all_movies" in results and not results["all_movies"].empty:
            df = results["all_movies"].copy()
            
            # Limpiar y transformar datos
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            df = df.dropna(subset=['release_year'])
            df['release_year'] = df['release_year'].astype(int)
            
            # Calcular ROI y otras métricas
            df['roi'] = df.apply(lambda x: (x['revenue'] - x['budget'])/x['budget'] if x['budget'] > 0 else 0, axis=1)
            df['is_profitable'] = df['revenue'] > df['budget']
            
            # Crear figura con subplots
            fig = plt.figure(figsize=(20, 20))
            gs = gridspec.GridSpec(3, 2, figure=fig)
            
            # 1. Distribución de popularidad
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(df['popularity'], bins=30, kde=True, color='blue', ax=ax1)
            ax1.set_title('Distribución de Popularidad', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Popularidad', fontsize=12)
            ax1.set_ylabel('Frecuencia', fontsize=12)
            
            # 2. Distribución de calificaciones
            ax2 = fig.add_subplot(gs[0, 1])
            sns.histplot(df['vote_average'], bins=20, kde=True, color='green', ax=ax2)
            ax2.set_title('Distribución de Calificaciones', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Calificación', fontsize=12)
            ax2.set_ylabel('Frecuencia', fontsize=12)
            
            # 3. Scatter plot presupuesto vs ingresos con ROI
            ax3 = fig.add_subplot(gs[1, 0])
            profitable = df[df['is_profitable']]
            unprofitable = df[~df['is_profitable']]
            
            # Limitar a películas con presupuesto >0 para evitar distorsiones
            profitable = profitable[profitable['budget'] > 0]
            unprofitable = unprofitable[unprofitable['budget'] > 0]
            
            # Convertir a millones para mejor visualización
            ax3.scatter(profitable['budget']/1e6, profitable['revenue']/1e6, label='Rentable', 
                       alpha=0.6, c='green', s=30)
            ax3.scatter(unprofitable['budget']/1e6, unprofitable['revenue']/1e6, label='No Rentable', 
                       alpha=0.6, c='red', s=30)
            
            # Línea de equilibrio
            max_val = max(df['budget'].max(), df['revenue'].max()) / 1e6
            ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax3.set_title('Presupuesto vs Ingresos', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Presupuesto (Millones $)', fontsize=12)
            ax3.set_ylabel('Ingresos (Millones $)', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Boxplot de popularidad por género
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Procesar géneros (están en formato de lista como string)
            genre_data = []
            for _, row in df.iterrows():
                if pd.notna(row['genres']) and row['genres']:
                    genres = row['genres'].split(',')
                    for genre in genres:
                        if genre.strip():
                            genre_data.append({
                                'title': row['title'],
                                'genre': genre.strip(),
                                'popularity': row['popularity'],
                                'vote_average': row['vote_average']
                            })
            
            if genre_data:
                genre_df = pd.DataFrame(genre_data)
                
                # Agregar por género y obtener los principales
                top_genres = genre_df.groupby('genre')['popularity'].count().reset_index()
                top_genres = top_genres.sort_values('popularity', ascending=False).head(10)['genre'].tolist()
                
                # Filtrar solo los géneros principales
                genre_df_filtered = genre_df[genre_df['genre'].isin(top_genres)]
                
                # Crear boxplot
                sns.boxplot(x='genre', y='popularity', data=genre_df_filtered, ax=ax4)
                ax4.set_title('Distribución de Popularidad por Género', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Género', fontsize=12)
                ax4.set_ylabel('Popularidad', fontsize=12)
                ax4.tick_params(axis='x', rotation=45)
            
            # 5. Evolución de calificaciones por año
            ax5 = fig.add_subplot(gs[2, 0])
            
            # Agrupar por año
            year_data = df.groupby('release_year').agg({
                'popularity': 'mean',
                'vote_average': 'mean',
                'vote_count': 'mean',
                'budget': 'mean',
                'revenue': 'mean',
                'tmdb_id': 'count'
            }).reset_index()
            
            year_data = year_data[(year_data['release_year'] >= 1980) & (year_data['release_year'] <= 2023)]
            
            ax5.plot(year_data['release_year'], year_data['vote_average'], 'b-o', linewidth=2)
            ax5.set_title('Evolución de Calificaciones por Año', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Año', fontsize=12)
            ax5.set_ylabel('Calificación Media', fontsize=12)
            ax5.grid(True, alpha=0.3)
            
            # 6. Relación entre número de votos y calificación
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.scatter(df['vote_count'], df['vote_average'], alpha=0.5, c=df['popularity'], cmap='viridis')
            ax6.set_title('Relación entre Número de Votos y Calificación', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Número de Votos', fontsize=12)
            ax6.set_ylabel('Calificación', fontsize=12)
            ax6.set_xscale('log')  # Escala logarítmica para mejor visualización
            ax6.grid(True, alpha=0.3)
            
            # Ajustar espaciado y guardar
            plt.tight_layout()
            plt.savefig(f"{output_path}/dashboard_metricas.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Dashboard de métricas guardado")
        
        logger.info(f"Visualizaciones guardadas en {output_path}")
        return output_path
    
    def train_ml_model(self, results, batch_number):
        """Entrena modelos avanzados de Machine Learning para predecir el éxito de películas."""
        logger.info("Entrenando modelos avanzados para predicción de éxito de películas...")
        output_path = f"{OUTPUT_DIR}/batch_{batch_number}"
        os.makedirs(output_path, exist_ok=True)

        # Si no tenemos datos completos, intentamos obtenerlos
        if "all_movies" not in results or results["all_movies"].empty:
            try:
                with self.engine.connect() as conn:
                    query = """
                        SELECT 
                            m.tmdb_id, m.title, m.release_date, m.popularity, m.vote_average, 
                            m.vote_count, m.budget, m.revenue, m.runtime, genre, director,
                            roi, popularity_level, rating_level, release_year
                        FROM movie_data_warehouse m
                    """
                    all_movies = pd.read_sql(query, conn)
                    results["all_movies"] = all_movies
                    logger.info(f"Datos completos obtenidos: {len(all_movies)} películas")
            except Exception as e:
                logger.error(f"Error al obtener datos completos: {e}")
                logger.error(traceback.format_exc())
                return None

        df = results["all_movies"]
        
        if len(df) < 20:
            logger.warning("No hay suficientes datos para entrenar un modelo robusto")
            return None

        try:
            # Preprocesamiento de datos
            logger.info("Iniciando preprocesamiento de datos...")
            
            # Limpiar y transformar datos
            df = df.copy()
            df = df.fillna(0)
            
            # Convertir a tipos numéricos apropiados
            numeric_cols = ['popularity', 'vote_average', 'vote_count', 'budget', 'revenue', 'runtime', 'roi']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Extraer o asegurar que tenemos año de lanzamiento
            if 'release_year' not in df.columns and 'release_date' in df.columns:
                df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
                df['release_year'] = df['release_year'].fillna(2000).astype(int)
            elif 'release_year' in df.columns:
                df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2000).astype(int)
            
            # Crear características adicionales para el modelo
            df['budget_million'] = df['budget'] / 1_000_000
            df['revenue_million'] = df['revenue'] / 1_000_000
            
            # Asegurar que tenemos el género principal como 'main_genre'
            if 'main_genre' not in df.columns:
                if 'genre' in df.columns:
                    df['main_genre'] = df['genre'].fillna('Unknown')
                elif 'genres' in df.columns:
                    # Si genres es una cadena separada por comas
                    if isinstance(df['genres'].iloc[0], str):
                        df['main_genre'] = df['genres'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) and x else 'Unknown')
                    # Si genres es una lista
                    elif isinstance(df['genres'].iloc[0], list):
                        df['main_genre'] = df['genres'].apply(lambda x: x[0] if x and len(x) > 0 else 'Unknown')
                    else:
                        df['main_genre'] = 'Unknown'
                else:
                    df['main_genre'] = 'Unknown'
            
            # Crear una métrica compuesta de éxito
            # Primero normalizar componentes entre 0 y 1
            pop_max = df['popularity'].max()
            rating_max = 10  # Escala estándar
            roi_max = df['roi'].clip(0, 3).max()  # Limitar ROI extremos
            
            df['popularity_norm'] = df['popularity'] / pop_max if pop_max > 0 else df['popularity']
            df['rating_norm'] = df['vote_average'] / rating_max
            df['roi_norm'] = df['roi'].clip(0, 3) / 3  # Normalizar ROI y recortar valores extremos
            
            # Métrica compuesta con pesos ajustados
            df['success_score'] = (
                0.5 * df['popularity_norm'] + 
                0.3 * df['rating_norm'] + 
                0.2 * df['roi_norm']
            ) * 100  # Escalar a 0-100 para mejor interpretación
            
            # Filtrar filas con valores válidos
            df = df.dropna(subset=['success_score'])
            df = df[df['vote_count'] > 10]  # Filtrar películas con pocos votos
            
            if len(df) < 20:
                logger.warning("Datos insuficientes después de filtrar")
                return None
            
            # Seleccionar características para el modelo
            features = []
            
            # Añadir características numéricas disponibles
            for feature in ['budget_million', 'revenue_million', 'runtime', 'vote_count', 
                        'vote_average', 'release_year']:
                if feature in df.columns:
                    features.append(feature)
            
            # Añadir características categóricas
            categorical_features = []
            if 'main_genre' in df.columns:
                categorical_features.append('main_genre')
                features.append('main_genre')
            
            # Separar variables y target
            X = df[features]
            y = df['success_score']
            
            # Dividir en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            logger.info(f"Datos de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
            
            # Preprocesadores para diferentes tipos de características
            numeric_features = [f for f in features if f not in categorical_features]
            
            # Crear preprocesadores
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            transformers = [('num', numeric_transformer, numeric_features)]
            
            if categorical_features:
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                transformers.append(('cat', categorical_transformer, categorical_features))
            
            # Combinar preprocesadores
            preprocessor = ColumnTransformer(transformers=transformers)
            
            # Modelo XGBoost optimizado
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,  # Regularización L1
                reg_lambda=1.0,  # Regularización L2
                random_state=42
            )
            
            # Crear pipeline completo
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', xgb_model)
            ])
            
            # Entrenar el modelo
            logger.info("Entrenando modelo XGBoost...")
            pipeline.fit(X_train, y_train)
            
            # Evaluar modelo en conjunto de prueba
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(f"Rendimiento del modelo:")
            logger.info(f"- R² Score: {r2:.4f}")
            logger.info(f"- MSE: {mse:.4f}")
            logger.info(f"- RMSE: {rmse:.4f}")
            logger.info(f"- MAE: {mae:.4f}")
            
            # Extraer información de importancia de características
            try:
                # Obtener el modelo del pipeline
                model = pipeline.named_steps['model']
                
                # Preprocesar datos para obtener nombres de características
                preprocessor = pipeline.named_steps['preprocessor']
                X_transformed = preprocessor.transform(X_train)
                
                # Obtener nombres de características
                all_feature_names = []
                
                # Para características numéricas
                for name in numeric_features:
                    all_feature_names.append(name)
                
                # Para características categóricas
                if categorical_features:
                    cat_transformer = preprocessor.named_transformers_['cat'] if 'cat' in preprocessor.named_transformers_ else None
                    if cat_transformer:
                        onehot = cat_transformer.named_steps['onehot']
                        categories = onehot.categories_
                        for i, feature in enumerate(categorical_features):
                            for category in categories[i]:
                                all_feature_names.append(f"{feature}_{category}")
                
                # Ajustar longitud si es necesario
                feature_importances = model.feature_importances_
                if len(all_feature_names) != len(feature_importances):
                    logger.warning(f"Discrepancia de nombres de características: {len(all_feature_names)} vs {len(feature_importances)}")
                    # Ajustar all_feature_names
                    if len(all_feature_names) > len(feature_importances):
                        all_feature_names = all_feature_names[:len(feature_importances)]
                    else:
                        all_feature_names.extend([f"feature_{i}" for i in range(len(all_feature_names), len(feature_importances))])
                
                # Crear DataFrame de importancia
                feature_importance = pd.DataFrame({
                    'feature': all_feature_names,
                    'importance': feature_importances
                })
                
                # Normalizar importancias
                feature_importance['importance'] = feature_importance['importance'] / feature_importance['importance'].sum()
                
                # Ordenar por importancia
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
            except Exception as e:
                logger.error(f"Error al extraer importancia de características: {e}")
                logger.error(traceback.format_exc())
                
                # Crear importancia predeterminada si falla
                feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': [1/len(features)] * len(features)
                })
            
            # Guardar modelo y métricas
            model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = f"{output_path}/movie_success_model"
            os.makedirs(model_dir, exist_ok=True)
            
            # También guardar en directorios adicionales para redundancia
            permanent_dirs = [
                f"{OUTPUT_DIR}/xgb_model_latest",
                f"{OUTPUT_DIR}/ml_models/latest_model",
                "data/movie_analytics/xgb_model_latest",
                "data/latest_xgboost_model",
                "data/public_models"
            ]
            
            for dir_path in permanent_dirs:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Creado directorio: {dir_path}")
                
                # Limpiar directorios de destino para evitar archivos antiguos
                try:
                    for existing_file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, existing_file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Eliminado archivo anterior: {file_path}")
                except Exception as e:
                    logger.error(f"Error al limpiar {dir_path}: {e}")
            
            # Guardar modelo
            model_files = {
                "model.pkl": pipeline,
                "model_final.pkl": pipeline  # Duplicado para compatibilidad
            }
            
            for filename, model_obj in model_files.items():
                # Guardar en el directorio original
                original_path = os.path.join(model_dir, filename)
                with open(original_path, 'wb') as f:
                    pickle.dump(model_obj, f)
                logger.info(f"Modelo guardado en: {original_path}")
                
                # Copiar a los directorios permanentes
                for perm_dir in permanent_dirs:
                    perm_path = os.path.join(perm_dir, filename)
                    try:
                        # Guardar directamente
                        with open(perm_path, 'wb') as f:
                            pickle.dump(model_obj, f)
                        
                        # Establecer permisos amplios
                        os.chmod(perm_path, 0o777)
                        logger.info(f"Modelo guardado y permisos establecidos en: {perm_path}")
                    except Exception as e:
                        logger.error(f"Error al guardar en {perm_path}: {e}")
            
            # Guardar métricas y feature importance
            metrics = {
                'model_type': 'XGBoost',
                'r2': float(r2),
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'feature_importance': feature_importance.to_dict('records'),
                'timestamp': model_timestamp
            }
            
            for perm_dir in [model_dir] + permanent_dirs:
                metrics_path = os.path.join(perm_dir, "metrics.json")
                feature_path = os.path.join(perm_dir, "feature_importance.csv")
                
                try:
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    os.chmod(metrics_path, 0o777)
                    
                    feature_importance.to_csv(feature_path, index=False)
                    os.chmod(feature_path, 0o777)
                    
                    logger.info(f"Métricas y características guardadas en: {perm_dir}")
                except Exception as e:
                    logger.error(f"Error al guardar métricas en {perm_dir}: {e}")
            
            # Crear archivo de ubicación del modelo
            model_info_paths = [
                f"{OUTPUT_DIR}/model_location.txt",
                "data/model_location.txt",
                "model_location.txt"
            ]
            
            model_info_content = (
                "data/latest_xgboost_model/model_final.pkl\n"
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "xgboost\n"
                "data/public_models/model_final.pkl\n"
            )
            
            for info_path in model_info_paths:
                try:
                    with open(info_path, 'w') as f:
                        f.write(model_info_content)
                    os.chmod(info_path, 0o777)
                    logger.info(f"Creado archivo de ubicación de modelo en: {info_path}")
                except Exception as e:
                    logger.error(f"Error al crear archivo de ubicación en {info_path}: {e}")
            
            # Crear archivo de señalización
            try:
                with open(f"{OUTPUT_DIR}/model_ready.txt", 'w') as f:
                    f.write("yes")
                os.chmod(f"{OUTPUT_DIR}/model_ready.txt", 0o777)
                
                with open("data/model_ready.txt", 'w') as f:
                    f.write("yes")
                os.chmod("data/model_ready.txt", 0o777)
                
                logger.info("Archivos de señalización creados")
            except Exception as e:
                logger.error(f"Error al crear archivos de señalización: {e}")
            
            # Generar visualizaciones para el modelo
            self.generate_model_visualizations(pipeline, feature_importance, X_test, y_test, y_pred, model_dir)
            
            logger.info(f"Modelo entrenado y guardado en múltiples ubicaciones")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error en el entrenamiento del modelo: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_model_visualizations(self, model, feature_importance, X_test, y_test, y_pred, output_dir):
        """Genera visualizaciones para evaluar el rendimiento del modelo."""
        try:
            # 1. Gráfico de importancia de características
            plt.figure(figsize=(12, 8))
            
            # Tomar top 15 características
            top_features = feature_importance.head(15)
            
            # Crear gráfico de barras horizontales
            sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
            
            plt.title('Importancia de Características en la Predicción', fontsize=16, fontweight='bold')
            plt.xlabel('Importancia Relativa', fontsize=12)
            plt.ylabel('Característica', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Gráfico de predicciones vs reales
            plt.figure(figsize=(10, 8))
            
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Línea de predicción perfecta
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Valores Reales vs. Predicciones', fontsize=16, fontweight='bold')
            plt.xlabel('Valores Reales', fontsize=12)
            plt.ylabel('Predicciones', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predictions_vs_actual.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Gráfico de residuos
            plt.figure(figsize=(10, 8))
            
            residuals = y_test - y_pred
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.title('Gráfico de Residuos', fontsize=16, fontweight='bold')
            plt.xlabel('Predicciones', fontsize=12)
            plt.ylabel('Residuos', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Histograma de residuos
            plt.figure(figsize=(10, 8))
            
            sns.histplot(residuals, kde=True)
            
            plt.title('Distribución de Residuos', fontsize=16, fontweight='bold')
            plt.xlabel('Residuo', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/residuals_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizaciones del modelo guardadas en {output_dir}")
            
        except Exception as e:
            logger.error(f"Error al generar visualizaciones del modelo: {e}")
            logger.error(traceback.format_exc())
    
    def process_data(self):
        """Procesa continuamente los datos de Kafka."""
        buffer = []
        batch_count = 0
        
        logger.info(f"Esperando mensajes en el topic '{KAFKA_TOPIC}'...")
        try:
            for message in self.consumer:
                movie_data = message.value
                logger.info(f"Recibido: {movie_data.get('title', 'Desconocido')}")
                buffer.append(movie_data)
                
                if len(buffer) >= 10:  # Procesar cada 10 películas
                    batch_count += 1
                    logger.info(f"Procesando lote #{batch_count} ({len(buffer)} películas)")
                    
                    # Procesar datos con Dask
                    processed_df = self.process_batch(buffer)
                    
                    # Guardar en PostgreSQL
                    processed_count = self.save_to_database(processed_df)
                    
                    # Si hay nuevos datos, realizar análisis
                    if processed_count > 0:
                        # Realizar análisis
                        analysis_results = self.analyze_data()
                        
                        # Generar visualizaciones
                        self.generate_visualizations(analysis_results, batch_count)
                        
                        # Entrenar modelo ML
                        self.train_ml_model(analysis_results, batch_count)
                    
                    buffer = []
                    
                    # Pequeña pausa para no sobrecargar
                    time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Proceso interrumpido por el usuario")
        except Exception as e:
            logger.error(f"Error durante el procesamiento: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Libera recursos."""
        logger.info("Limpiando recursos...")
        try:
            self.consumer.close()
            self.session.close()
            self.client.close()
            logger.info("Recursos liberados correctamente")
        except Exception as e:
            logger.error(f"Error al liberar recursos: {e}")

if __name__ == "__main__":
    try:
        analyzer = MovieAnalyzer()
        # Probar la conexión antes de procesar datos
        if analyzer.test_postgres_connection():
            analyzer.process_data()
        else:
            logger.error("No se pudo conectar a PostgreSQL. Verifique la configuración.")
    except Exception as e:
        logger.error(f"Error al iniciar el analizador: {e}")
        logger.error(traceback.format_exc())