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

# Configuración de PostgreSQL (ajustada para conexión correcta)
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Villeta-11"  # Agregada la contraseña correcta
POSTGRES_HOST = "movie_postgres"
POSTGRES_PORT = "5433"
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
        
        return df
    
    def calculate_metrics(self, df):
        """Calcula métricas adicionales para análisis."""
        # Calcular retorno sobre inversión (ROI)
        df['roi'] = df.apply(
            lambda row: (row['revenue'] - row['budget']) / row['budget'] if row['budget'] > 0 else 0, 
            axis=1
        )
        
        # Categorizar películas por popularidad
        conditions = [
            (df['popularity'] > 20),
            (df['popularity'] > 10),
            (df['popularity'] > 5)
        ]
        choices = ['Alta', 'Media', 'Baja']
        df['popularity_category'] = np.select(conditions, choices, default='Muy Baja')
        
        # Categorizar por calificación
        conditions = [
            (df['vote_average'] >= 8),
            (df['vote_average'] >= 6),
            (df['vote_average'] >= 4)
        ]
        choices = ['Excelente', 'Buena', 'Regular']
        df['rating_category'] = np.select(conditions, choices, default='Mala')
        
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
            if movie_data.get('release_date') and len(movie_data.get('release_date')) >= 4:
                try:
                    release_year = int(movie_data.get('release_date')[:4])
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
        
        return results
    
    def generate_visualizations(self, results, batch_number):
        """Genera visualizaciones a partir de los resultados del análisis."""
        logger.info("Generando visualizaciones...")
        output_path = f"{OUTPUT_DIR}/batch_{batch_number}"
        os.makedirs(output_path, exist_ok=True)
        
        # Visualización 1: Top 10 películas por popularidad
        if "Top 10 películas por popularidad" in results:
            plt.figure(figsize=(12, 8))
            df = results["Top 10 películas por popularidad"]
            
            if not df.empty:
                sns.barplot(x='popularity', y='title', data=df)
                plt.title('Top 10 Películas por Popularidad')
                plt.tight_layout()
                plt.savefig(f"{output_path}/top_peliculas_popularidad.png")
                plt.close()
                logger.info(f"Gráfico 'Top 10 películas por popularidad' guardado")
        
        # Visualización 2: Géneros más populares
        if "Géneros más populares" in results:
            plt.figure(figsize=(14, 8))
            df = results["Géneros más populares"]
            
            if not df.empty:
                sns.barplot(x='movie_count', y='name', data=df)
                plt.title('Géneros más Populares por Número de Películas')
                plt.tight_layout()
                plt.savefig(f"{output_path}/generos_populares.png")
                plt.close()
                
                # Segunda visualización de géneros
                plt.figure(figsize=(14, 8))
                sns.scatterplot(x='movie_count', y='avg_rating', size='movie_count', 
                               sizes=(50, 400), data=df)
                for _, row in df.iterrows():
                    plt.text(row['movie_count'], row['avg_rating'], row['name'])
                plt.title('Relación entre Cantidad de Películas y Valoración Media por Género')
                plt.xlabel('Cantidad de Películas')
                plt.ylabel('Valoración Media')
                plt.tight_layout()
                plt.savefig(f"{output_path}/generos_valoracion.png")
                plt.close()
                logger.info(f"Gráficos de géneros guardados")
        
        # Visualización 3: Análisis del data warehouse
        if "Análisis de data warehouse" in results:
            plt.figure(figsize=(14, 8))
            df = results["Análisis de data warehouse"]
            
            if not df.empty:
                # Filtrar solo los géneros principales para mejor visualización
                top_genres = df.head(10)
                
                # Gráfico de barras para cantidad de películas por género
                plt.subplot(1, 2, 1)
                sns.barplot(x='count', y='genre', data=top_genres)
                plt.title('Número de Películas por Género')
                
                # Gráfico de barras para valoración media por género
                plt.subplot(1, 2, 2)
                sns.barplot(x='avg_rating', y='genre', data=top_genres)
                plt.title('Valoración Media por Género')
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/analisis_warehouse.png")
                plt.close()
                logger.info(f"Gráfico de análisis del data warehouse guardado")
        
        logger.info(f"Visualizaciones guardadas en {output_path}")
        return output_path
    
    def train_ml_model(self, results, batch_number):
        """Entrena un modelo de Machine Learning para predecir el éxito de películas."""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import pickle
        
        logger.info("Entrenando modelo de ML para predicción de éxito de películas...")
        output_path = f"{OUTPUT_DIR}/batch_{batch_number}"
        os.makedirs(output_path, exist_ok=True)
        
        # Consultar datos para el entrenamiento
        query = """
            SELECT m.tmdb_id, m.title, m.popularity, m.vote_average, m.vote_count, 
                   m.budget, m.revenue, m.runtime, 
                   CASE WHEN m.budget > 0 THEN m.revenue::float / m.budget ELSE 0 END AS roi
            FROM movies m
            WHERE m.vote_count > 0
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if len(df) < 10:
                logger.warning("No hay suficientes datos para entrenar un modelo")
                return None
            
            logger.info(f"Datos cargados para entrenamiento: {len(df)} registros")
            
            # Crear target: combinar popularidad y calificación
            df['success_score'] = df['popularity'] * 0.7 + df['vote_average'] * 10 * 0.3
            
            # Preparar datos
            features = ['budget', 'runtime', 'vote_count', 'revenue']
            X = df[features]
            y = df['success_score']
            
            # Dividir en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Métricas del modelo - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Guardar modelo
            with open(f"{output_path}/movie_success_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar importancia de características
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(f"{output_path}/feature_importance.csv", index=False)
            
            # Visualizar importancia de características
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Importancia de Características')
            plt.tight_layout()
            plt.savefig(f"{output_path}/feature_importance.png")
            plt.close()
            
            # Predicciones
            df['predicted_success'] = model.predict(df[features])
            
            # Top 10 películas con mayor éxito predicho
            top_predicted = df.sort_values('predicted_success', ascending=False).head(10)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='predicted_success', y='title', data=top_predicted)
            plt.title('Top 10 Películas con Mayor Éxito Predicho')
            plt.tight_layout()
            plt.savefig(f"{output_path}/top_predicted.png")
            plt.close()
            
            logger.info(f"Modelo ML entrenado y guardado en {output_path}")
            return {
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance.to_dict('records'),
                'top_predicted': top_predicted['title'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {e}")
            logger.error(traceback.format_exc())
            return None
    
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