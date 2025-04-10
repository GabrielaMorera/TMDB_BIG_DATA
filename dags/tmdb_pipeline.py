from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import xgboost as xgb

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
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import warnings
from wordcloud import WordCloud

# Ignorar advertencias
warnings.filterwarnings('ignore')

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
POSTGRES_PORT = "5432"  # Puerto dentro de Docker
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

# Función para almacenar el modelo ML entrenado
def store_ml_model(**kwargs):
    """Store the trained ML model in a specific location for Streamlit, ensuring the latest is always used."""
    import shutil
    import os
    import json
    import logging
    from datetime import datetime
    import traceback
    import glob

    logger = logging.getLogger(__name__)
    ti = kwargs['ti']
    model_dir = ti.xcom_pull(key='model_dir', task_ids='train_ml_model')

    if not model_dir or not os.path.exists(model_dir):
        logger.error("No model directory found to store. Attempting to locate any existing model...")
        search_patterns = [
            'data/movie_analytics/xgb_model_*',
            'data/movie_analytics/ensemble_model_*',
            'data/movie_analytics/ml_model_v3_*',
        ]
        model_dirs = []
        for pattern in search_patterns:
            model_dirs.extend(glob.glob(pattern))

        if model_dirs:
            model_dir = max(model_dirs, key=os.path.getmtime)
            logger.info(f"Using existing model from: {model_dir}")
        else:
            logger.warning("No model directories found.")
            return None

    # Define the permanent storage location for the latest model
    permanent_model_dir = "/opt/airflow/data/latest_xgboost_model"
    os.makedirs(permanent_model_dir, exist_ok=True)

    try:
        # Find the most recent model file in the source directory
        model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
        if not model_files:
            logger.error(f"No model files found in {model_dir}")
            return None

        # Prefer model_final.pkl if it exists, otherwise take the most recent
        model_path = os.path.join(model_dir, "model_final.pkl") if os.path.exists(os.path.join(model_dir, "model_final.pkl")) else max(model_files, key=os.path.getmtime)

        # Clear the destination directory
        for existing_file in os.listdir(permanent_model_dir):
            file_path = os.path.join(permanent_model_dir, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")

        # Copy the latest model file to the destination
        dst_model_path = os.path.join(permanent_model_dir, "model.pkl")
        shutil.copy2(model_path, dst_model_path)
        os.chmod(dst_model_path, 0o777)  # Set permissions

        logger.info(f"Latest model copied to: {dst_model_path}")

        # Create a file to indicate the model's location
        model_location_path = "/opt/airflow/data/model_location.txt"
        with open(model_location_path, 'w') as f:
            f.write(os.path.abspath(dst_model_path))  # Write the absolute path
        os.chmod(model_location_path, 0o777)
        logger.info(f"Model location file created: {model_location_path}")

        return permanent_model_dir

    except Exception as e:
        logger.error(f"Error storing model: {e}")
        logger.error(traceback.format_exc())
        return None
    
# Función para crear las tablas en PostgreSQL
def create_postgres_tables():
    """Crea las tablas necesarias en PostgreSQL si no existen."""
    try:
        import psycopg2

        # Intentar conexión a PostgreSQL
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )

        # Crear cursor y ejecutar creación de tablas
        cursor = conn.cursor()

        # Tabla principal de películas
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id SERIAL PRIMARY KEY,
            tmdb_id INTEGER UNIQUE,
            title VARCHAR(255),
            overview TEXT,
            release_date VARCHAR(20),
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Tabla de géneros
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS genres (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE
        )
        """)

        # Tabla de relación película-género
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_genres (
            id SERIAL PRIMARY KEY,
            movie_id INTEGER REFERENCES movies(id),
            genre_id INTEGER REFERENCES genres(id),
            UNIQUE(movie_id, genre_id)
        )
        """)

        # Tabla de directores
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS directors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE
        )
        """)

        # Tabla de relación película-director
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_directors (
            id SERIAL PRIMARY KEY,
            movie_id INTEGER REFERENCES movies(id),
            director_id INTEGER REFERENCES directors(id),
            UNIQUE(movie_id, director_id)
        )
        """)

        # Tabla de actores
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS actors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE
        )
        """)

        # Tabla de relación película-actor
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_actors (
            id SERIAL PRIMARY KEY,
            movie_id INTEGER REFERENCES movies(id),
            actor_id INTEGER REFERENCES actors(id),
            character_name VARCHAR(255),
            UNIQUE(movie_id, actor_id)
        )
        """)

        # Tabla para data warehouse con datos desnormalizados - Ajustada para incluir todas las columnas necesarias
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_data_warehouse (
            id SERIAL PRIMARY KEY,
            tmdb_id INTEGER,
            title VARCHAR(255),
            release_date VARCHAR(20),
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
        )
        """)

        # Aplicar cambios
        conn.commit()

        # Cerrar cursor y conexión
        cursor.close()
        conn.close()

        logger.info("Tablas creadas o verificadas correctamente en PostgreSQL")
        
        # Ahora verificar si necesitamos agregar alguna columna adicional
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        cursor = conn.cursor()
        
        # Verificar y añadir columnas adicionales si no existen
        try:
            # Verificar si release_year existe en movie_data_warehouse
            cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'movie_data_warehouse' AND column_name = 'release_year'
            """)
            
            if not cursor.fetchone():
                cursor.execute("""
                ALTER TABLE movie_data_warehouse 
                ADD COLUMN release_year INTEGER
                """)
                conn.commit()
                logger.info("Columna release_year añadida a movie_data_warehouse")
                
            conn.commit()
        except Exception as e:
            logger.error(f"Error al verificar o añadir columnas: {e}")
            conn.rollback()
        
        cursor.close()
        conn.close()
        
        return True

    except Exception as e:
        logger.error(f"Error al crear tablas en PostgreSQL: {e}")
        logger.error(traceback.format_exc())
        return False
    
# Función para preparar el entorno
def setup_environment(**kwargs):
    """Prepara el entorno para la ejecución del DAG."""
    try:
        # Crear directorios necesarios
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Directorio de datos creado: {OUTPUT_DIR}")

        # Crear subdirectorios
        subdirs = ['visualizations', 'ml_models', 'raw_data', 'processed_data']
        for subdir in subdirs:
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
            logger.info(f"Subdirectorio creado: {os.path.join(OUTPUT_DIR, subdir)}")

        # Verificar dependencias
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import sklearn
            logger.info("Dependencias principales verificadas correctamente")
        except ImportError as e:
            logger.warning(f"Dependencia faltante: {e}")

        # Verificar conexión a PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )
            conn.close()
            logger.info("Conexión a PostgreSQL verificada correctamente")
        except Exception as e:
            logger.warning(f"No se pudo conectar a PostgreSQL: {e}")

        # Crear tablas en PostgreSQL
        create_postgres_tables()

        return "Entorno configurado correctamente"

    except Exception as e:
        logger.error(f"Error al configurar el entorno: {e}")
        logger.error(traceback.format_exc())
        raise

# Función para obtener datos de TMDB y enviarlos a Kafka
def fetch_and_send_to_kafka(**kwargs):
    """Obtiene datos de TMDB y los envía a Kafka para procesamiento."""
    try:
        from kafka import KafkaProducer
        import json
        import random

        # Configuración de parámetros
        ti = kwargs['ti']
        popular_pages = 5  # Número de páginas de películas populares a obtener
        top_rated_pages = 5  # Número de páginas de películas mejor valoradas
        upcoming_pages = 3  # Número de páginas de próximos estrenos
        movie_ids = []  # Lista para almacenar IDs de películas

        # Intentar crear productor Kafka
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"Productor Kafka conectado a {KAFKA_BROKER}")
        except Exception as e:
            logger.error(f"Error al conectar con Kafka: {e}")
            logger.error("Utilizando método alternativo: guardar datos localmente")
            producer = None

        # Headers para API TMDB
        headers = {
            'Authorization': f'Bearer {TMDB_TOKEN}',
            'Content-Type': 'application/json;charset=utf-8'
        }

        # 1. Obtener películas populares
        logger.info("Obteniendo películas populares de TMDB")
        for page in range(1, popular_pages + 1):
            url = f"https://api.themoviedb.org/3/movie/popular?language=es-ES&page={page}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                movies = response.json().get('results', [])
                logger.info(f"Obtenidas {len(movies)} películas populares (página {page})")

                # Extraer IDs
                for movie in movies:
                    movie_ids.append(movie['id'])
            else:
                logger.error(f"Error al obtener películas populares (página {page}): {response.status_code}")

        # 2. Obtener películas mejor valoradas
        logger.info("Obteniendo películas mejor valoradas de TMDB")
        for page in range(1, top_rated_pages + 1):
            url = f"https://api.themoviedb.org/3/movie/top_rated?language=es-ES&page={page}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                movies = response.json().get('results', [])
                logger.info(f"Obtenidas {len(movies)} películas mejor valoradas (página {page})")

                # Extraer IDs
                for movie in movies:
                    movie_ids.append(movie['id'])
            else:
                logger.error(f"Error al obtener películas mejor valoradas (página {page}): {response.status_code}")

        # 3. Obtener próximos estrenos
        logger.info("Obteniendo próximos estrenos de TMDB")
        for page in range(1, upcoming_pages + 1):
            url = f"https://api.themoviedb.org/3/movie/upcoming?language=es-ES&page={page}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                movies = response.json().get('results', [])
                logger.info(f"Obtenidos {len(movies)} próximos estrenos (página {page})")

                # Extraer IDs
                for movie in movies:
                    movie_ids.append(movie['id'])
            else:
                logger.error(f"Error al obtener próximos estrenos (página {page}): {response.status_code}")

        # Eliminar duplicados
        movie_ids = list(set(movie_ids))
        logger.info(f"Total de {len(movie_ids)} IDs de películas únicas recolectadas")

        # Guardar IDs para uso posterior si es necesario
        ids_file = f"{OUTPUT_DIR}/movie_ids_{datetime.now().strftime('%Y%m%d')}.json"
        with open(ids_file, 'w') as f:
            json.dump(movie_ids, f)
        logger.info(f"IDs de películas guardados en {ids_file}")

        # 4. Obtener detalles de cada película y enviar a Kafka
        movie_details = []
        processed_count = 0

        # Limitar número de películas a procesar para evitar saturación
        max_movies = min(len(movie_ids), 100)  # Procesar máximo 100 películas
        selected_ids = random.sample(movie_ids, max_movies)

        logger.info(f"Obteniendo detalles de {max_movies} películas")

        for movie_id in selected_ids:
            try:
                # Obtener detalles de la película
                url = f"https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=credits&language=es-ES"
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    movie_data = response.json()

                    # Extraer datos relevantes
                    movie_detail = {
                        'id': movie_data.get('id'),
                        'title': movie_data.get('title'),
                        'original_title': movie_data.get('original_title'),
                        'overview': movie_data.get('overview'),
                        'release_date': movie_data.get('release_date'),
                        'popularity': movie_data.get('popularity'),
                        'vote_average': movie_data.get('vote_average'),
                        'vote_count': movie_data.get('vote_count'),
                        'budget': movie_data.get('budget'),
                        'revenue': movie_data.get('revenue'),
                        'runtime': movie_data.get('runtime'),
                        'adult': movie_data.get('adult', False),
                        'poster_path': movie_data.get('poster_path'),
                        'backdrop_path': movie_data.get('backdrop_path'),
                        'genres': [genre.get('name') for genre in movie_data.get('genres', [])],
                        'production_companies': [company.get('name') for company in movie_data.get('production_companies', [])],
                        'production_countries': [country.get('name') for country in movie_data.get('production_countries', [])],
                        'timestamp': datetime.now().isoformat()
                    }

                    # Añadir información de créditos si está disponible
                    if 'credits' in movie_data:
                        # Directores (job=Director en crew)
                        directors = [
                            person.get('name') for person in movie_data.get('credits', {}).get('crew', [])
                            if person.get('job') == 'Director'
                        ]
                        movie_detail['directors'] = directors

                        # Actores principales (primeros 10 del cast)
                        cast = [
                            {
                                'name': person.get('name'),
                                'character': person.get('character'),
                                'order': person.get('order')
                            }
                            for person in movie_data.get('credits', {}).get('cast', [])[:10]
                        ]
                        movie_detail['cast'] = cast

                    # Enviar a Kafka o guardar localmente
                    if producer:
                        producer.send(KAFKA_TOPIC, value=movie_detail)
                        producer.flush()  # Garantizar entrega
                        logger.info(f"Película enviada a Kafka: {movie_detail['title']}")
                    else:
                        # Guardar localmente si no hay Kafka
                        movie_details.append(movie_detail)

                    processed_count += 1

                    # Pausa para evitar saturar la API
                    time.sleep(0.5)

                else:
                    logger.warning(f"Error al obtener detalles de película ID {movie_id}: {response.status_code}")

            except Exception as e:
                logger.error(f"Error procesando película ID {movie_id}: {e}")

        # Si no hay Kafka, guardar datos localmente
        if not producer and movie_details:
            output_file = f"{OUTPUT_DIR}/raw_data/movie_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(movie_details, f)
            logger.info(f"Datos de películas guardados localmente en {output_file}")

        # Cerrar productor Kafka
        if producer:
            producer.close()

        logger.info(f"Proceso completado. {processed_count} películas procesadas.")

        # Pasar información a la siguiente tarea
        kwargs['ti'].xcom_push(key='movie_count', value=processed_count)
        kwargs['ti'].xcom_push(key='local_file', value=ids_file)

        return processed_count

    except Exception as e:
        logger.error(f"Error en la obtención de datos TMDB: {e}")
        logger.error(traceback.format_exc())
        raise

# Función para consumir y procesar datos de Kafka
def process_kafka_data(**kwargs):
    """Consume datos de Kafka, los procesa y los prepara para almacenamiento y análisis."""
    try:
        from kafka import KafkaConsumer
        import json
        import pandas as pd
        import numpy as np
        import os
        from datetime import datetime

        # Intentar crear consumidor Kafka
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='movie_analytics_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=60000  # 60 segundos de timeout
            )
            logger.info(f"Consumidor Kafka conectado a {KAFKA_BROKER}, tema {KAFKA_TOPIC}")

            # Esperar datos
            messages = []
            count = 0
            for message in consumer:
                movie_data = message.value
                messages.append(movie_data)
                count += 1
                logger.info(f"Película recibida de Kafka: {movie_data.get('title', 'Sin título')}")

            logger.info(f"Consumidos {count} mensajes de Kafka")
            
            # Si no hay mensajes, usar datos locales
            if count == 0:
                logger.warning("No se recibieron mensajes de Kafka. Usando método alternativo.")
                raise Exception("No hay mensajes de Kafka")

        except Exception as e:
            logger.error(f"Error al conectar con Kafka o consumir mensajes: {e}")
            logger.info("Utilizando método alternativo: cargar datos locales")

            # Si no podemos conectar a Kafka, intentar usar datos guardados localmente
            messages = []
            
            # Buscar en xcom primero
            local_file = kwargs.get('ti', None)
            if local_file:
                local_file = local_file.xcom_pull(key='local_file', task_ids='fetch_and_send_to_kafka')
            
            # Si no tenemos archivo desde xcom, buscar en el directorio
            if not local_file:
                # Crear directorio si no existe
                os.makedirs(f"{OUTPUT_DIR}/raw_data", exist_ok=True)
                
                # Buscar cualquier archivo de datos en el directorio
                data_files = []
                try:
                    data_files = [f for f in os.listdir(f"{OUTPUT_DIR}/raw_data") if f.endswith('.json')]
                except Exception as e:
                    logger.error(f"Error al listar archivos en {OUTPUT_DIR}/raw_data: {e}")
                
                if data_files:
                    local_file = os.path.join(OUTPUT_DIR, "raw_data", data_files[-1])  # Tomar el último
                    logger.info(f"Encontrado archivo local: {local_file}")
                    
                    # Cargar datos del archivo
                    try:
                        with open(local_file, 'r') as f:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                messages = file_data
                            else:
                                messages = [file_data]
                            logger.info(f"Cargados {len(messages)} mensajes desde archivo local")
                    except Exception as e:
                        logger.error(f"Error al cargar archivo {local_file}: {e}")
            
            # Si no hay archivos o no se pudo cargar, crear datos de ejemplo
            if not messages:
                logger.warning("Creando datos de ejemplo para continuar el proceso")
                messages = [
                    {
                        "id": 1,
                        "title": "Película de Ejemplo 1",
                        "release_date": "2023-01-01",
                        "popularity": 15.5,
                        "vote_average": 7.8,
                        "vote_count": 1500,
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

        # Convertir a DataFrame de pandas con manejo de errores
        try:
            df = pd.DataFrame(messages)
        except Exception as e:
            logger.error(f"Error al convertir mensajes a DataFrame: {e}")
            # Si falla, intentar normalizar la estructura primero
            normalized_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    normalized_messages.append(msg)
                elif isinstance(msg, str):
                    try:
                        normalized_messages.append(json.loads(msg))
                    except:
                        logger.error(f"Error al parsear mensaje como JSON: {msg[:100]}...")
            
            df = pd.DataFrame(normalized_messages)
            if df.empty:
                logger.error("No se pudo crear un DataFrame válido")
                return 0

        # Limpiar y procesar datos
        logger.info("Limpiando y transformando datos")
        
        # Verificar columnas existentes para evitar errores
        logger.info(f"Columnas originales: {df.columns.tolist()}")
        
        # Llenar valores nulos para columnas numéricas
        numeric_columns = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
                logger.warning(f"Columna {col} no encontrada, creada con valores 0")

        # Extraer año de lanzamiento
        if 'release_date' in df.columns:
            try:
                df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
            except Exception as e:
                logger.error(f"Error al extraer release_year: {e}")
                df['release_year'] = 2000  # Valor por defecto

        # Procesar géneros
        if 'genres' in df.columns:
            # Verificar si géneros es una lista
            if isinstance(df['genres'].iloc[0] if len(df) > 0 and not df['genres'].iloc[0] is None else [], list):
                # Extraer género principal
                df['main_genre'] = df['genres'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
                )
                # Convertir lista de géneros a string para almacenamiento
                df['genres_str'] = df['genres'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else ''
                )
                # Género individual para movie_data_warehouse
                df['genre'] = df['main_genre']
            else:
                logger.warning("La columna 'genres' no contiene listas, intentando procesarla como string")
                # Intentar procesar como string (posiblemente ya procesado)
                df['genres_str'] = df['genres']
                df['main_genre'] = df['genres'].apply(
                    lambda x: x.split(',')[0].strip() if isinstance(x, str) and x else 'Unknown'
                )
                df['genre'] = df['main_genre']
        else:
            logger.warning("Columna 'genres' no encontrada, usando valores por defecto")
            df['genres_str'] = ''
            df['main_genre'] = 'Unknown'
            df['genre'] = 'Unknown'

        # Procesar directores
        if 'directors' in df.columns:
            if isinstance(df['directors'].iloc[0] if len(df) > 0 and not df['directors'].iloc[0] is None else [], list):
                df['directors_str'] = df['directors'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else ''
                )
                df['director'] = df['directors'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
                )
            else:
                df['directors_str'] = df['directors']
                df['director'] = df['directors'].apply(
                    lambda x: x.split(',')[0].strip() if isinstance(x, str) and x else 'Unknown'
                )
        else:
            df['directors_str'] = ''
            df['director'] = 'Unknown'

        # Procesar reparto
        if 'cast' in df.columns:
            if isinstance(df['cast'].iloc[0] if len(df) > 0 and not df['cast'].iloc[0] is None else [], list):
                df['cast_str'] = df['cast'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else ''
                )
            else:
                df['cast_str'] = df['cast']
        else:
            df['cast_str'] = ''

        # Procesar compañías productoras
        if 'production_companies' in df.columns:
            if isinstance(df['production_companies'].iloc[0] if len(df) > 0 and not df['production_companies'].iloc[0] is None else [], list):
                df['companies_str'] = df['production_companies'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else ''
                )
            else:
                df['companies_str'] = df['production_companies']
        else:
            df['companies_str'] = ''

        # Convertir a millones para el modelo
        if 'budget' in df.columns:
            df['budget_million'] = df['budget'] / 1000000

        if 'revenue' in df.columns:
            df['revenue_million'] = df['revenue'] / 1000000

        # Calcular métricas
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

        # Categorizar películas por popularidad
        df['popularity_category'] = 'Baja'
        df.loc[df['popularity'] > 5, 'popularity_category'] = 'Media'
        df.loc[df['popularity'] > 10, 'popularity_category'] = 'Alta'
        df.loc[df['popularity'] > 20, 'popularity_category'] = 'Muy Alta'

        df['rating_category'] = 'Mala'
        df.loc[df['vote_average'] >= 4, 'rating_category'] = 'Regular'
        df.loc[df['vote_average'] >= 6, 'rating_category'] = 'Buena'
        df.loc[df['vote_average'] >= 8, 'rating_category'] = 'Excelente'

        # Normalizar nombres de variables para consistencia con el modelo
        if 'popularity_category' in df.columns:
            df['popularity_level'] = df['popularity_category']
        
        if 'rating_category' in df.columns:
            df['rating_level'] = df['rating_category']

        # Definir si la película es rentable
        df['is_profitable'] = df['revenue'] > df['budget']

        # Guardar datos procesados
        output_file = f"{OUTPUT_DIR}/processed_movies_v3.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Datos procesados guardados en {output_file}")

        # Guardar también como JSON para mayor compatibilidad
        try:
            json_output = f"{OUTPUT_DIR}/processed_movies_v3.json"
            df.to_json(json_output, orient='records')
            logger.info(f"Datos procesados guardados también como JSON en {json_output}")
        except Exception as e:
            logger.warning(f"No se pudo guardar como JSON: {e}")

        # Pasar a la siguiente tarea
        ti = kwargs.get('ti')
        if ti:
            ti.xcom_push(key='processed_data_path', value=output_file)
            ti.xcom_push(key='movie_count', value=len(df))

        # Cerrar consumidor si existe
        if 'consumer' in locals():
            try:
                consumer.close()
            except:
                pass

        return len(df)

    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")
        logger.error(traceback.format_exc())

        # Asegurar que el consumidor se cierre incluso en caso de error
        if 'consumer' in locals():
            try:
                consumer.close()
            except:
                pass
                
        # Intentar crear un archivo mínimo para que el pipeline pueda continuar
        try:
            import pandas as pd
            # Crear un DataFrame con datos mínimos
            df_min = pd.DataFrame([{
                "id": 1,
                "title": "Película de Ejemplo (Error)",
                "release_date": "2023-01-01",
                "popularity": 10.0,
                "vote_average": 5.0,
                "vote_count": 100,
                "budget": 1000000,
                "revenue": 2000000,
                "runtime": 120,
                "genre": "Drama",
                "main_genre": "Drama",
                "genres_str": "Drama",
                "director": "Director Ejemplo",
                "directors_str": "Director Ejemplo",
                "cast_str": "Actor 1, Actor 2",
                "companies_str": "Estudio Ejemplo",
                "roi": 1.0,
                "popularity_category": "Media",
                "rating_category": "Regular",
                "popularity_level": "Media",
                "rating_level": "Regular",
                "is_profitable": True,
                "release_year": 2023,
                "budget_million": 1.0,
                "revenue_million": 2.0
            }])
            
            # Guardar archivo mínimo
            output_file = f"{OUTPUT_DIR}/processed_movies_v3_error_fallback.csv"
            df_min.to_csv(output_file, index=False)
            logger.warning(f"Creado archivo mínimo de fallback en {output_file}")
            
            # Pasar a la siguiente tarea
            ti = kwargs.get('ti')
            if ti:
                ti.xcom_push(key='processed_data_path', value=output_file)
                ti.xcom_push(key='movie_count', value=1)
                
            return 1
            
        except Exception as fallback_error:
            logger.error(f"Error al crear fallback: {fallback_error}")
            raise e  # Re-lanzar la excepción original

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
                            tmdb_id, title, overview, release_date,
                            popularity, vote_average, vote_count, budget, revenue,
                            runtime, adult, genres_str, directors_str, cast_str,
                            companies_str, roi, popularity_category, rating_category
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """

                        adult_value = row.get('adult', False)
                        if not isinstance(adult_value, bool):
                            adult_value = False

                        cur.execute(insert_query, (
                            row.get('id'),
                            row.get('title'),
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
                        
                        # Usar valor de release_year si ya existe
                        if 'release_year' in row and pd.notna(row['release_year']):
                            release_year = int(row['release_year'])

                        # Obtener género principal
                        genre = row.get('genre', 'Unknown')
                        if not genre and 'genres_str' in row and row['genres_str']:
                            genre = row['genres_str'].split(',')[0].strip()

                        # Obtener director principal
                        director = row.get('director', 'Unknown')
                        if not director and 'directors_str' in row and row['directors_str']:
                            director = row['directors_str'].split(',')[0].strip()

                        # Asegurar que tenemos los niveles de popularidad y calificación
                        popularity_level = row.get('popularity_level', row.get('popularity_category', 'Desconocida'))
                        rating_level = row.get('rating_level', row.get('rating_category', 'Desconocida'))

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
                            popularity_level,
                            rating_level,
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

# Función 5: Generar visualizaciones
def generate_visualizations(**kwargs):
    """Genera visualizaciones avanzadas a partir de los datos almacenados"""

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
                """,
                "correlacion_presupuesto_ingresos": """
                    SELECT title, budget/1000000 as budget_millions, revenue/1000000 as revenue_millions,
                        popularity, vote_average
                    FROM movies
                    WHERE budget > 0 AND revenue > 0
                    ORDER BY budget DESC
                    LIMIT 30;
                """,
                "distribucion_por_año": """
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
                "peliculas_mayor_impacto": """
                    SELECT title, (popularity * 0.6 + vote_average * 10 * 0.4) as impact_score,
                        popularity, vote_average, release_date
                    FROM movies
                    ORDER BY impact_score DESC
                    LIMIT 15;
                """
            }

            results = {}
            for name, query in queries.items():
                results[name] = pd.read_sql(query, conn)
                logger.info(f"Consulta '{name}' completada: {len(results[name])} filas")

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

            # Crear datos para correlación presupuesto-ingresos
            if 'budget' in df.columns and 'revenue' in df.columns:
                results["correlacion_presupuesto_ingresos"] = df[['title', 'budget', 'revenue', 'popularity', 'vote_average']].copy()
                results["correlacion_presupuesto_ingresos"]['budget_millions'] = results["correlacion_presupuesto_ingresos"]['budget'] / 1000000
                results["correlacion_presupuesto_ingresos"]['revenue_millions'] = results["correlacion_presupuesto_ingresos"]['revenue'] / 1000000
                results["correlacion_presupuesto_ingresos"] = results["correlacion_presupuesto_ingresos"].sort_values('budget', ascending=False).head(30)

            # Crear datos para películas de mayor impacto
            if 'popularity' in df.columns and 'vote_average' in df.columns:
                df['impact_score'] = df['popularity'] * 0.6 + df['vote_average'] * 10 * 0.4
                results["peliculas_mayor_impacto"] = df[['title', 'impact_score', 'popularity', 'vote_average', 'release_date']].sort_values('impact_score', ascending=False).head(15)

        # Generar visualizaciones
        visualizations_dir = f"{OUTPUT_DIR}/visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(visualizations_dir, exist_ok=True)

        # 1. Top películas por popularidad
        if "top_movies" in results and not results["top_movies"].empty:
            plt.figure(figsize=(14, 8))
            df = results["top_movies"]

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
            plt.savefig(f"{visualizations_dir}/top_peliculas_popularidad.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico 'Top 10 películas por popularidad' guardado")

        # 2. Distribución de géneros
        if "genre_distribution" in results and not results["genre_distribution"].empty:
            plt.figure(figsize=(14, 8))
            df = results["genre_distribution"]

            # Paleta de colores basada en la cantidad
            colors = plt.cm.plasma(np.linspace(0, 1, len(df)))

            ax = sns.barplot(x='count', y='genre', data=df, palette=colors)

            plt.title('Géneros Cinematográficos por Popularidad', fontsize=16, fontweight='bold')
            plt.xlabel('Número de Películas', fontsize=12)
            plt.ylabel('Género', fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{visualizations_dir}/generos_popularidad.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de distribución de géneros guardado")

            # Crear nube de palabras para géneros
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                            colormap='viridis', max_words=100)

                genre_freq = dict(zip(df['genre'], df['count']))
                wordcloud.generate_from_frequencies(genre_freq)

                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Nube de Palabras de Géneros Cinematográficos', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{visualizations_dir}/generos_nube.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Nube de palabras de géneros guardada")
            except Exception as e:
                logger.error(f"Error al crear nube de palabras: {e}")

        # 3. Correlación presupuesto-ingresos
        if "correlacion_presupuesto_ingresos" in results and not results["correlacion_presupuesto_ingresos"].empty:
            plt.figure(figsize=(14, 10))
            df = results["correlacion_presupuesto_ingresos"]

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

            plt.title('Relación entre Presupuesto e Ingresos de Películas', fontsize=16, fontweight='bold')
            plt.xlabel('Presupuesto (Millones $)', fontsize=12)
            plt.ylabel('Ingresos (Millones $)', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{visualizations_dir}/presupuesto_ingresos.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de correlación presupuesto-ingresos guardado")

        # 4. Películas de mayor impacto
        if "peliculas_mayor_impacto" in results and not results["peliculas_mayor_impacto"].empty:
            plt.figure(figsize=(14, 10))
            df = results["peliculas_mayor_impacto"].head(10)  # Top 10

            # Ordenar para que la mayor quede arriba
            df = df.sort_values('impact_score', ascending=True)

            # Calcular contribuciones
            popularity_contribution = df['popularity'] * 0.6
            rating_contribution = df['vote_average'] * 10 * 0.4

            # Crear barras apiladas
            x_pos = range(len(df))
            plt.barh(x_pos, popularity_contribution, color='#1976D2', alpha=0.8, label='Contribución de Popularidad')
            plt.barh(x_pos, rating_contribution, left=popularity_contribution, color='#FFC107', alpha=0.8,
                     label='Contribución de Calificación')

            # Añadir títulos de películas
            plt.yticks(x_pos, df['title'])

            plt.title('Top 10 Películas de Mayor Impacto', fontsize=16, fontweight='bold')
            plt.xlabel('Score de Impacto (Contribución de Popularidad + Calificación)', fontsize=12)
            plt.legend(loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{visualizations_dir}/peliculas_mayor_impacto.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de películas de mayor impacto guardado")

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

# Función 6: Entrenar modelo ML
# En tmdb_pipeline.py
def train_ml_model(**kwargs):
    """Entrena un modelo de Machine Learning avanzado para predecir el éxito de películas."""
    try:
        # Importar librerías necesarias aquí para asegurar que estén disponibles
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.impute import SimpleImputer
        import pickle
        import os
        import logging
        import json
        import traceback
        from datetime import datetime

        logger = logging.getLogger(__name__)

        # Configuración de directorios
        OUTPUT_DIR = "/opt/airflow/data/movie_analytics"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Intentar leer datos de PostgreSQL
        try:
            # Conexión a PostgreSQL
            import psycopg2
            # Configuración de PostgreSQL
            POSTGRES_DB = "postgres"
            POSTGRES_USER = "postgres"
            POSTGRES_PASSWORD = "Villeta-11"
            POSTGRES_HOST = "movie_postgres"
            POSTGRES_PORT = "5432"

            conn = psycopg2.connect(
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT
            )

            # Consultar datos para el entrenamiento desde data warehouse
            query = """
                SELECT 
                    tmdb_id, title, release_date, popularity, vote_average,
                    vote_count, budget, revenue, runtime, genre, director,
                    roi, popularity_level, rating_level, release_year
                FROM movie_data_warehouse
                WHERE vote_count > 10 AND budget > 0
            """
            df = pd.read_sql(query, conn)
            conn.close()
            logger.info(f"Datos leídos de PostgreSQL (movie_data_warehouse): {len(df)} películas")

        except Exception as e:
            logger.error(f"Error al leer datos de PostgreSQL para ML: {e}")
            logger.error(traceback.format_exc())

            # Si falla, usar datos de respaldo - buscar CSV en varias ubicaciones posibles
            processed_data_path = None
            possible_paths = [
                kwargs['ti'].xcom_pull(key='processed_data_path', task_ids='process_kafka_data'),
                f"{OUTPUT_DIR}/processed_movies_v3.csv",
                "/opt/airflow/data/processed_movies_v3.csv",
                "data/processed_movies_v3.csv"
            ]

            for path in possible_paths:
                if path and os.path.exists(path):
                    processed_data_path = path
                    break

            if not processed_data_path:
                logger.error("No se encontraron datos para entrenar modelo ML")
                return None

            # Cargar datos desde CSV
            df = pd.read_csv(processed_data_path)
            logger.info(f"Datos leídos de CSV: {len(df)} películas")

        if len(df) < 20:
            logger.warning("No hay suficientes datos para entrenar un modelo ML robusto")
            return None

        # Verificar datos cargados
        logger.info(f"Columnas disponibles: {df.columns.tolist()}")
        logger.info(f"Primeras filas:\n{df.head(2)}")

        # Preprocesamiento de datos
        logger.info("Iniciando preprocesamiento de datos...")

        # Limpiar y transformar datos
        df = df.copy()
        df = df.fillna(0)

        # Convertir a tipos apropiados
        numeric_cols = ['popularity', 'vote_average', 'vote_count', 'budget', 'revenue', 'runtime', 'roi']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Extraer año de lanzamiento si no existe
        if 'release_year' not in df.columns and 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            df['release_year'] = df['release_year'].fillna(2000).astype(int)

        # Asegurar que tenemos 'main_genre' como simplificación de 'genre'
        if 'main_genre' not in df.columns and 'genre' in df.columns:
            df['main_genre'] = df['genre']
        elif 'main_genre' not in df.columns and 'genres' in df.columns:
            # Si solo tenemos 'genres' como string con comas, extraer el primer género
            def extract_main_genre(genres):
                if pd.isna(genres) or not genres:
                    return 'Unknown'
                if isinstance(genres, list):
                    return genres[0] if genres else 'Unknown'
                if isinstance(genres, str):
                    return genres.split(',')[0].strip() if genres else 'Unknown'
                return 'Unknown'
            
            df['main_genre'] = df['genres'].apply(extract_main_genre)
        elif 'main_genre' not in df.columns:
            df['main_genre'] = 'Unknown'

        # Crear características adicionales para entrenamiento
        df['budget_million'] = df['budget'] / 1000000
        
        # Calcular ROI con manejo seguro de nulos/ceros
        df['roi'] = 0  # Valor por defecto
        mask = (df['budget'] > 0)
        df.loc[mask, 'roi'] = (df.loc[mask, 'revenue'] - df.loc[mask, 'budget']) / df.loc[mask, 'budget']
        
        # Definir el target: ROI > 3
        df['high_roi'] = (df['roi'] >= 3).astype(int)
        
        # Crear características relevantes pre-estreno
        # Número de películas del mismo género en los últimos 5 años
        df['release_year_group'] = (df['release_year'] // 5) * 5  # Agrupar por periodos de 5 años
        
        # Crear características derivadas del presupuesto
        df['budget_category'] = pd.cut(
            df['budget_million'], 
            bins=[0, 10, 30, 70, 100, 1000], 
            labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Blockbuster']
        ).astype(str)
        
        # Crear variables de temporada (épocas de estrenos importantes)
        if 'release_date' in df.columns:
            df['release_month'] = pd.to_datetime(df['release_date'], errors='coerce').dt.month
            # Temporadas:
            # Verano (mayo-agosto): 1
            # Navidad/premios (noviembre-diciembre): 2
            # Primavera (febrero-abril): 3
            # Otoño (septiembre-octubre, enero): 4
            season_map = {
                1: 4,  # Enero
                2: 3, 3: 3, 4: 3,  # Primavera
                5: 1, 6: 1, 7: 1, 8: 1,  # Verano
                9: 4, 10: 4,  # Otoño
                11: 2, 12: 2  # Navidad/premios
            }
            df['release_season'] = df['release_month'].map(season_map).fillna(4).astype(int)
        
        # Variables específicas para predicción de ROI
        # Ratio duración/presupuesto (calidad del tiempo en pantalla)
        df['runtime_budget_ratio'] = df['runtime'] / (df['budget_million'] + 1)
        
        # Duración óptima (películas entre 90-150 minutos suelen funcionar mejor)
        df['optimal_runtime'] = ((df['runtime'] >= 90) & (df['runtime'] <= 150)).astype(int)

        # Verificar y reportar la distribución del target
        roi_distribution = df['high_roi'].mean() * 100
        logger.info(f"Porcentaje de películas con ROI >= 3: {roi_distribution:.2f}%")

        # Seleccionar características para el modelo (solo las que se conocerían ANTES del estreno)
        features = [
            'budget_million',
            'runtime',
            'main_genre',
            'vote_average',  # Aunque esto podría no conocerse, es útil para películas similares
            'optimal_runtime',
            'runtime_budget_ratio',
            'release_season',
            'release_year'
        ]
        
        # Características categóricas
        categorical_features = ['main_genre', 'budget_category']
        for cat_feature in categorical_features:
            if cat_feature not in features and cat_feature in df.columns:
                features.append(cat_feature)

        logger.info(f"Características seleccionadas: {features}")
        logger.info(f"Características categóricas: {categorical_features}")

        # Filtrar datos con valores válidos
        df = df.dropna(subset=['high_roi'] + features)

        if len(df) < 20:
            logger.warning("Datos insuficientes después de filtrar")
            return None

        # Separar variables y target para la predicción de ROI alto
        X = df[features]
        y = df['high_roi']

        logger.info(f"Dataset final: {X.shape[0]} muestras, {X.shape[1]} características")

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Crear preprocesadores por tipo de característica
        transformers = []

        # Características numéricas
        numeric_features = [f for f in features if f not in categorical_features]
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))

        # Características categóricas
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        # Combinar preprocesadores
        preprocessor = ColumnTransformer(transformers=transformers)

        # Modelo XGBoost para clasificación (predecir ROI alto)
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=1,  # Ajustar si los datos están desbalanceados
            random_state=42
        )

        # Crear pipeline completo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb_model)
        ])

        # Entrenar el modelo
        logger.info("Entrenando modelo XGBoost para clasificación de alto ROI...")
        pipeline.fit(X_train, y_train)

        # Evaluar modelo en conjunto de prueba
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades
        y_pred = pipeline.predict(X_test)  # Clases predichas

        # Métricas de clasificación
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info(f"Rendimiento del modelo de clasificación:")
        logger.info(f"- Accuracy: {accuracy:.4f}")
        logger.info(f"- Precision: {precision:.4f}")
        logger.info(f"- Recall: {recall:.4f}")
        logger.info(f"- F1-score: {f1:.4f}")

        # Si el modelo básico es decente, intentar optimizar con RandomizedSearchCV
        if accuracy > 0.6:  # Solo optimizar si el modelo base no es terrible
            logger.info("Iniciando optimización de hiperparámetros con RandomizedSearchCV...")
            
            # Definir espacio de búsqueda para hiperparámetros
            param_distributions = {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [3, 4, 5, 6, 7],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__subsample': [0.6, 0.7, 0.8, 0.9],
                'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'model__min_child_weight': [1, 3, 5, 7],
                'model__gamma': [0, 0.1, 0.2],
                'model__scale_pos_weight': [1, 2, 3, 5]  # Para datos desbalanceados
            }
            
            search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_distributions,
                n_iter=20,
                cv=5, 
                scoring='f1',  # Optimizar para F1-score
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            try:
                search.fit(X_train, y_train)
                
                # Obtener el mejor modelo y sus hiperparámetros
                best_pipeline = search.best_estimator_
                best_params = search.best_params_
                
                # Evaluar el modelo optimizado
                y_pred_opt = best_pipeline.predict(X_test)
                accuracy_opt = accuracy_score(y_test, y_pred_opt)
                precision_opt = precision_score(y_test, y_pred_opt, zero_division=0)
                recall_opt = recall_score(y_test, y_pred_opt, zero_division=0)
                f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
                
                logger.info(f"Rendimiento del modelo optimizado:")
                logger.info(f"- Accuracy: {accuracy_opt:.4f} (mejora: {accuracy_opt-accuracy:.4f})")
                logger.info(f"- Precision: {precision_opt:.4f}")
                logger.info(f"- Recall: {recall_opt:.4f}")
                logger.info(f"- F1-score: {f1_opt:.4f} (mejora: {f1_opt-f1:.4f})")
                
                # Si el modelo optimizado es mejor, usarlo
                if f1_opt > f1:
                    pipeline = best_pipeline
                    accuracy, precision, recall, f1 = accuracy_opt, precision_opt, recall_opt, f1_opt
                    logger.info("Se utilizará el modelo optimizado")
                else:
                    logger.info("Se mantendrá el modelo base (mejor rendimiento)")
            
            except Exception as e:
                logger.error(f"Error durante la optimización de hiperparámetros: {e}")
                logger.error(traceback.format_exc())
                logger.info("Continuando con el modelo base")

        # Extraer información de importancia de características
        try:
            # Obtener el modelo del pipeline
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['model']
            
            # Preprocesar datos de entrenamiento para obtener nombres reales de características
            X_transformed = preprocessor.transform(X_train)
            
            # Obtener nombres de características después de transformación
            all_feature_names = []
            
            # Obtener nombres de características numéricas
            if numeric_features:
                all_feature_names.extend(numeric_features)
            
            # Obtener nombres de características categóricas transformadas
            if categorical_features:
                for i, feature_name in enumerate(categorical_features):
                    transformer = preprocessor.transformers_[len(numeric_features) + i][1]
                    encoder = transformer.named_steps['onehot']
                    categories = encoder.categories_[0]
                    for category in categories:
                        all_feature_names.append(f"{feature_name}_{category}")
            
            # Verificar que tenemos la cantidad correcta de nombres de características
            if len(all_feature_names) != X_transformed.shape[1]:
                logger.warning(f"Discrepancia en nombres de características: {len(all_feature_names)} vs {X_transformed.shape[1]}")
                # Ajustar la lista si hay discrepancia
                if len(all_feature_names) > X_transformed.shape[1]:
                    all_feature_names = all_feature_names[:X_transformed.shape[1]]
                else:
                    for i in range(len(all_feature_names), X_transformed.shape[1]):
                        all_feature_names.append(f"feature_{i}")
            
            # Obtener importancia de características directamente del modelo
            importances = model.feature_importances_
            
            # Crear DataFrame de importancia
            feature_importance = pd.DataFrame({
                'feature': all_feature_names,
                'importance': importances
            })
            
            # Normalizar importancias para que sumen 1
            feature_importance['importance'] = feature_importance['importance'] / feature_importance['importance'].sum()
            
            # Ordenar por importancia
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            logger.info("Top 10 características más importantes:")
            for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(10), 
                                                         feature_importance['importance'].head(10))):
                logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        except Exception as e:
            logger.error(f"Error al extraer importancia de características: {e}")
            logger.error(traceback.format_exc())
            
            # Crear importancia predeterminada si falla
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': [1/len(features)] * len(features)
            })

        # Guardar modelo y métricas en ubicaciones clave
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Directorios donde guardar el modelo
        model_dirs = [
            f"{OUTPUT_DIR}/xgb_model_{timestamp}",
            f"{OUTPUT_DIR}/xgb_model_latest",
            "/opt/airflow/data/movie_analytics/xgb_model_latest",
            "/opt/airflow/data/latest_xgboost_model",
            "/opt/airflow/data/public_models"
        ]

        # Crear todos los directorios y guardar modelo en cada uno
        for dir_path in model_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)

                # Guardar modelo
                model_path = os.path.join(dir_path, "model_final.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(pipeline, f)
                os.chmod(model_path, 0o777)  # Permisos amplios

                # También guardar como model.pkl
                alt_model_path = os.path.join(dir_path, "model.pkl")
                with open(alt_model_path, 'wb') as f:
                    pickle.dump(pipeline, f)
                os.chmod(alt_model_path, 0o777)

                # Guardar métricas
                metrics = {
                    'model_type': 'XGBoost Classifier',
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'feature_importance': feature_importance.to_dict('records'),
                    'timestamp': timestamp,
                    'samples_count': len(X),
                    'features_used': features,
                    'target': 'high_roi (ROI >= 3)',
                    'positive_class_ratio': float(y.mean())
                }

                metrics_path = os.path.join(dir_path, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                os.chmod(metrics_path, 0o777)

                # Guardar feature importance como CSV
                feature_path = os.path.join(dir_path, "feature_importance.csv")
                feature_importance.to_csv(feature_path, index=False)
                os.chmod(feature_path, 0o777)

                logger.info(f"Modelo guardado en: {dir_path}")
            except Exception as e:
                logger.error(f"Error al guardar modelo en {dir_path}: {e}")

        # Crear archivos de señalización
        try:
            # Archivo con la ubicación del modelo
            location_paths = [
                f"{OUTPUT_DIR}/model_location.txt",
                "/opt/airflow/data/model_location.txt",
                "model_location.txt"
            ]

            location_content = (
                "/opt/airflow/data/latest_xgboost_model/model_final.pkl\n"
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "xgboost\n"
                "/opt/airflow/data/public_models/model_final.pkl\n"
            )

            for path in location_paths:
                with open(path, 'w') as f:
                    f.write(location_content)
                os.chmod(path, 0o777)
                logger.info(f"Archivo de ubicación creado: {path}")

            # Archivo ready.txt
            ready_paths = [
                f"{OUTPUT_DIR}/model_ready.txt",
                "/opt/airflow/data/model_ready.txt"
            ]

            for path in ready_paths:
                with open(path, 'w') as f:
                    f.write("yes")
                os.chmod(path, 0o777)
                logger.info(f"Archivo de señalización creado: {path}")

        except Exception as e:
            logger.error(f"Error al crear archivos de señalización: {e}")

        # Pasar información a la siguiente tarea
        model_dir = model_dirs[0]  # Usar el directorio con timestamp
        kwargs['ti'].xcom_push(key='model_dir', value=model_dir)
        kwargs['ti'].xcom_push(key='model_metrics', value=metrics)

        logger.info(f"Modelo entrenado exitosamente: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        return model_dir

    except Exception as e:
        logger.error(f"Error global en train_ml_model: {e}")
        logger.error(traceback.format_exc())
        raise

# Función para generar visualizaciones del modelo
def generate_model_visualizations(model, feature_importance, X_test, y_test, y_pred, output_dir):
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

# Definir default_args para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Esta función verificará los modelos disponibles y registrará sus rutas
def verify_model_ready(**kwargs):
    """Verify that model is ready and accessible for Streamlit"""
    import os
    import glob
    import json
    import logging
    import traceback

    logger = logging.getLogger(__name__)

    # Verificar archivo de señalización
    ready_file = "data/model_ready.txt"
    if os.path.exists(ready_file):
        try:
            with open(ready_file, 'r') as f:
                content = f.read().strip()
            if content.lower() == "yes":
                logger.info("Model readiness signal confirmed")
            else:
                logger.warning(f"Unexpected content in model_ready.txt: {content}")
        except Exception as e:
            logger.error(f"Error reading model_ready.txt: {e}")
    else:
        logger.warning("model_ready.txt not found")

    # Verificar la ubicación estándar (la más importante)
    standard_location = "/opt/airflow/data/latest_xgboost_model/model.pkl"
    if os.path.exists(standard_location):
        logger.info(f"Model found at standard location: {standard_location}")
        return True
    else:
        logger.warning(f"Model not found at standard location: {standard_location}")

    # Buscar cualquier archivo de modelo (opcional, para diagnóstico)
    model_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pkl"):
                model_files.append(os.path.join(root, file))

    if model_files:
        logger.info(f"Found {len(model_files)} model files (for diagnostic purposes):")
        for file in model_files[:5]:  # Mostrar solo los primeros 5
            logger.info(f"- {file}")
    else:
        logger.warning("No model files found anywhere!")

    return False

with DAG(
    'tmdb_pipeline_streamlit',
    default_args=default_args,
    description='Pipeline de datos TMDB con interfaz Streamlit y modelos ML avanzados',
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
        bash_command='echo "Iniciando pipeline de datos de TMDB con Streamlit y modelos ML avanzados" && mkdir -p {{ params.output_dir }}',
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

    # Tarea 7: Almacenar el modelo ML (usando nuestra nueva versión de la función)
    store_model_task = PythonOperator(
        task_id='store_ml_model',
        python_callable=store_ml_model,  # La nueva función mejorada
        provide_context=True
    )

    # Tarea 8: Verificar que el modelo esté disponible para Streamlit
    verify_model_task = PythonOperator(
        task_id='verify_model_availability',
        python_callable=verify_model_ready,  # La nueva función mejorada
        provide_context=True
    )

    # Tarea 10: Instalar dependencias de Streamlit
    install_deps_task = BashOperator(
        task_id='install_streamlit_deps',
        bash_command='pip install streamlit plotly matplotlib seaborn psycopg2-binary scikit-learn wordcloud xgboost',
    )

    # Tarea 11: Iniciar la aplicación Streamlit
    streamlit_task = StreamlitOperator(
        task_id='start_streamlit_app',
        script_path=STREAMLIT_PATH,
        port=8502,  # Cambiado a 8502 para evitar conflictos
        host="0.0.0.0"
    )

    # Tarea 12: Finalizar el pipeline
    end_task = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "Pipeline de datos de TMDB con Streamlit y modelos ML avanzados completado con éxito a las $(date)"'
    )

# Definir el flujo de tareas
setup_task >> start_task >> fetch_task >> process_task >> load_task >> visualization_task >> ml_task >> store_model_task >> verify_model_task >> install_deps_task >> end_task >> streamlit_task