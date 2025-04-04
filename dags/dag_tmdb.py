from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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

# Agregar rutas para que Airflow encuentre los módulos
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/data')

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

# Crear directorio para datos
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Función 2: Consumir datos de Kafka y procesarlos (usando Pandas en lugar de Dask)
def process_kafka_data(**kwargs):
    """Consume datos de Kafka y los procesa con Pandas en lugar de Dask"""
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
            group_id='airflow_consumer_group_v2',  # Cambiado para evitar conflictos
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
        output_file = f"{OUTPUT_DIR}/processed_movies_v2.csv"
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
        
        # Procesar con Pandas (en vez de Dask)
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
        output_file = f"{OUTPUT_DIR}/processed_movies_v2.csv"
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
        processed_data_path = f"{OUTPUT_DIR}/processed_movies_v2.csv"
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
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            results_file = f"{OUTPUT_DIR}/results_data_v2.csv"
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
                processed_data_path = f"{OUTPUT_DIR}/processed_movies_v2.csv"
            
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
        visualizations_dir = f"{OUTPUT_DIR}/visualizations_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            f.write("ANÁLISIS DE PELÍCULAS (Versión 2)\n")
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
                processed_data_path = f"{OUTPUT_DIR}/processed_movies_v2.csv"
            
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
        model_dir = f"{OUTPUT_DIR}/ml_model_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

# Definir el DAG
default_args = {
    'owner': 'gabriela',
    'start_date': days_ago(1),
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False
}

with DAG(
    'tmdb_pipeline_v2',  # Nombre actualizado del DAG
    default_args=default_args,
    description='Pipeline mejorado para datos de películas de TMDB (Versión 2)',
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
        bash_command='echo "Iniciando pipeline v2 de datos de TMDB" && mkdir -p {{ params.output_dir }}',
        params={'output_dir': OUTPUT_DIR}
    )
    
    # Tarea 2: Obtener datos de TMDB y enviarlos a Kafka
    fetch_task = PythonOperator(
        task_id='fetch_and_send_to_kafka',
        python_callable=fetch_and_send_to_kafka,
        provide_context=True
    )
    
    # Tarea 3: Consumir y procesar datos de Kafka (usando Pandas en vez de Dask)
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
    
    # Tarea 7: Finalizar el pipeline
    end_task = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "Pipeline v2 de datos de TMDB completado con éxito a las $(date)"'
    )
    
    # Definir el flujo de tareas
    setup_task >> start_task >> fetch_task >> process_task >> load_task >> visualization_task >> ml_task >> end_task