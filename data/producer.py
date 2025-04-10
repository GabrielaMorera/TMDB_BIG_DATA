import json
import time
import logging
import requests
import os
from kafka import KafkaProducer
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tmdb_producer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de Kafka
KAFKA_TOPIC = "tmdb_data"
KAFKA_BROKER = "localhost:9092"  # Para ejecutar fuera de Docker

# Configuración de la API TMDB
API_KEY = "e8e1dae84a0345bd3ec23e3030905258"
TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlOGUxZGFlODRhMDM0NWJkM2VjMjNlMzAzMDkwNTI1OCIsIm5iZiI6MTc0MTkxMzEzNi4xMjEsInN1YiI6IjY3ZDM3YzMwYmY0ODE4ODU0YzY0ZTVmNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tdq930_qLqYbYAqwVLl3Tdw84HdEsZtM41CX_9-lJNU"
BASE_URL = "https://api.themoviedb.org/3"

# Carpeta para guardar datos localmente
OUTPUT_DIR = "data/movie_analytics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TMDBProducer:
    def __init__(self):
        """Inicializa el productor de Kafka y configura la conexión."""
        logger.info("Inicializando productor TMDB...")
        
        # Intentar crear el productor de Kafka con reintentos
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BROKER,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    max_block_ms=10000,
                    retries=3
                )
                logger.info(f"Productor Kafka conectado a {KAFKA_BROKER}")
                break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Intento {retry_count}/{max_retries} - Error al conectar con Kafka: {e}")
                
                if retry_count >= max_retries:
                    logger.error("No se pudo conectar a Kafka después de varios intentos.")
                    raise
                
                time.sleep(5)  # Esperar antes del próximo intento
    
    def get_popular_movies(self, page=1):
        """Obtiene películas populares desde TMDB."""
        url = f"{BASE_URL}/movie/popular"
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json;charset=utf-8"
        }
        params = {
            "language": "es-ES",
            "page": page
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener datos de TMDB (página {page}): {e}")
            return []
    
    def get_movie_details(self, movie_id):
        """Obtiene detalles adicionales de una película por su ID."""
        url = f"{BASE_URL}/movie/{movie_id}"
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json;charset=utf-8"
        }
        params = {
            "language": "es-ES",
            "append_to_response": "credits,keywords"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener detalles de la película {movie_id}: {e}")
            return {}
    
    def send_movies_to_kafka(self, num_pages=5):
        """Envía datos de películas a Kafka, procesando varias páginas."""
        sent_count = 0
        error_count = 0
        
        try:
            for page in range(1, num_pages + 1):
                logger.info(f"Obteniendo películas populares - página {page}")
                movies = self.get_popular_movies(page)
                
                if not movies:
                    logger.warning(f"No se encontraron películas en la página {page}")
                    continue
                
                logger.info(f"Obtenidas {len(movies)} películas de la página {page}")
                
                for movie in movies:
                    movie_id = movie.get("id")
                    
                    # Enriquecer con detalles adicionales
                    if movie_id:
                        logger.info(f"Obteniendo detalles de película: {movie.get('title', 'Desconocido')} (ID: {movie_id})")
                        details = self.get_movie_details(movie_id)
                        
                        if details:
                            # Enriquecer el objeto de película con detalles
                            movie.update({
                                "budget": details.get("budget"),
                                "revenue": details.get("revenue"),
                                "runtime": details.get("runtime"),
                                "genres": [genre.get("name") for genre in details.get("genres", [])],
                                "production_companies": [company.get("name") for company in details.get("production_companies", [])],
                                "directors": [person.get("name") for person in details.get("credits", {}).get("crew", []) 
                                            if person.get("job") == "Director"],
                                "cast": [person.get("name") for person in details.get("credits", {}).get("cast", [])[:5]]  # Los 5 actores principales
                            })
                    
                    try:
                        # Enviar a Kafka
                        future = self.producer.send(KAFKA_TOPIC, value=movie)
                        # Esperar resultado y manejar excepciones
                        record_metadata = future.get(timeout=10)
                        logger.info(f"Enviado a Kafka: {movie.get('title')} (Partición: {record_metadata.partition}, Offset: {record_metadata.offset})")
                        sent_count += 1
                    except Exception as e:
                        logger.error(f"Error al enviar a Kafka: {e}")
                        error_count += 1
                    
                    # Guardar una copia local
                    self.save_movie_locally(movie)
                    
                    # Pequeña pausa para no sobrecargar la API
                    time.sleep(1)
                
                # Pausa entre páginas
                if page < num_pages:
                    logger.info(f"Esperando antes de procesar la siguiente página...")
                    time.sleep(3)
            
            # Asegurar que todos los mensajes se envíen
            self.producer.flush()
            logger.info(f"Proceso completado: {sent_count} películas enviadas, {error_count} errores")
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Error general en el proceso: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
        finally:
            # Cerrar productor
            self.producer.close()
            logger.info("Productor cerrado")
    
    def save_movie_locally(self, movie):
        """Guarda una copia de la película localmente como respaldo."""
        try:
            movie_id = movie.get("id")
            file_path = f"{OUTPUT_DIR}/movie_{movie_id}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(movie, f, indent=4, ensure_ascii=False)
                
            logger.debug(f"Guardada copia local de {movie.get('title')} en {file_path}")
        except Exception as e:
            logger.error(f"Error al guardar copia local: {e}")

def main():
    """Función principal para ejecutar el productor."""
    try:
        logger.info("=== Iniciando productor de datos TMDB ===")
        producer = TMDBProducer()
        
        # Número de páginas a procesar (cada página tiene ~20 películas)
        num_pages = 3
        
        # Enviar películas a Kafka
        sent_count = producer.send_movies_to_kafka(num_pages)
        
        logger.info(f"Proceso terminado. Se enviaron {sent_count} películas a Kafka.")
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()