#!/bin/bash

echo "Configurando entorno para el proyecto Big Data Final..."

# Verificar Docker
echo "Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

# Verificar Docker Compose
echo "Verificando Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose no está instalado. Por favor instala Docker Compose primero."
    exit 1
fi

# Crear estructura de directorios
echo "Creando estructura de directorios..."
mkdir -p dags logs config plugins data data/movie_analytics

# Verificar si los archivos existen
echo "Verificando archivos necesarios..."
if [ ! -f "docker-compose.yaml" ]; then
    echo "ERROR: No se encontró docker-compose.yaml"
    exit 1
fi

if [ ! -f "data/producer.py" ]; then
    echo "ERROR: No se encontró data/producer.py"
    exit 1
fi

if [ ! -f "data/consumer.py" ]; then
    echo "ERROR: No se encontró data/consumer.py"
    exit 1
fi

if [ ! -f "dags/tmdb_pipeline.py" ]; then
    echo "ERROR: No se encontró dags/tmdb_pipeline.py"
    exit 1
fi

# Detener contenedores si existen
echo "Deteniendo contenedores existentes..."
docker-compose down

# Iniciar servicios
echo "Iniciando servicios con Docker Compose..."
docker-compose up -d

# Esperar a que los servicios estén listos
echo "Esperando a que los servicios estén listos..."
sleep 30

# Verificar que PostgreSQL está funcionando
echo "Verificando conexión a PostgreSQL..."
docker-compose exec movie_postgres psql -U postgres -c "SELECT version();"

# Verificar que Kafka está funcionando
echo "Verificando Kafka..."
docker-compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

# Crear tema Kafka si no existe
echo "Creando tema Kafka 'tmdb_data'..."
docker-compose exec kafka kafka-topics.sh --create --topic tmdb_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --if-not-exists

# Instalar dependencias en los contenedores de Airflow
echo "Instalando dependencias en contenedores de Airflow..."
docker-compose exec airflow-webserver pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python
docker-compose exec airflow-scheduler pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python
docker-compose exec airflow-worker pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python

echo "Configuración completada."
echo "Para acceder a Airflow, visita: http://localhost:8080"
echo "Usuario por defecto: admin"
echo "Contraseña por defecto: admin"
echo ""
echo "Para ejecutar el productor de datos, usa: python data/producer.py"
echo "Para ejecutar el consumidor de datos, usa: python data/consumer.py"
echo ""
echo "Para detener todos los servicios, usa: docker-compose down"