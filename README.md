# TMDB_BIG_DATA

# Pipeline de Análisis de Películas con TMDB, Kafka, Airflow y PostgreSQL

## Descripción General del Proyecto

Este proyecto implementa un pipeline de procesamiento de datos de películas completamente automatizado y escalable, utilizando tecnologías de big data y flujo de trabajo modernas. El objetivo principal es extraer, transformar, cargar (ETL) y analizar datos de películas desde la API de The Movie Database (TMDB), procesarlos en tiempo real utilizando Kafka, orquestar el flujo de trabajo con Apache Airflow y almacenar los resultados en una base de datos PostgreSQL.

## Componentes Arquitecturales

### 1. Extracción de Datos
- **Fuente de Datos**: API de TMDB
- **Método de Extracción**: Productor de Kafka (`producer.py`)
- **Frecuencia**: Cada 12 horas
- **Datos Extraídos**: Películas populares con detalles completos

### 2. Procesamiento de Datos
- **Tecnologías**: 
  - Kafka para streaming de datos
  - Pandas para transformación
  - Dask para procesamiento paralelo (opcional)
- **Procesos**:
  - Limpieza de datos
  - Enriquecimiento de información
  - Cálculo de métricas (ROI, categorías de popularidad)

### 3. Almacenamiento
- **Base de Datos**: PostgreSQL
- **Esquema**:
  - Tabla `movies`: Información principal de películas
  - Tabla `genres`: Catálogo de géneros
  - Tabla `directors`: Catálogo de directores
  - Tabla `movie_data_warehouse`: Tabla analítica para reportes

### 4. Análisis y Machine Learning
- **Modelo Predictivo**: Regresión Lineal
- **Objetivo**: Predecir popularidad de películas
- **Métricas**: MSE, R²
- **Visualizaciones**: Gráficos de distribución, importancia de características

### 5. Orquestación
- **Herramienta**: Apache Airflow
- **Flujo de Trabajo**:
  1. Preparación del entorno
  2. Extracción de datos
  3. Procesamiento
  4. Carga en PostgreSQL
  5. Generación de visualizaciones
  6. Entrenamiento de modelo ML

## Tecnologías Utilizadas

- Python
- Apache Kafka
- Apache Airflow
- PostgreSQL
- Pandas
- Scikit-learn
- Docker
- Docker Compose

## Requisitos Previos

- Docker
- Docker Compose
- Python 3.8+
- Cuenta en TMDB para obtener API Key

## Estructura de Archivos

```
movie_analytics/
│
├── docker-compose.yaml         # Configuración de servicios Docker
├── dags/
│   └── dag_tmdb.py             # DAG de Airflow
├── data/
│   ├── producer.py              # Productor de datos Kafka
│   └── consumer.py              # Consumidor de datos Kafka
├── logs/                        # Logs de Airflow
├── plugins/                     # Plugins de Airflow
└── README.md                    # Documentación del proyecto
```

## Pasos de Implementación

1. Clonar repositorio
2. Configurar variables de entorno
3. Iniciar servicios Docker
4. Ejecutar pipeline de Airflow

## Contribuciones

Las contribuciones son bienvenidas. Por favor, consulta las guías de contribución antes de realizar un pull request.

## Licencia

[Especificar la licencia del proyecto]
```

## Pasos de Instalación en MacBook Pro

Aquí te detallo los comandos específicos para tu MacBook Pro:

### 1. Instalar Dependencias Previas

```bash
# Instalar Homebrew (si no lo tienes)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Docker Desktop
brew install --cask docker

# Instalar Python (si no lo tienes)
brew install python

# Instalar dependencias de Python
pip3 install apache-airflow kafka-python pandas scikit-learn psycopg2-binary requests sqlalchemy matplotlib seaborn
```

### 2. Configurar Proyecto

```bash
# Crear directorio del proyecto
mkdir movie_analytics
cd movie_analytics

# Clonar o crear archivos del proyecto
touch docker-compose.yaml
mkdir -p dags data logs plugins config
```

### 3. Iniciar Servicios Docker

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs de servicios
docker-compose logs -f
```

### 4. Configurar Airflow

```bash
# Crear usuario admin en Airflow
docker-compose exec airflow-webserver airflow users create \
    --username "admin" \
    --firstname "Gabriela" \
    --lastname "Morera" \
    --role "Admin" \
    --email "moreragabriela1@gmail.com" \
    --password "tuContraseña"
```

### 5. Ejecutar Pipeline

```bash
# Acceder al webserver de Airflow
open http://localhost:8080

# Activar DAG manualmente
docker-compose exec airflow-webserver airflow dags trigger tmdb_pipeline_v2
```

### 6. Conectarte a PostgreSQL desde la terminal

```bash
docker-compose exec movie_postgres psql -U postgres -d postgres

# Una vez dentro, listar las tablas
\dt

# Ver el contenido de una tabla específica
SELECT * FROM movies LIMIT 5;

# Ver el contenido de la tabla de data warehouse
SELECT * FROM movie_data_warehouse LIMIT 5;

# Salir de PostgreSQL
\q
```

## Consideraciones Finales

- Reemplaza `tuContraseña` con una contraseña segura
- Verifica que todos los archivos (`producer.py`, `consumer.py`, `dag_tmdb.py`) estén correctamente configurados
- Mantén tus credenciales de API seguras
- Monitorea los recursos de tu máquina durante la ejecución

## Solución de Problemas

- Si encuentras errores de conexión, verifica la configuración de red
- Revisa los logs de Docker y Airflow para diagnóstico
- Asegúrate de tener suficiente espacio en disco
