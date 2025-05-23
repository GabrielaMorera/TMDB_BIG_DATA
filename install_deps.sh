#!/bin/bash
echo "Instalando dependencias en airflow-webserver..."
docker-compose exec airflow-webserver pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python xgboost shap joblib plotly streamlit concurrent-futures urllib3 sqlalchemy

echo "Instalando dependencias en airflow-scheduler..."
docker-compose exec airflow-scheduler pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python xgboost shap joblib plotly streamlit concurrent-futures urllib3 sqlalchemy

echo "Instalando dependencias en airflow-worker..."
docker-compose exec airflow-worker pip install dask[complete] distributed scikit-learn matplotlib seaborn pandas psycopg2-binary kafka-python xgboost shap joblib plotly streamlit concurrent-futures urllib3 sqlalchemy

echo "Instalaciones completadas."