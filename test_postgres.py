import psycopg2

# Conexión a PostgreSQL
try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="Villeta-11",
        host="movie_postgres",
        port="5433"  # Cambiado a 5433
    )
    
    cur = conn.cursor()
    
    # Probar conexión
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print(f"Conexión exitosa: {version[0]}")
    
    # Listar tablas existentes
    cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public';")
    tables = cur.fetchall()
    
    print("Tablas existentes:")
    if tables:
        for table in tables:
            print(f"- {table[0]}")
    else:
        print("No hay tablas en el esquema público.")
    
    # Crear una tabla de prueba
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        print("Tabla de prueba creada correctamente")
    except Exception as e:
        print(f"Error al crear tabla: {e}")
        conn.rollback()
    
    # Cerrar conexión
    cur.close()
    conn.close()

except Exception as e:
    print(f"Error al conectar: {e}")