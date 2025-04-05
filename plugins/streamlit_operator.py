import os
import subprocess
import logging
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults

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