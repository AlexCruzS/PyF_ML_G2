#!/usr/bin/env python3
"""
🏠 PyF_ML_G2 - Sistema MLOps para Predicción de Precios Inmobiliarios

Punto de entrada principal que orquesta todo el sistema:
- Auto-verificación de dependencias
- Setup automático de base de datos
- Auto-entrenamiento de modelos (primera vez)
- API Server (FastAPI)
- Dashboard (Streamlit) 
- MLflow UI
- Gestión del ciclo de vida completo

Uso:
    python main.py                    # Iniciar sistema completo
    python main.py --setup-only       # Solo configurar entorno
    python main.py --train-only       # Solo entrenar modelos
    python main.py --api-only         # Solo API Server
    python main.py --dashboard-only   # Solo Dashboard

Autor: Equipo PyF_ML_G2
Versión: 1.0.0
"""

import asyncio
import os
import subprocess
import sys
import time
import webbrowser
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pyf_ml_g2.log')
    ]
)
logger = logging.getLogger(__name__)

class PyFMLG2Orchestrator:
    """
    Orquestador principal del sistema PyF_ML_G2
    
    Gestiona el ciclo de vida completo del sistema MLOps:
    - Verificación de dependencias
    - Configuración de entorno
    - Entrenamiento de modelos
    - Inicio de servicios
    - Monitoreo de salud del sistema
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.env_file = self.base_dir / ".env"
        self.processes: List[subprocess.Popen] = []
        self.config = self._load_config()
        
        # URLs del sistema
        self.urls = {
            "api": f"http://localhost:{self.config['API_PORT']}",
            "dashboard": f"http://localhost:{self.config['DASHBOARD_PORT']}",
            "mlflow": f"http://localhost:{self.config['MLFLOW_PORT']}",
            "docs": f"http://localhost:{self.config['API_PORT']}/docs",
            "health": f"http://localhost:{self.config['API_PORT']}/health"
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde variables de entorno"""
        from dotenv import load_dotenv
        
        # Cargar .env si existe
        if self.env_file.exists():
            load_dotenv(self.env_file)
        
        return {
            "API_PORT": int(os.getenv("API_PORT", "8000")),
            "DASHBOARD_PORT": int(os.getenv("DASHBOARD_PORT", "8501")),
            "MLFLOW_PORT": int(os.getenv("MLFLOW_PORT", "5000")),
            "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///pyf_ml_g2.db"),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
            "DEBUG": os.getenv("DEBUG", "true").lower() == "true"
        }
    
    def check_python_version(self) -> bool:
        """Verificar versión de Python"""
        logger.info("🐍 Verificando versión de Python...")
        
        if sys.version_info < (3, 9):
            logger.error("❌ Python 3.9+ es requerido")
            logger.error(f"Versión actual: {sys.version}")
            return False
        
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def check_dependencies(self) -> bool:
        """Verificar que las dependencias críticas estén instaladas"""
        logger.info("🔍 Verificando dependencias críticas...")
        
        critical_packages = [
            "fastapi", "streamlit", "mlflow", 
            "pandas", "scikit-learn", "uvicorn",
            "pydantic", "python-dotenv"
        ]
        
        missing_packages = []
        for package in critical_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.debug(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package}")
        
        if missing_packages:
            logger.error(f"❌ Dependencias faltantes: {', '.join(missing_packages)}")
            logger.info("💡 Ejecuta: pip install -r requirements.txt")
            return False
        
        logger.info("✅ Todas las dependencias críticas están instaladas")
        return True
    
    def setup_environment(self) -> bool:
        """Configurar entorno y archivos necesarios"""
        logger.info("🔧 Configurando entorno...")
        
        try:
            # Crear archivo .env si no existe
            if not self.env_file.exists():
                logger.info("📝 Creando archivo .env...")
                self._create_env_file()
            
            # Crear directorios necesarios
            directories = [
                "data/raw", "data/processed", "data/models",
                "logs", "mlruns", "temp"
            ]
            
            for dir_path in directories:
                full_path = self.base_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"📁 {dir_path}")
            
            logger.info("✅ Entorno configurado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando entorno: {e}")
            return False
    
    def _create_env_file(self):
        """Crear archivo .env con configuración por defecto"""
        env_content = """# PyF_ML_G2 - Configuración de Entorno

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Dashboard Configuration  
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_PORT=5000
MLFLOW_EXPERIMENT_NAME=PyF_ML_G2_Property_Valuation

# Database Configuration (SQLite para desarrollo)
DATABASE_URL=sqlite:///pyf_ml_g2.db

# Development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Data Paths
DATA_PATH=./data
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
MODELS_PATH=./data/models

# Model Configuration
MODEL_NAME=property_valuation_model
MODEL_VERSION=latest
"""
        
        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info("✅ Archivo .env creado con configuración por defecto")
    
    def check_models_exist(self) -> bool:
        """Verificar si existen modelos entrenados"""
        models_dir = self.base_dir / "data" / "models"
        mlruns_dir = self.base_dir / "mlruns"
        
        # Verificar archivos de modelo
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        
        # Verificar experimentos MLflow
        has_mlflow_runs = mlruns_dir.exists() and any(mlruns_dir.iterdir())
        
        exists = len(model_files) > 0 or has_mlflow_runs
        
        if exists:
            logger.info(f"✅ Modelos encontrados: {len(model_files)} archivos, MLflow: {has_mlflow_runs}")
        else:
            logger.info("ℹ️ No se encontraron modelos entrenados")
        
        return exists
    
    def train_initial_models(self) -> bool:
        """Entrenar modelos iniciales si no existen"""
        if self.check_models_exist():
            logger.info("🤖 Modelos existentes encontrados, omitiendo entrenamiento")
            return True
        
        logger.info("🤖 Iniciando entrenamiento de modelos iniciales...")
        
        try:
            # Por ahora, crear un modelo dummy para que el sistema funcione
            self._create_dummy_model()
            logger.info("✅ Modelo inicial creado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelos: {e}")
            logger.info("💡 Continuando sin modelos (modo desarrollo)")
            return True
    
    def _create_dummy_model(self):
        """Crear un modelo dummy para desarrollo"""
        import pickle
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        # Generar datos dummy
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        
        # Entrenar modelo simple
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Guardar modelo
        models_dir = self.base_dir / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "dummy_property_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"✅ Modelo dummy creado en: {model_path}")
    
    def start_mlflow_server(self) -> Optional[subprocess.Popen]:
        """Iniciar servidor MLflow"""
        logger.info("🔬 Iniciando MLflow server...")
        
        try:
            cmd = [
                sys.executable, "-m", "mlflow", "server",
                "--host", "127.0.0.1",
                "--port", str(self.config['MLFLOW_PORT']),
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--default-artifact-root", "./mlruns"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Esperar que el servidor inicie
            time.sleep(5)
            
            if process.poll() is None:
                logger.info(f"✅ MLflow server iniciado en {self.urls['mlflow']}")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Error iniciando MLflow: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error iniciando MLflow: {e}")
            return None
    
    def start_api_server(self) -> Optional[subprocess.Popen]:
        """Iniciar servidor FastAPI"""
        logger.info("🚀 Iniciando API server...")
        
        try:
            # Verificar si el módulo API existe
            api_module = "pyf_ml_g2.api.main:app"
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                api_module,
                "--host", "0.0.0.0",
                "--port", str(self.config['API_PORT']),
                "--reload" if self.config['DEBUG'] else "--no-reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Esperar que el servidor inicie
            time.sleep(5)
            
            if process.poll() is None:
                logger.info(f"✅ API server iniciado en {self.urls['api']}")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.warning(f"⚠️ API server no disponible aún: {stderr.decode()[:200]}")
                logger.info("💡 Continuando sin API (se iniciará cuando esté implementada)")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error iniciando API: {e}")
            return None
    
    def start_dashboard(self) -> Optional[subprocess.Popen]:
        """Iniciar dashboard Streamlit"""
        logger.info("📊 Iniciando dashboard...")
        
        try:
            dashboard_file = self.base_dir / "pyf_ml_g2" / "monitoring" / "dashboard.py"
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_file),
                "--server.port", str(self.config['DASHBOARD_PORT']),
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false",
                "--server.headless", "true"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Esperar que el servidor inicie
            time.sleep(8)
            
            if process.poll() is None:
                logger.info(f"✅ Dashboard iniciado en {self.urls['dashboard']}")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.warning(f"⚠️ Dashboard no disponible aún: {stderr.decode()[:200]}")
                logger.info("💡 Continuando sin dashboard (se iniciará cuando esté implementado)")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error iniciando dashboard: {e}")
            return None
    
    def open_browsers(self):
        """Abrir navegadores con las URLs principales"""
        if self.config['ENVIRONMENT'] == 'development':
            logger.info("🌐 Abriendo navegadores...")
            
            time.sleep(3)  # Esperar que los servicios estén listos
            
            try:
                # Abrir MLflow primero
                webbrowser.open(self.urls["mlflow"])
                time.sleep(1)
                
                # Abrir dashboard si está disponible
                if any(p for p in self.processes if p and p.poll() is None):
                    webbrowser.open(self.urls["dashboard"])
                    time.sleep(1)
                
                logger.info("✅ Navegadores abiertos")
                
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron abrir los navegadores: {e}")
    
    def print_system_info(self):
        """Mostrar información del sistema"""
        running_services = len([p for p in self.processes if p and p.poll() is None])
        
        print("\n" + "="*70)
        print("🏠 PyF_ML_G2 - Sistema MLOps para Predicción de Precios Inmobiliarios")
        print("="*70)
        print(f"📊 Dashboard Principal:    {self.urls['dashboard']}")
        print(f"🚀 API Documentation:     {self.urls['docs']}")
        print(f"🔬 MLflow Experiments:    {self.urls['mlflow']}")
        print(f"🏥 Health Check:          {self.urls['health']}")
        print("="*70)
        print(f"🔄 Servicios ejecutándose: {running_services}")
        print(f"🌍 Entorno: {self.config['ENVIRONMENT']}")
        print(f"📝 Logs: pyf_ml_g2.log")
        print("="*70)
        print("💡 Presiona Ctrl+C para detener todos los servicios")
        print("="*70)
    
    def cleanup(self):
        """Limpiar procesos al salir"""
        logger.info("🛑 Deteniendo servicios...")
        
        for i, process in enumerate(self.processes):
            if process and process.poll() is None:
                try:
                    if os.name == 'nt':  # Windows
                        process.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                    else:  # Unix/Linux
                        process.terminate()
                    
                    process.wait(timeout=5)
                    logger.info(f"✅ Servicio {i+1} detenido")
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning(f"⚠️ Servicio {i+1} forzado a cerrar")
                except Exception as e:
                    logger.error(f"❌ Error deteniendo servicio {i+1}: {e}")
        
        logger.info("✅ Todos los servicios han sido detenidos")
    
    async def run_full_system(self):
        """Ejecutar sistema completo"""
        logger.info("🚀 Iniciando PyF_ML_G2 Sistema MLOps Completo...")
        print("="*50)
        
        try:
            # 1. Verificaciones preliminares
            if not self.check_python_version():
                return False
            
            if not self.check_dependencies():
                return False
            
            # 2. Configurar entorno
            if not self.setup_environment():
                return False
            
            # 3. Entrenar modelos si es necesario
            self.train_initial_models()
            
            # 4. Iniciar servicios
            logger.info("🔄 Iniciando servicios...")
            
            mlflow_process = self.start_mlflow_server()
            if mlflow_process:
                self.processes.append(mlflow_process)
            
            api_process = self.start_api_server()
            if api_process:
                self.processes.append(api_process)
            
            dashboard_process = self.start_dashboard()
            if dashboard_process:
                self.processes.append(dashboard_process)
            
            # 5. Abrir navegadores
            self.open_browsers()
            
            # 6. Mostrar información del sistema
            self.print_system_info()
            
            # 7. Mantener vivo el sistema
            try:
                while True:
                    await asyncio.sleep(5)
                    
                    # Verificar salud de los procesos
                    alive_processes = [p for p in self.processes if p.poll() is None]
                    if len(alive_processes) != len(self.processes):
                        logger.warning("⚠️ Algunos servicios se han detenido")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Recibida señal de interrupción...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            return False
        
        finally:
            self.cleanup()


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="PyF_ML_G2 - Sistema MLOps para Predicción de Precios Inmobiliarios"
    )
    parser.add_argument("--setup-only", action="store_true", help="Solo configurar entorno")
    parser.add_argument("--train-only", action="store_true", help="Solo entrenar modelos")
    parser.add_argument("--api-only", action="store_true", help="Solo iniciar API")
    parser.add_argument("--dashboard-only", action="store_true", help="Solo iniciar Dashboard")
    parser.add_argument("--mlflow-only", action="store_true", help="Solo iniciar MLflow")
    parser.add_argument("--version", action="version", version="PyF_ML_G2 1.0.0")
    
    args = parser.parse_args()
    
    orchestrator = PyFMLG2Orchestrator()
    
    try:
        if args.setup_only:
            logger.info("🔧 Modo: Solo configuración")
            success = orchestrator.setup_environment()
            sys.exit(0 if success else 1)
        
        elif args.train_only:
            logger.info("🤖 Modo: Solo entrenamiento")
            orchestrator.setup_environment()
            success = orchestrator.train_initial_models()
            sys.exit(0 if success else 1)
        
        elif args.api_only:
            logger.info("🚀 Modo: Solo API")
            orchestrator.setup_environment()
            process = orchestrator.start_api_server()
            if process:
                orchestrator.processes.append(process)
                input("Presiona Enter para detener...")
                orchestrator.cleanup()
        
        elif args.dashboard_only:
            logger.info("📊 Modo: Solo Dashboard")
            orchestrator.setup_environment()
            process = orchestrator.start_dashboard()
            if process:
                orchestrator.processes.append(process)
                input("Presiona Enter para detener...")
                orchestrator.cleanup()
        
        elif args.mlflow_only:
            logger.info("🔬 Modo: Solo MLflow")
            orchestrator.setup_environment()
            process = orchestrator.start_mlflow_server()
            if process:
                orchestrator.processes.append(process)
                input("Presiona Enter para detener...")
                orchestrator.cleanup()
        
        else:
            # Modo completo (por defecto)
            asyncio.run(orchestrator.run_full_system())
    
    except KeyboardInterrupt:
        logger.info("👋 Sistema detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()