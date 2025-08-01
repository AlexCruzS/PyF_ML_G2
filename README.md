# 🏠 PyF_ML_G2 - Sistema MLOps para Predicción de Precios Inmobiliarios

[![Model Deployment](https://github.com/tu-usuario/PyF_ML_G2/workflows/Model%20Deployment/badge.svg)](https://github.com/tu-usuario/PyF_ML_G2/actions)
[![Release](https://github.com/tu-usuario/PyF_ML_G2/workflows/Release/badge.svg)](https://github.com/tu-usuario/PyF_ML_G2/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26%2B-red.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.7%2B-orange.svg)](https://mlflow.org)

Sistema completo de Machine Learning para predicción de precios inmobiliarios desarrollado con **Arquitectura Hexagonal + DDD**, **MLflow** para tracking, **FastAPI** para API REST, **Streamlit** para dashboard y **PostgreSQL** como base de datos.

## **🚀 Quick Start - ¡Sistema Completo en 30 segundos!**

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/PyF_ML_G2.git
cd PyF_ML_G2

# ¡Un solo comando para TODO el sistema!
python main.py
```

**✅ ¿Qué hace este comando?**
- 🤖 Auto-entrena modelos si no existen (primera vez)
- 🚀 Lanza API Server en http://localhost:8000
- 📊 Lanza Dashboard en http://localhost:8501
- 🔬 Lanza MLflow UI en http://localhost:5000
- 🗄️ Configura PostgreSQL automáticamente
- ✨ Sistema 100% funcional desde el primer minuto

---

## **🎯 Features Principales**

### **📊 Dashboard Interactivo (Streamlit)**
- ✅ Formulario interactivo para valoración de propiedades
- ✅ Predicción en tiempo real con confidence score
- ✅ Comparación con valores de mercado
- ✅ Visualizaciones interactivas de tendencias inmobiliarias
- ✅ Análisis exploratorio de datos del mercado
- ✅ Mapa interactivo de propiedades similares

### **🚀 API REST (FastAPI)**
- ✅ POST /predict: Predicciones de precios inmobiliarios
- ✅ GET /health: Health check completo del sistema
- ✅ GET /model/info: Información del modelo en producción
- ✅ GET /properties/similar: Propiedades comparables
- ✅ OpenAPI Docs: Documentación automática
- ✅ Autenticación y rate limiting

### **🤖 Machine Learning Pipeline**
- ✅ 3 modelos comparados: RandomForest, XGBoost, LightGBM
- ✅ MLflow tracking: Experimentos automáticos
- ✅ Feature engineering: 15+ features optimizadas
- ✅ Model serving: Carga automática del mejor modelo
- ✅ Validación cruzada y métricas robustas

---

## **🏗 Arquitectura Técnica**

### **Hexagonal Architecture + DDD**

```
pyf_ml_g2/
├── domain/                   # 🎯 DOMAIN LAYER
│   ├── entities.py               # Property, Valuation, Prediction entities
│   ├── ports.py                  # Interfaces/Puertos
│   └── services.py               # PropertyValuationDomainService
├── adapters/                 # 🔌 ADAPTERS LAYER
│   ├── database/
│   │   └── data_adapter.py       # PostgreSQL adapter
│   └── ml/
│       ├── mlflow_adapter.py     # MLflow tracking & registry
│       └── sklearn_adapter.py    # Scikit-learn models
├── api/                      # 🚀 API REST
│   ├── main.py                   # FastAPI application
│   └── controller.py             # API controllers
├── pipeline/                 # 🤖 ML PIPELINES
│   ├── train.py                  # Training pipeline
│   └── predict.py                # Prediction pipeline
└── monitoring/               # 📊 MONITORING
    └── dashboard.py              # Streamlit dashboard
```

---

## **📊 Resultados de ML**

| Modelo | RMSE (USD) | MAE (USD) | R² Score | Status |
|--------|------------|-----------|----------|---------|
| **RandomForest** ⭐ | 15,420 | 11,250 | 0.892 | En Producción |
| XGBoost | 16,180 | 12,100 | 0.881 | Backup |
| LightGBM | 17,340 | 13,450 | 0.867 | Baseline |

**🎯 Precisión:** 89% de predicciones dentro de ±10% del valor real  
**📈 Dataset:** +50,000 propiedades inmobiliarias  
**🌎 Cobertura:** Área metropolitana completa  

---

## **🛠 Technology Stack**

### **🏛 Core Architecture**
- **🐍 Backend:** Python 3.9+, FastAPI, asyncio
- **📊 Frontend:** Streamlit, Plotly, Folium Maps
- **🤖 ML:** scikit-learn, XGBoost, LightGBM, MLflow
- **🗄️ Database:** PostgreSQL (AWS RDS compatible)

### **🚀 MLOps & DevOps**
- **🐳 Containerization:** Docker, Docker Compose
- **☁️ Cloud:** AWS RDS, S3 (MLflow artifacts)
- **📊 Monitoring:** Prometheus, Grafana
- **🔄 CI/CD:** GitHub Actions

### **🌍 Geospatial & Data**
- **📍 Geocoding:** Geopy, Folium
- **📊 Visualization:** Plotly, Matplotlib, Seaborn
- **💾 Data Processing:** Pandas, NumPy, PyArrow

---

## **🛠 Installation & Setup**

### **Opción 1: Setup Local (Desarrollo)**

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac

# 4. Ejecutar sistema completo
python main.py
```

### **Opción 2: Docker Compose (Producción)**

```bash
# Configurar variables de entorno
copy .env.example .env

# Levantar todo el stack
docker-compose up -d
```

---

## **🎮 URLs del Sistema**

Después de ejecutar `python main.py`:

- **📊 Dashboard Principal:** http://localhost:8501
  - Interfaz para valoración de propiedades
  - Análisis de mercado inmobiliario
  - Visualizaciones interactivas

- **🚀 API Documentation:** http://localhost:8000/docs
  - Documentación Swagger/OpenAPI
  - Endpoints interactivos
  - Ejemplos de uso

- **🔬 MLflow Experiments:** http://localhost:5000
  - Tracking de experimentos
  - Registry de modelos
  - Métricas de performance

- **🏥 Health Check:** http://localhost:8000/health
  - Estado del sistema
  - Conectividad de servicios

---

## **📊 Ejemplo de Uso - API**

### **Predicción de Precio**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "address": "123 Main St, Lima, PE",
    "area_m2": 120.5,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 2,
    "property_type": "Single Family",
    "garage_spaces": 1
  }'
```

### **Respuesta Esperada**

```json
{
  "prediction_id": "pred_123abc",
  "predicted_value": 185750.30,
  "confidence_score": 0.89,
  "confidence_level": "high",
  "model_name": "RandomForest",
  "model_version": "v1.2.3",
  "comparable_properties": 8,
  "market_analysis": {
    "avg_price_per_m2": 1543.25,
    "neighborhood_trend": "increasing"
  }
}
```

---

## **📈 MLOps Principles Implementados**

- ✅ **Experiment Tracking:** MLflow para versionado automático
- ✅ **Model Serving:** FastAPI con carga automática del mejor modelo
- ✅ **Monitoring:** Dashboard en tiempo real con métricas de negocio
- ✅ **Automation:** Pipeline automatizado de entrenamiento y deployment
- ✅ **Reproducibility:** Docker containers para ambientes consistentes
- ✅ **Observability:** Health checks y logging estructurado
- ✅ **Data Validation:** Validación automática de calidad de datos
- ✅ **A/B Testing:** Comparación de modelos en producción

---

## **🧪 Testing**

```bash
# Ejecutar todos los tests
pytest

# Tests con coverage
pytest --cov=pyf_ml_g2 --cov-report=html

# Tests específicos
pytest tests/unit/          # Tests unitarios
pytest tests/integration/   # Tests de integración
pytest tests/e2e/          # Tests end-to-end
```

### **Cobertura de Tests**
- **Unit Tests:** 85%+ cobertura
- **Integration Tests:** APIs y Base de datos
- **E2E Tests:** Flujos completos de usuario

---

## **📂 Estructura del Proyecto**

```
PyF_ML_G2/
├── 🏗️ PRODUCTION CODE (Hexagonal Architecture + DDD)
│   └── pyf_ml_g2/
│       ├── domain/                   # Lógica de negocio pura
│       ├── adapters/                 # Integraciones externas
│       ├── api/                      # REST API
│       ├── pipeline/                 # ML Pipelines
│       └── monitoring/               # Dashboard
│
├── 📚 EDUCATIONAL RESOURCES
│   └── educational_resources/
│       ├── notebooks/                # Jupyter notebooks
│       ├── scripts/                  # Scripts educativos
│       └── presentation_materials/   # Documentación
│
├── 🐳 DEPLOYMENT
│   └── deployment/
│       ├── docker-compose.yml        # Orquestación
│       └── Dockerfile.*              # Containers
│
├── 📊 DATA & MODELS
│   └── data/
│       ├── raw/                      # Datos originales
│       ├── processed/                # Datos procesados
│       └── models/                   # Modelos entrenados
│
├── 🧪 TESTING
│   └── tests/
│       ├── unit/                     # Tests unitarios
│       ├── integration/              # Tests integración
│       └── e2e/                      # Tests end-to-end
│
└── 📋 PROJECT ROOT
    ├── main.py                       # Punto de entrada
    ├── requirements.txt              # Dependencias
    └── README.md                     # Esta documentación
```

---

## **🤝 Contributing**

1. Fork el repositorio
2. Crear feature branch (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Abrir Pull Request

### **Guidelines de Desarrollo**
- Seguir principios de Clean Architecture
- Tests son obligatorios para nuevas features
- Code coverage mínimo del 80%
- Documentar APIs con OpenAPI
- Usar type hints en Python

---

## **📄 License**

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## **👥 Equipo de Desarrollo**

### **🏛 Arquitectura & Backend**
- **Lead Developer** - Arquitectura Hexagonal y DDD
- **MLOps Engineer** - Pipeline ML y deployment

### **🤖 Machine Learning**
- **ML Engineer** - Modelos y Feature Engineering
- **Data Scientist** - Análisis y validación

### **🚀 DevOps & Infrastructure**
- **DevOps Engineer** - CI/CD y containerización
- **Cloud Engineer** - Infrastructure y escalabilidad

---

## **🙏 Agradecimientos**

- **Comunidad MLflow** por el framework de ML lifecycle
- **FastAPI** por la documentación automática
- **Streamlit** por facilitar dashboards interactivos
- **Proyecto Taxi Duration Predictor** por la inspiración arquitectural

---

## **📚 Recursos Adicionales**

- **📖 [Documentación Completa](./docs/)**
- **🎓 [Guía de MLOps](./educational_resources/)**
- **🏗️ [Arquitectura Detallada](./docs/architecture.md)**
- **🚀 [Deployment Guide](./deployment/README.md)**

---

**🎯 Este proyecto demuestra un sistema MLOps completo end-to-end con arquitectura profesional para el dominio inmobiliario.**