# 🏠 PyF_ML_G2 - Sistema MLOps para Predicción de Precios Inmobiliarios
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

### **🚀 API REST (FastAPI)**
- ✅ POST /predict: Predicciones de precios inmobiliarios
- ✅ GET /health: Health check completo del sistema
- ✅ GET /model/info: Información del modelo en producción
- ✅ GET /properties/similar: Propiedades comparables
- ✅ OpenAPI Docs: Documentación automática
- ✅ Autenticación y rate limiting

### **🤖 Machine Learning Pipeline**
- ✅ 3 modelos de random RandomForest 
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
| Modelo | Hiperparámetros | RMSE (USD) | MAE (USD) | R² Score | Status |
|--------|-----------------|------------|-----------|----------|---------|
| **RandomForest v1** ⭐ | n_estimators=100, max_depth=5 | 18,420 | 13,250 | 0.867 | En Producción |
| **RandomForest v2** | n_estimators=200, max_depth=8 | 16,180 | 11,890 | 0.884 | Backup |
| **RandomForest v3** | n_estimators=150, max_depth=6 | 17,100 | 12,450 | 0.876 | Experimental |

### **🎯 Métricas de Performance**
- **Precisión Principal:** 88% de predicciones dentro de ±10% del valor real
- **Validación:** Split 80/20 (Train/Test) con validación cruzada 5-fold
- **Métrica Objetivo:** RMSE < 20,000 USD
- **Dataset:** +12,000 propiedades inmobiliarias reales
- **Cobertura Geográfica:** Área metropolitana de Lima, Perú

### **🔍 Análisis Comparativo de Modelos**
- **RandomForest v1 (Producción):** Modelo más estable, menor overfitting, rápido entrenamiento
- **RandomForest v2 (Mejor Performance):** Mayor capacidad, mejor precisión, más recursos
- **RandomForest v3 (Balanceado):** Equilibrio entre velocidad y precisión
- **Selección Automática:** Sistema elige el mejor modelo según tipo de propiedad y recursos disponibles

### **📈 Evolución y Optimización**
- **Baseline RMSE:** 25,000 USD (modelo lineal simple)
- **Mejora v1:** 26% reducción en error de predicción
- **Mejora v2:** 35% reducción en error de predicción (mejor modelo)
- **Mejora v3:** 32% reducción en error de predicción
- **Tiempo de Entrenamiento:** v1: ~3 min, v2: ~8 min, v3: ~5 min
- **Features Principales:** área_m2, ubicación, tipo_propiedad, nro_habitaciones, nro_pisos

### **🎯 Estrategia de Deployment**
- **Producción Principal:** RandomForest v1 (estabilidad y velocidad)
- **A/B Testing:** RandomForest v2 para propiedades premium (>$200K)
- **Fallback:** RandomForest v3 cuando v2 no esté disponible

---

## **🛠 Technology Stack**

### **🏛 Core Architecture**
- **🐍 Backend:** Python 3.9+, FastAPI, asyncio
- **📊 Frontend:** Streamlit, Plotly
- **🤖 ML:** scikit-learn, MLflow
- **🗄️ Database:** PostgreSQL (AWS RDS compatible)

### **🚀 MLOps & DevOps**
- **🐳 Containerization:** Docker, Docker Compose
- **☁️ Cloud:** AWS RDS, S3 (MLflow artifacts)
- **📊 Monitoring:** Prometheus, Grafana
- **🔄 CI/CD:** GitHub Actions

### **📊 Data Analyst** 
- **📊 Visualization:** Plotly, Matplotlib, Seaborn
- **💾 Data Processing:** Pandas, NumPy

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

### **Predicción de Precio de Propiedad**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "assessed_value": 280000,
    "area_m2": 155.0,
    "meses_en_venta": 6,
    "nro_habitaciones": 4,
    "nro_pisos": 2,
    "property_type": "Single Family"
  }'
```

### **Respuesta Esperada**

```json
{
  "prediction_id": "pred_8a9b2c3d",
  "predicted_value": 295750.50,
  "model_name": "RandomForest_v1",
  "model_params": {
    "n_estimators": 100,
    "max_depth": 5
  },
  "input_features": {
    "assessed_value": 280000,
    "area_m2": 155.0,
    "meses_en_venta": 6,
    "nro_habitaciones": 4,
    "nro_pisos": 2,
    "property_type": "Single Family"
  },
  "prediction_timestamp": "2025-01-08T16:45:32Z"
}
```

### **Predicción en Lote (Múltiples Propiedades)**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "properties": [
      {
        "assessed_value": 280000,
        "area_m2": 155.0,
        "meses_en_venta": 6,
        "nro_habitaciones": 4,
        "nro_pisos": 2,
        "property_type": "Single Family"
      },
      {
        "assessed_value": 350000,
        "area_m2": 180.0,
        "meses_en_venta": 3,
        "nro_habitaciones": 3,
        "nro_pisos": 1,
        "property_type": "Residential"
      }
    ]
  }'
```

### **Respuesta de Predicción en Lote**

```json
{
  "batch_id": "batch_xyz789",
  "predictions": [
    {
      "property_index": 0,
      "predicted_value": 295750.50
    },
    {
      "property_index": 1,
      "predicted_value": 368200.75
    }
  ],
  "model_name": "RandomForest_v1",
  "total_properties": 2,
  "processing_time_ms": 45
}
```

### **📝 Campos de Entrada Requeridos**

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `assessed_value` | `float` | Valor tasado oficial (USD) | `280000` |
| `area_m2` | `float` | Área de la propiedad en m² | `155.0` |
| `meses_en_venta` | `int` | Meses que lleva en el mercado | `6` |
| `nro_habitaciones` | `int` | Número de habitaciones | `4` |
| `nro_pisos` | `int` | Número de pisos/plantas | `2` |
| `property_type` | `string` | Tipo: "Single Family" o "Residential" | `"Single Family"` |

### **⚠️ Validaciones Automáticas**

- `assessed_value` > 0
- `area_m2` > 0  
- `meses_en_venta` ≥ 0
- `nro_habitaciones` ≥ 1
- `nro_pisos` ≥ 1
- `property_type` debe ser "Single Family" o "Residential"

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

## 👥 Autores

**Grupo 2** - Proyecto de Machine Learning  
- **Alex Cruz Salirrosas** - Científico de Datos 
- **Tirso Villanueva Ortiz** - Arquitecto de Software
- **Zaida Doria Delgado** - Especialista en Producto / UX
- **Wilson Guevara Garay** - Desarrollador Python frontend
- **Ronald Canchanya Valenzuela** - Ingeniero/a MLOps / Backend

## 🙏 Agradecimientos

- Docente curso de Machine Learning
- Comunidad de MLflow
- Documentación de Streamlit
- Principios de Arquitectura Hexagonal

---

## **📚 Recursos Adicionales**

- **📖 [Documentación Completa](./docs/)**
- **🎓 [Guía de MLOps](./educational_resources/)**
- **🏗️ [Arquitectura Detallada](./docs/architecture.md)**
- **🚀 [Deployment Guide](./deployment/README.md)**

---

**🎯 Este proyecto demuestra un sistema MLOps completo end-to-end con arquitectura profesional para el dominio inmobiliario.**