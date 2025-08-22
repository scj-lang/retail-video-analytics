# Retail Video Analytics

Sistema de video-analytics para detección de personas usando YOLOX, tracking anónimo y análisis de Zonas de Interés (ROIs) en entornos retail.

## Objetivos del Proyecto

- **Detección de Personas**: Utilizar YOLOX para detectar personas en video streams
- **Tracking Anónimo**: Implementar ByteTrack para seguimiento de personas sin identificación
- **Zonas de Interés (ROIs)**: Definir y monitorear áreas específicas del espacio retail
- **Análisis de Eventos**: Generar métricas y eventos basados en movimiento en ROIs

## Estructura del Proyecto

```
retail-video-analytics/
├─ README.md
├─ requirements.txt
├─ notebooks/
│  └─ 01_semana1_prototipo.ipynb
├─ src/
│  ├─ roi_editor.py
│  ├─ tracker_pipeline.py
│  ├─ utils.py
│  └─ config/
│     └─ rois_cartagena.json
├─ data/
│  ├─ raw/          # Videos originales
│  ├─ processed/    # Videos con overlays
│  └─ logs/         # CSV de eventos
└─ reports/
   └─ semana1_bitacora.md
```

## Instalación

### Opción 1: Entorno Local

1. Clonar el repositorio
2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Opción 2: Google Colab

1. Abrir el notebook `notebooks/01_semana1_prototipo.ipynb` en Colab
2. Ejecutar la primera celda para instalar dependencias:
```python
!pip install yolox opencv-python supervision shapely numpy pandas torch torchvision
```

## Demo Mínimo

### 1. Definir ROIs
```bash
python src/roi_editor.py --input data/raw/video_sample.mp4
```

### 2. Ejecutar Pipeline de Análisis
```bash
python src/tracker_pipeline.py --input data/raw/video_sample.mp4 --output data/processed/
```

## Características Principales

- **YOLOX-Nano**: Modelo ligero para detección en tiempo real
- **ByteTrack**: Tracking robusto sin necesidad de re-identificación
- **ROIs Dinámicos**: Editor visual para definir zonas poligonales
- **Métricas Automáticas**: Conteo de personas, tiempo de permanencia, flujo

## Próximos Pasos

- [ ] Completar implementación del editor de ROIs
- [ ] Integrar pipeline completo YOLOX + ByteTrack
- [ ] Añadir métricas de análisis temporal
- [ ] Implementar dashboard de visualización

## Contribución

Este es un prototipo en desarrollo para análisis de video retail. Ver `reports/semana1_bitacora.md` para detalles del sprint actual.