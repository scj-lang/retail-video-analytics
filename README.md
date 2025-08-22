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

### Opción 1: Google Colab (Recomendado)

1. **Abrir el notebook en Colab**:
   - Subir el archivo `notebooks/01_semana1_prototipo.ipynb` a Google Colab
   - O usar el enlace directo si está en GitHub: `https://colab.research.google.com/github/tu-usuario/retail-video-analytics/blob/main/notebooks/01_semana1_prototipo.ipynb`

2. **Configurar el entorno**:
   ```python
   # Ejecutar en la primera celda del notebook
   !pip install yolox opencv-python supervision shapely numpy pandas torch torchvision
   
   # Clonar el repositorio (si está en GitHub)
   !git clone https://github.com/tu-usuario/retail-video-analytics.git
   %cd retail-video-analytics
   ```

3. **Subir video de prueba**:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Seleccionar tu video
   ```

4. **Descargar modelo YOLOX**:
   ```python
   !wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
   ```

5. **Ejecutar el pipeline**:
   ```python
   !python src/tracker_pipeline.py --input tu_video.mp4 --model yolox_nano.pth --output /content/processed/
   ```

### Opción 2: Entorno Local

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

## Demo Mínimo

### En Google Colab
1. Seguir las instrucciones de instalación para Colab
2. Ejecutar todas las celdas del notebook `01_semana1_prototipo.ipynb`
3. Los resultados se guardarán en `/content/processed/` y `/content/logs/`

### En Local
### 1. Definir ROIs
```bash
python src/roi_editor.py --input data/raw/video_sample.mp4
```

### 2. Ejecutar Pipeline de Análisis
```bash
python src/tracker_pipeline.py --input data/raw/video_sample.mp4 --output data/processed/
```

## Estructura de Archivos en Colab

Cuando ejecutes en Google Colab, la estructura será:
```
/content/
├─ retail-video-analytics/     # Código del proyecto
├─ tu_video.mp4               # Video subido
├─ yolox_nano.pth            # Modelo descargado
├─ processed/                # Videos procesados
└─ logs/                     # Eventos CSV
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