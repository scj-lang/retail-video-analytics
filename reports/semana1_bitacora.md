# Bitácora Semana 1 - Retail Video Analytics

## Resumen del Sprint

**Objetivo**: Crear la estructura base del proyecto y un prototipo funcional de video-analytics para retail.

**Duración**: Semana 1  
**Estado**: En desarrollo  

## Entregables Completados

### ✅ Estructura del Proyecto
- [x] Directorios organizados (`src/`, `data/`, `notebooks/`, `reports/`)
- [x] Archivo `requirements.txt` con dependencias
- [x] README.md con documentación inicial
- [x] Configuración base de ROIs en JSON

### ✅ Código Base
- [x] `roi_editor.py`: Editor visual de ROIs con interfaz OpenCV
- [x] `tracker_pipeline.py`: Pipeline completo YOLO + ByteTrack
- [x] `utils.py`: Funciones auxiliares para geometría y video
- [x] Notebook de prototipo con ejemplos

## Funcionalidades Implementadas

### ROI Editor (`roi_editor.py`)
- **Interfaz visual**: Click izquierdo para agregar puntos, click derecho para finalizar ROI
- **Gestión de ROIs**: Crear, visualizar y guardar múltiples zonas
- **Exportación JSON**: Configuración persistente para el pipeline
- **Validación**: Mínimo 3 puntos por polígono

### Pipeline de Tracking (`tracker_pipeline.py`)
- **Detección YOLOv8**: Modelo ligero (yolov8n.pt) para personas
- **Tracking ByteTrack**: Seguimiento robusto sin re-identificación
- **Eventos ROI**: Detección automática de entrada/salida de zonas
- **Output múltiple**: Video anotado + CSV de eventos
- **Métricas en tiempo real**: Conteo de personas, progreso de procesamiento

### Utilities (`utils.py`)
- **Geometría**: Validación de puntos en polígonos usando Shapely
- **Video handling**: Información de video, validación de archivos
- **Configuración**: Carga/guardado de ROIs en JSON
- **Visualización**: Funciones para dibujar polígonos y anotaciones

## Arquitectura Técnica

```
YOLOv8 (Detección) → ByteTrack (Tracking) → ROI Analysis → Event Logging
                                              ↓
                                        Video Annotation
```

### Dependencias Principales
- **ultralytics**: YOLOv8 para detección de objetos
- **supervision**: Utilidades para computer vision y tracking
- **shapely**: Operaciones geométricas con polígonos
- **opencv-python**: Procesamiento de video e interfaz visual
- **pandas**: Manejo de datos y eventos

## Decisiones de Diseño

1. **YOLOv8n vs modelos más pesados**: Elegido por balance rendimiento/precisión
2. **ByteTrack**: Tracking sin features visuales, ideal para anonimato
3. **OpenCV para ROIs**: Interfaz simple y directa para definición manual
4. **JSON para configuración**: Formato legible y fácil de editar
5. **Centro inferior de bbox**: Mejor representación de posición en el suelo

## Pruebas Realizadas

### ✅ Tests Básicos
- [x] Carga de modelos YOLOv8
- [x] Interfaz ROI editor (clicks y visualización)
- [x] Guardado/carga de configuración JSON
- [x] Pipeline básico sin errores de dependencias

### 🔄 Tests Pendientes
- [ ] Procesamiento video completo end-to-end
- [ ] Validación de eventos ROI con video real
- [ ] Performance con videos largos
- [ ] Precisión del tracking en escenas complejas

## Problemas Encontrados

### Resueltos
1. **Imports de supervision**: Ajustado para compatibilidad con ByteTrack
2. **Formato de detecciones**: Conversión correcta YOLO → supervision
3. **Coordenadas ROI**: Usar centro inferior de bbox para mejor tracking

### En seguimiento
1. **Performance**: Evaluar velocidad de procesamiento en videos largos
2. **Memoria**: Monitorear uso con videos de alta resolución
3. **Edge cases**: Tracking cuando personas salen/entran del frame

## Métricas del Sprint

- **Archivos creados**: 8
- **Líneas de código**: ~800
- **Funciones implementadas**: 25+
- **Tiempo estimado**: 2-3 días de desarrollo

## Plan Semana 2

### Prioridad Alta
- [ ] Testing completo con videos reales
- [ ] Optimización de performance
- [ ] Métricas avanzadas (tiempo de permanencia, flujo direccional)
- [ ] Manejo de errores y edge cases

### Prioridad Media
- [ ] Dashboard de visualización web
- [ ] Configuración dinámica de parámetros
- [ ] Exportación de reportes automáticos
- [ ] Integración con base de datos

### Prioridad Baja
- [ ] Detección de eventos complejos (grupos, colas)
- [ ] Calibración automática de ROIs
- [ ] API REST para procesamiento remoto

## Notas Técnicas

### Comandos de Uso
```bash
# Definir ROIs
python src/roi_editor.py --input video.mp4

# Procesar video
python src/tracker_pipeline.py --input video.mp4 --output data/processed/

# Jupyter notebook
jupyter notebook notebooks/01_semana1_prototipo.ipynb
```

### Estructura de Eventos
```csv
timestamp,frame_number,person_id,event_type,roi_id,roi_name,position_x,position_y
12.5,375,1,roi_entry,0,entrance,245,890
15.2,456,1,roi_exit,0,entrance,180,892
```

## Conclusiones

El prototipo inicial cumple con los objetivos planteados. La arquitectura modular permite extensiones futuras y el código está listo para testing con datos reales. La implementación prioriza simplicidad y funcionalidad sobre optimizaciones prematuras.

**Status general**: ✅ **MVP Completo** - Listo para testing y refinamiento en Semana 2.