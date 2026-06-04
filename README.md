# tfg_deteccion_maderas
Sistema de Deep Learning basado en arquitecturas YOLO para la detección automática de rasgos anatómicos en imágenes macroscópicas de madera.
Este repositorio contiene el código fuente, los scripts de análisis y la metodología experimental desarrollados para el Trabajo de Fin de Grado en **Ciencia de Datos e Inteligencia Artificial** por la Universidad Politécnica de Madrid (UPM).

El objetivo principal del proyecto es la detección y delimitación automática de 14 rasgos anatómicos (como poros, distintos tipos de parénquima y radios leñosos) estandarizados por la IAWA en imágenes macroscópicas de madera, apoyando así los procesos de inspección aduanera y control de deforestación (Reglamento EUDR 2023/1115).

## Desafíos Técnicos y Aportaciones
El principal desafío abordado en este trabajo es el **desequilibrio de clases extremo (ratio de 1790:1)**  dado así en las estructuras biológicas de la madera. Para mitigarlo, se ha implementado un *pipeline* experimental iterativo que incluye:

1. Transformación algorítmica de anotaciones morfológicas de expertos a *bounding boxes* (YOLO format).
2. Estrategia de partición estratificada.
3. Evaluación y calibración de optimizadores (AdamW vs MuSGD) en arquitecturas **YOLOv11n** y **YOLO26**.
4. Intervención a nivel de datos mediante la generación de **recortes adaptativos selectivos** centrados en rasgos minoritarios para evitar el *domain shift*.
