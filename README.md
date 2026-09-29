# TFM — Segmentación semántica de construcciones en imágenes satelitales

Repositorio del Trabajo de Fin de Máster (TFM) sobre **segmentación semántica de edificaciones** en imágenes satelitales de 1 m de resolución espacial, comparando tres arquitecturas de referencia bajo un mismo protocolo de entrenamiento y evaluación.

## Modelos evaluados

| Modelo | Encoder | Implementación |
|--------|---------|----------------|
| U-Net | ResNet101 (ImageNet) | `segmentation-models-pytorch` |
| DeepLabV3+ | ResNet101 (ImageNet) | `segmentation-models-pytorch` |
| SegFormer | MiT-B2 (ImageNet) | `segmentation-models-pytorch` |

Todos comparten el mismo protocolo: pérdida **Dice + Focal**, optimizador **AdamW**, scheduler **CosineAnnealingLR**, y evaluación con **Test-Time Augmentation (TTA)** y búsqueda del **umbral de decisión óptimo**, además de las métricas habituales (IoU, Dice/F1, Precision, Recall, Accuracy).

## Experimentos

| Experimento | Dataset | Notebooks |
|-------------|---------|-----------|
| **Experimento 1** — Massachusetts | [balraj98/massachusetts-buildings-dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset) (carpeta `tiff`) | `notebooks/experimento_1_massachusetts/` |
| **Experimento 2** — Dataset tileado | [prabalpratapsinghml/building](https://www.kaggle.com/datasets/prabalpratapsinghml/building) (`archive_patch`, nombres tileados) | `notebooks/experimento_2_tileado/` |

La diferencia entre ambos experimentos es el dataset de origen: el segundo ya viene particionado en patches con nombres tileados, mientras que el primero trabaja sobre los tiles de 256×256 derivados del dataset original.

## Estructura del repositorio

```
TFM-segmentacion-construcciones/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── experimento_1_massachusetts/
    │   ├── unet_resnet101.ipynb
    │   ├── deeplabv3plus_resnet101.ipynb
    │   └── segformer_mit_b2.ipynb
    └── experimento_2_tileado/
        ├── unet_resnet101_tileado.ipynb
        ├── deeplabv3plus_resnet101_tileado.ipynb
        └── segformer_mit_b2_tileado.ipynb
```

Cada notebook es **autocontenido**: instala sus dependencias críticas (`segmentation-models-pytorch==0.4.0`, `albumentations==1.4.24`), define la carga de datos, entrena, evalúa y genera los artefactos (checkpoints `.pth`, históricos CSV, CSV de umbral óptimo y máscaras de predicción).

## Cómo reproducirlo

1. **Sube el notebook a [Kaggle](https://www.kaggle.com/)** (los notebooks apuntan a rutas `/kaggle/input/datasets/...`) o ajusta la constante `CARPETA_DATOS` a tu ruta local.
2. Selecciona un entorno con GPU (T4 o superior).
3. Ejecuta las celdas en orden. La primera celda instala/verifica las versiones requeridas.
4. Los resultados se escriben en el directorio de trabajo: `historial_entrenamiento.csv`, `threshold_validation_*.csv`, checkpoints `*_epoca*_iou*.pth` y las tablas finales de test.

Instalación local (opcional):

```bash
pip install -r requirements.txt
jupyter lab
```

## Artefactos generados por cada notebook

- **Checkpoints**: mejor modelo por IoU de validación (`*_epoca*_iou*.pth`).
- **`historial_entrenamiento.csv`**: métricas por época (train/val).
- **`threshold_validation_*.csv`**: barrido del umbral de decisión óptimo.
- **Tabla final de test**: métricas con TTA y umbral óptimo aplicados.
- **Visuales**: curvas de aprendizaje, ejemplos de predicción y comparativas GT vs predicción.

## Estado

Cuaderno con **salidas ejecutadas** (métricas y gráficas incluidas) para facilitar la revisión sin tener que reentrenar.
