"""
train.py
========
Script de entrenamiento para los seis modelos del TFG.

Uso:
    python src/07_train.py --model M1
    python src/07_train.py --model M6
    python src/07_train.py --model all   # entrena todos en secuencia

Modelos disponibles: M1, M2, M3, M4, M4r, M5, M6
"""

import argparse
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIGURACIONES DE CADA MODELO
# ─────────────────────────────────────────────

CONFIGS = {

    "M1": dict(
        description="Baseline, configuración estándar sin modificaciones",
        weights="yolo11n.pt",
        data="data/data.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=20,
        name="M1_Baseline",
    ),

    "M2": dict(
        description="Bocetos, imágenes binarizadas de bordes",
        weights="yolo11n.pt",
        data="data/data_sketches.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=20,
        name="M2_Sketches",
    ),

    "M3": dict(
        description="Ajuste de hiperparámetros, cls aumentado + copy-paste",
        weights="yolo11n.pt",
        data="data/data.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=50,
        cls=1.5,
        copy_paste=0.4,
        cos_lr=True,
        degrees=10.0,
        flipud=0.1,
        name="M3_Hiperparametros",
    ),

    "M4": dict(
        description="Arquitectura alternativa YOLO26 (sin calibrar)",
        weights="yolo26n.pt",
        data="data/data.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=50,
        cls=1.5,
        copy_paste=0.4,
        cos_lr=True,
        name="M4_YOLO26",
    ),

    "M4r": dict(
        description="YOLO26 revisado, optimizador MuSGD calibrado",
        weights="yolo26n.pt",
        data="data/data.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=50,
        cls=1.5,
        copy_paste=0.4,
        cos_lr=True,
        lr0=0.001,
        warmup_epochs=10,
        warmup_momentum=0.8,
        name="M4r_YOLO26_Calibrado",
    ),

    "M5": dict(
        description="Recortes adaptativos, dataset ampliado con 1.374 parches",
        weights="yolo11n.pt",
        data="data/data_recortes_adaptativos.yaml",
        epochs=100,
        imgsz=1536,
        batch=8,
        patience=50,
        cls=1.5,
        copy_paste=0.4,
        cos_lr=True,
        name="M5_RecortesAdaptativos",
    ),

    "M6": dict(
        description="Recortes selectivos, solo parches de clases 82 y 84",
        weights="yolo11n.pt",
        data="data/data_recortes_selectivos.yaml",
        epochs=400,
        imgsz=1536,
        batch=8,
        patience=60,
        cls=1.5,
        copy_paste=0.4,
        cos_lr=True,
        name="M6_RecortesSelectivos",
    ),
}



def train(model_id: str):
    if model_id not in CONFIGS:
        raise ValueError(f"Modelo '{model_id}' no reconocido. "
                         f"Opciones: {list(CONFIGS.keys())}")

    cfg = CONFIGS[model_id].copy()
    description = cfg.pop("description")
    weights     = cfg.pop("weights")

    print(f"\n{'─'*55}")
    print(f"  Entrenando: {model_id} — {description}")
    print(f"{'─'*55}\n")

    model = YOLO(weights)
    model.train(project="runs/detect", **cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de los modelos del TFG de detección de "
                    "rasgos anatómicos en madera.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="ID del modelo a entrenar (M1, M2, M3, M4, M4r, M5, M6) "
             "o 'all' para entrenar todos en secuencia.",
    )
    args = parser.parse_args()

    if args.model.upper() == "ALL":
        for model_id in CONFIGS:
            train(model_id)
    else:
        train(args.model.upper())


if __name__ == "__main__":
    main()