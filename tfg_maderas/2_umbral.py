import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────
BASE_DIR = r'C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\1parte\YOLO_Dataset_Final'
SPLITS = ['train', 'valid', 'test'] 
IMG_W, IMG_H = 1500, 1500 

ID_TO_NAME = {
    0:'8', 1:'56', 2:'58', 3:'79', 4:'81', 5:'82',
    6:'83', 7:'84', 8:'85', 9:'86', 10:'89',
    11:'10', 12:'V1', 13:'radio'
}

def analizar_splits():
    all_data = []
    
    for split in SPLITS:
        labels_path = os.path.join(BASE_DIR, split, 'labels')
        if not os.path.exists(labels_path):
            print(f"Carpeta no encontrada: {labels_path}")
            continue
            
        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
        print(f"Procesando {split}: {len(label_files)} archivos...")

        for file in label_files:
            with open(os.path.join(labels_path, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    cls_id = int(parts[0])
                    # YOLO: class x y width height (normalizado)
                    w_px = float(parts[3]) * IMG_W
                    h_px = float(parts[4]) * IMG_H
                    area_px = w_px * h_px
                    
                    all_data.append({
                        'split': split,
                        'class_name': ID_TO_NAME.get(cls_id, str(cls_id)),
                        'area_px': area_px
                    })

    df = pd.DataFrame(all_data)

    # 1. estadísticas globales por split (Percentiles bajos para detectar ruido)
    print("\n PERCENTILES DE ÁREA POR SPLIT (px²):")
    stats = df.groupby('split')['area_px'].describe(percentiles=[.01, .05, .1, .5])
    print(stats)

    # 2. comparativa de ruido entre Train, Valid y Test
    plt.figure(figsize=(14, 7))
    sns.boxenplot(data=df, x='split', y='area_px', palette='Set2')
    plt.yscale('log')
    plt.axhline(y=15, color='red', linestyle='--', label='Umbral de Ruido Sugerido (15px²)')
    plt.title('Distribución de Áreas: Train vs Valid vs Test')
    plt.ylabel('Área en Píxeles² (Escala Logarítmica)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 3. reporte de menores a 10px
    print("\n ANOTACIONES POTENCIALMENTE ERRÓNEAS (< 10 px²):")
    ruido = df[df['area_px'] < 10].groupby('split').size()
    print(ruido if not ruido.empty else "No se detectaron anotaciones menores a 10px².")

    return df

if __name__ == "__main__":
    df_final = analizar_splits()