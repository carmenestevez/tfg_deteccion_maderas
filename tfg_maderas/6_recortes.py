"""
Este algoritmo tiene como objetivo mitigar el desequilibrio 
extremo de clases y el problema de escala en la detección de rasgos anatómicos
minoritarios.

Dado que las imágenes originales son de muy alta resolución (1500x1500px), 
las estructuras minoritarias pueden diluirse al pasar por las capas 
convolucionales de YOLO. Para evitarlo, este script escanea las imágenes de 
entrenamiento en busca de clases críticas y extrae parches centrados 
en estos rasgos.

El tamaño del parche no es fijo se calcula dinámicamente multiplicando el tamaño del rasgo por 1.6, 
garantizando que la red aprenda no solo el objeto, sino su contexto local.
Utiliza distancia euclidiana entre centros  para evitar generar recortes idénticos si hay varios 
rasgos minoritariosmuy juntos.
Recalcula matemáticamente las coordenadas  de todas las cajas delimitadoras originales para proyectarlas 
correctamente sobre el nuevo sistema de referencia del parche recortado.

"""

import os
import cv2
import numpy as np
from collections import Counter

BASE_DIR = r'C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\1parte\YOLO_Dataset_Final'
OUT_DIR  = os.path.join(BASE_DIR, 'train_adaptive_patches')

CONTEXT_FACTOR       = 1.6
MIN_SIZE             = 640    # parche mínimo aunque el rasgo sea pequeño
MAX_RASGO_PX         = 400    # rasgos más grandes que esto no necesitan recorte
MIN_VISIBILITY       = 0.5
REDUNDANCY_THRESHOLD = 250

MINORITY_IDS = [7, 11, 9, 13, 5, 10, 1, 2, 6]

MAX_PER_CLASS = {
    7:  150,
    11: 100,
    9:  150,
    13: 400,
    5:  200,
    10: 200,
    1:  300,
    2:  300,
    6:  400,
}

ID_TO_NAME = {
    0:'8', 1:'56', 2:'58', 3:'79', 4:'81', 5:'82',
    6:'83', 7:'84', 8:'85', 9:'86', 10:'89',
    11:'10', 12:'V1', 13:'radio'
}

os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'labels'), exist_ok=True)


def main():
    img_dir     = os.path.join(BASE_DIR, 'train', 'images')
    lab_dir     = os.path.join(BASE_DIR, 'train', 'labels')
    label_files = [f for f in os.listdir(lab_dir) if f.endswith('.txt')]

    patches_created   = Counter()
    processed_centers = {}

    for label_file in label_files:
        stem     = label_file[:-4]
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        with open(os.path.join(lab_dir, label_file)) as f:
            all_lines = f.readlines()

        for i, line in enumerate(all_lines):
            parts = line.strip().split()
            if not parts:
                continue

            cls, cx_n, cy_n, bw_n, bh_n = map(float, parts)
            cls = int(cls)

            if cls not in MINORITY_IDS:
                continue

            max_cls = MAX_PER_CLASS.get(cls, 300)
            if patches_created[cls] >= max_cls:
                continue

            cx_px = cx_n * W
            cy_px = cy_n * H
            bw_px = bw_n * W
            bh_px = bh_n * H

            # Rasgos grandes ya son visibles en imagen completa
            if bw_px > MAX_RASGO_PX or bh_px > MAX_RASGO_PX:
                continue

            # Tamaño adaptativo con mínimo garantizado
            side = int(max(bw_px, bh_px) * CONTEXT_FACTOR)
            side = max(side, MIN_SIZE)
            side = min(side, W, H)

            # Centrar ventana con clamping
            x1 = int(max(0, min(cx_px - side // 2, W - side)))
            y1 = int(max(0, min(cy_px - side // 2, H - side)))
            x2 = x1 + side
            y2 = y1 + side

            # Filtro de redundancia
            if stem in processed_centers:
                if any(np.sqrt((cx_px-px)**2 + (cy_px-py)**2) < REDUNDANCY_THRESHOLD
                       for px, py in processed_centers[stem]):
                    continue

            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Re-etiquetado con todos los rasgos que caen en el parche
            new_labels = []
            for other_line in all_lines:
                o_parts = other_line.strip().split()
                if not o_parts:
                    continue
                o_cls, o_cx, o_cy, o_bw, o_bh = map(float, o_parts)

                obx1 = (o_cx - o_bw / 2) * W
                oby1 = (o_cy - o_bh / 2) * H
                obx2 = (o_cx + o_bw / 2) * W
                oby2 = (o_cy + o_bh / 2) * H

                ix1 = max(obx1, x1)
                iy1 = max(oby1, y1)
                ix2 = min(obx2, x2)
                iy2 = min(oby2, y2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = (ix2 - ix1) * (iy2 - iy1)
                orig_area  = (obx2 - obx1) * (oby2 - oby1)
                if orig_area <= 0 or (inter_area / orig_area) < MIN_VISIBILITY:
                    continue

                ncx = ((ix1 + ix2) / 2 - x1) / side
                ncy = ((iy1 + iy2) / 2 - y1) / side
                nbw = (ix2 - ix1) / side
                nbh = (iy2 - iy1) / side

                ncx = min(max(ncx, 0.0), 1.0)
                ncy = min(max(ncy, 0.0), 1.0)
                nbw = min(nbw, 1.0)
                nbh = min(nbh, 1.0)

                new_labels.append(
                    f"{int(o_cls)} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}"
                )

            if not new_labels:
                continue

            patch_filename = f"adaptive_{cls}_{stem}_{i}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, 'images', patch_filename), patch)
            with open(os.path.join(OUT_DIR, 'labels',
                      patch_filename.replace('.jpg', '.txt')), 'w') as f_out:
                f_out.write("\n".join(new_labels))

            if stem not in processed_centers:
                processed_centers[stem] = []
            processed_centers[stem].append((cx_px, cy_px))
            patches_created[cls] += 1

    print(f"\n{'─'*50}")
    print(f"Parches generados por clase:")
    for cls_id in sorted(patches_created.keys()):
        name = ID_TO_NAME.get(cls_id, str(cls_id))
        print(f"  Clase {cls_id} ({name:<6}): {patches_created[cls_id]:,} parches")
    print(f"  TOTAL: {sum(patches_created.values()):,}")


if __name__ == "__main__":
    main()