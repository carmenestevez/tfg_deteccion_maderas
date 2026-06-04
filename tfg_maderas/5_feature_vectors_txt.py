"""
Procesa las etiquetas YOLO (cajas delimitadoras) y las cruza con 
los metadatos de las especies de madera para generar un vector de características 
relativo por imagen. 

En lugar de contar simplemente cuántos rasgos (ej. poros o parénquimas) hay, 
calcula una puntuación ponderada (score) basada en tres niveles de rareza:
  1. Local: Frecuencia del rasgo dentro de la propia imagen.
  2. Especie: Frecuencia del rasgo en comparación con el máximo de su especie.
  3. Global: Frecuencia del rasgo respecto a todo el dataset.

El CSV resultante (relative_feature_vectors.csv) sirve como entrada directa 
para el algoritmo de división estratificada (Trait-Mass Split).

"""

import csv
import json
import os
import pandas as pd
from collections import Counter, defaultdict

# 1. Ruta a la carpeta donde estén los .txt de cajas
INPUT_LABELS_DIR = r"C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\1parte\Dataset_Filtrado\labels"

# 2. Ruta a tu CSV original con los nombres de las especies
METADATA_CSV = r"C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\tabla_rasgos_por_imagen1(tabla_rasgos_por_imagen1).csv"

# Archivos de salida
OUTPUT_CSV = "relative_feature_vectors.csv"
OUTPUT_SCHEMA_JSON = "relative_feature_vectors.schema.json"

# Mapeo de IDs de YOLO (0-13) a nombres de rasgos
CLASS_MAP = {
    '0': '8', '1': '56', '2': '58', '3': '79', '4': '81', '5': '82',
    '6': '83', '7': '84', '8': '85', '9': '86', '10': '89',
    '11': '10', '12': 'V1', '13': 'radio'
}

# Pesos originales
LOCAL_WEIGHT = 0.50
SPECIES_WEIGHT = 0.20
GLOBAL_WEIGHT = 0.30
MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT = 5
WRITE_COMPONENT_COLUMNS = True

def safe_feature_name(label):
    return f"feat_rasgo_{label}"

def load_species_mapping():
    df = pd.read_csv(METADATA_CSV, sep=';', header=1)
    mapping = {}
    for _, row in df.iterrows():
        img = str(row['Imagen']).strip()
        if img != 'nan':
            # Quitamos la extensión para que coincida con el txt
            base_name = os.path.splitext(img)[0]
            mapping[base_name] = str(row['Especie']).strip()
    return mapping

def read_yolo_txt_folder(folder_path, species_mapping):
    rows = []
    observed_labels = set()
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
            
        base_name = os.path.splitext(filename)[0]
        species_name = species_mapping.get(base_name, "Desconocida")
        
        if species_name == "Desconocida":
            continue # Saltamos si no sabemos la especie
            
        label_counts = Counter()
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    trait_name = CLASS_MAP.get(class_id, f"clase_{class_id}")
                    label_counts[trait_name] += 1
                    observed_labels.add(trait_name)
                    
        rows.append({
            "source_txt": filename,
            "task_name": "YOLO_Dataset",
            "task_code": "001",
            "species_name": species_name,
            "image_id": base_name,
            "image_name": base_name + ".jpg",
            "width": 1500,  # Fijo por tus imágenes
            "height": 1500,
            "label_counts": dict(label_counts),
            "total_instances": sum(label_counts.values())
        })
        
    return rows, list(CLASS_MAP.values()), list(observed_labels)

def compute_normalization_stats(rows, labels):
    global_trait_max = {label: 0 for label in labels}
    species_trait_max = defaultdict(int)
    species_image_counts = Counter()

    for row in rows:
        species = row["species_name"]
        species_image_counts[species] += 1

        for label in labels:
            count = int(row["label_counts"].get(label, 0))
            if count > global_trait_max[label]:
                global_trait_max[label] = count
            key = (species, label)
            if count > species_trait_max[key]:
                species_trait_max[key] = count

    return global_trait_max, species_trait_max, species_image_counts

def species_reliability(species_name, species_image_counts):
    n = species_image_counts.get(species_name, 0)
    if MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT <= 0:
        return 1.0
    return min(1.0, n / float(MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT))

def compute_score_components(count, total_instances, species_max, global_max):
    if count <= 0: return 0.0, 0.0, 0.0
    local_c = count / float(total_instances) if total_instances > 0 else 0.0
    species_c = count / float(species_max) if species_max > 0 else 0.0
    global_c = count / float(global_max) if global_max > 0 else 0.0
    return local_c, species_c, global_c

def build_output_rows(rows, labels, global_trait_max, species_trait_max, species_image_counts):
    output_rows = []
    for row in rows:
        species = row["species_name"]
        total_instances = row["total_instances"]

        reliability = species_reliability(species, species_image_counts)
        eff_species_weight = SPECIES_WEIGHT * reliability
        eff_global_weight = GLOBAL_WEIGHT + SPECIES_WEIGHT * (1.0 - reliability)

        output_row = {
            "source_txt": row["source_txt"],
            "species_name": species,
            "species_reliability": round(reliability, 6),
            "image_name": row["image_name"],
            "total_instances": total_instances,
        }

        for label in labels:
            feature_name = safe_feature_name(label)
            count = int(row["label_counts"].get(label, 0))
            species_max = int(species_trait_max.get((species, label), 0))
            global_max = int(global_trait_max.get(label, 0))

            local_c, species_c, global_c = compute_score_components(count, total_instances, species_max, global_max)
            score = (LOCAL_WEIGHT * local_c) + (eff_species_weight * species_c) + (eff_global_weight * global_c)
            if count <= 0: score = 0.0

            output_row[feature_name] = round(min(max(score, 0.0), 1.0), 6)

            if WRITE_COMPONENT_COLUMNS:
                output_row[f"count_{feature_name}"] = count
                output_row[f"local_{feature_name}"] = round(local_c, 6)
                output_row[f"species_{feature_name}"] = round(species_c, 6)
                output_row[f"global_{feature_name}"] = round(global_c, 6)

        output_rows.append(output_row)
    return output_rows

def write_output_csv(output_rows, labels):
    if not output_rows: return
    fieldnames = list(output_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

def main():
    species_map = load_species_mapping()
    all_rows, meta_labels, observed_labels = read_yolo_txt_folder(INPUT_LABELS_DIR, species_map)
    
    labels = sorted(set(meta_labels) | set(observed_labels), key=str)
    global_max, species_max, sp_counts = compute_normalization_stats(all_rows, labels)
    output_rows = build_output_rows(all_rows, labels, global_max, species_max, sp_counts)

    write_output_csv(output_rows, labels)
    
    print(f"Imágenes procesadas: {len(output_rows)}")
    print(f"CSV guardado en: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()