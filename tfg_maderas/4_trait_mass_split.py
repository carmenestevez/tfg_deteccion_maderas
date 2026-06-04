"""
Este algoritmo implementa una estrategia de partición para 
dividir el conjunto de datos de maderas en Entrenamiento (80%), Validación (10%) 
y Test (10%). 

Debido al extremo desequilibrio de clases en las anotaciones anatómicas 
(ratio 1790:1) y a la distribución de cola larga de las especies, una división 
aleatoria tradicional dejaría a las clases minoritarias 
críticas (ej. 84, 82, 86) sin representación en validación y test.

Para evitarlo, el algoritmo agrupa los vectores de características por imagen 
o espécimen y los asigna iterativamente al subconjunto que más necesite esa 
distribución específica, minimizando una función de coste 
basada en la escasez del rasgo y la especie.

"""

import csv
import json
import math
import random
from collections import Counter, defaultdict

# Rutas de entrada y salida
INPUT_CSV = r"C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\1parte\relative_feature_vectors.csv"

OUTPUT_WITH_SPLIT   = "relative_vectors_with_split.csv"
OUTPUT_TRAIN_CSV    = "relative_train.csv"
OUTPUT_VAL_CSV      = "relative_val.csv"
OUTPUT_TEST_CSV     = "relative_test.csv"
OUTPUT_SUMMARY_JSON = "relative_split_summary.json"

# Proporciones de los subconjuntos
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 7

# Identificadores de columnas en el CSV
SPECIES_COLUMN = "species_name"
FEATURE_PREFIX = "feat_"
GROUP_COLUMN   = None 

# Pesos de la función de coste para la asignación
TRAIT_MASS_WEIGHT         = 1.00
SPECIES_WEIGHT            = 0.35
SPECIES_TRAIT_MASS_WEIGHT = 0.25


def read_csv_rows(path):
    """Lee el CSV y devuelve una lista de diccionarios."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path, rows, fieldnames):
    """Escribe una lista de diccionarios en un archivo CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def to_float(value):
    """Convierte un valor a float de forma segura."""
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0

def choose_group_key(row, row_index):
    """Determina la clave de agrupación (por defecto, la imagen)."""
    if GROUP_COLUMN and row.get(GROUP_COLUMN):
        return row[GROUP_COLUMN]
    if row.get("image_name"):
        return row["image_name"]
    if row.get("image_id"):
        return row["image_id"]
    return f"row_{row_index}"


def build_groups(rows, feature_columns):
    """
    Agrupa las filas por imagen y calcula la 'masa' total de cada rasgo 
    y especie presente en ese grupo para evaluar su rareza.
    """
    groups = {}

    for i, row in enumerate(rows):
        group_key = choose_group_key(row, i)
        species = str(row.get(SPECIES_COLUMN, "unknown")).strip() or "unknown"

        if group_key not in groups:
            groups[group_key] = {
                "group_key": group_key,
                "rows": [],
                "row_count": 0,
                "token_masses": defaultdict(float),
            }

        group = groups[group_key]
        group["rows"].append(row)
        group["row_count"] += 1
        group["token_masses"][f"species::{species}"] += 1.0

        # Acumular la puntuación de cada rasgo anatómico presente
        for feature in feature_columns:
            score = to_float(row.get(feature, 0.0))
            if score <= 0.0:
                continue
            group["token_masses"][f"trait_mass::{feature}"] += score
            group["token_masses"][f"species_trait_mass::{species}::{feature}"] += score

    return list(groups.values())

def token_weight(token, total_mass):
    """Asigna un peso dinámico inversamente proporcional a la frecuencia del rasgo."""
    base = max(1e-9, total_mass)
    if token.startswith("trait_mass::"):
        return TRAIT_MASS_WEIGHT / base
    if token.startswith("species::"):
        return SPECIES_WEIGHT / base
    if token.startswith("species_trait_mass::"):
        return SPECIES_TRAIT_MASS_WEIGHT / base
    return 0.0

def group_difficulty(group, total_token_masses):
    """
    Calcula la dificultad de un grupo. Las imágenes con rasgos minoritarios 
    reciben una puntuación más alta para ser asignadas primero.
    """
    score = 0.0
    for token, value in group["token_masses"].items():
        score += value * token_weight(token, total_token_masses[token])
    score += 0.05 * group["row_count"]
    return score

def projected_score(split_name, group, split_sizes, split_token_masses, target_sizes, target_token_masses, total_token_masses):
    """
    Calcula la penalización si el grupo se asignara a un split concreto.
    Busca minimizar el error respecto a la distribución ideal esperada.
    """
    new_size = split_sizes[split_name] + group["row_count"]
    size_score = abs(new_size - target_sizes[split_name]) / max(1.0, target_sizes[split_name])

    token_score = 0.0
    for token, value in group["token_masses"].items():
        new_value = split_token_masses[split_name][token] + value
        target_value = target_token_masses[token][split_name]
        token_score += token_weight(token, total_token_masses[token]) * abs(new_value - target_value)

    # Penalización fuerte si sobrepasamos el límite de validación o test
    overflow_penalty = 0.0
    if split_name != "train":
        soft_limit = math.ceil(target_sizes[split_name])
        if new_size > soft_limit:
            overflow_penalty = 5.0 * (new_size - soft_limit)

    return size_score + token_score + overflow_penalty

def assign_groups(groups, split_ratios):
    """
    Realiza la asignación definitiva de cada imagen al mejor subconjunto posible.
    """
    total_rows = sum(group["row_count"] for group in groups)
    target_sizes = {name: total_rows * ratio for name, ratio in split_ratios.items()}

    total_token_masses = defaultdict(float)
    for group in groups:
        for token, value in group["token_masses"].items():
            total_token_masses[token] += value

    target_token_masses = {}
    for token, total_mass in total_token_masses.items():
        target_token_masses[token] = {}
        for split_name, ratio in split_ratios.items():
            target_token_masses[token][split_name] = total_mass * ratio

    # Ordenar grupos por dificultad (los más críticos se asignan primero)
    random.Random(RANDOM_SEED).shuffle(groups)
    groups = sorted(groups, key=lambda g: group_difficulty(g, total_token_masses), reverse=True)

    split_sizes = {name: 0 for name in split_ratios}
    split_token_masses = {name: defaultdict(float) for name in split_ratios}
    split_groups = {name: [] for name in split_ratios}

    for group in groups:
        best_split = None
        best_score = None

        for split_name in split_ratios:
            score = projected_score(
                split_name, group, split_sizes, split_token_masses, 
                target_sizes, target_token_masses, total_token_masses
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        split_groups[best_split].append(group)
        split_sizes[best_split] += group["row_count"]
        for token, value in group["token_masses"].items():
            split_token_masses[best_split][token] += value

    return split_groups, split_sizes

# Resultados

def add_split_column(rows, split_groups):
    """Añade la columna 'split' al CSV original indicando el conjunto asignado."""
    row_to_split = {}
    for split_name, groups in split_groups.items():
        for group in groups:
            for row in group["rows"]:
                row_to_split[id(row)] = split_name

    output_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["split"] = row_to_split[id(row)]
        output_rows.append(new_row)
    return output_rows

def build_summary(rows_with_split, feature_columns):
    """Genera un diccionario con las estadísticas finales de la partición."""
    summary = {
        "total_rows": len(rows_with_split),
        "split_sizes": Counter(row["split"] for row in rows_with_split),
        "species_counts_by_split": defaultdict(Counter),
        "trait_score_sums_by_split": defaultdict(lambda: defaultdict(float)),
        "trait_score_means_by_split": defaultdict(dict),
        "species_trait_score_sums_by_split": defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
    }

    split_row_counts = Counter()

    for row in rows_with_split:
        split_name = row["split"]
        split_row_counts[split_name] += 1

        species = str(row.get(SPECIES_COLUMN, "unknown")).strip() or "unknown"
        summary["species_counts_by_split"][split_name][species] += 1

        for feature in feature_columns:
            score = to_float(row.get(feature, 0.0))
            summary["trait_score_sums_by_split"][split_name][feature] += score
            summary["species_trait_score_sums_by_split"][split_name][species][feature] += score

    for split_name, feature_sums in summary["trait_score_sums_by_split"].items():
        n = max(1, split_row_counts[split_name])
        for feature, total in feature_sums.items():
            summary["trait_score_means_by_split"][split_name][feature] = round(total / n, 6)

    # Formatear el JSON para que sea legible
    clean_summary = {
        "total_rows": summary["total_rows"],
        "split_sizes": dict(summary["split_sizes"]),
        "species_counts_by_split": {k: dict(v) for k, v in summary["species_counts_by_split"].items()},
        "trait_score_sums_by_split": {k: {f: round(v, 6) for f, v in fmap.items()} for k, fmap in summary["trait_score_sums_by_split"].items()},
        "trait_score_means_by_split": {k: {f: round(v, 6) for f, v in fmap.items()} for k, fmap in summary["trait_score_means_by_split"].items()},
        "objective_weights": {
            "TRAIT_MASS_WEIGHT": TRAIT_MASS_WEIGHT,
            "SPECIES_WEIGHT": SPECIES_WEIGHT,
            "SPECIES_TRAIT_MASS_WEIGHT": SPECIES_TRAIT_MASS_WEIGHT,
        },
    }
    return clean_summary

def main():
    print("Iniciando división estratificada (Trait-Mass Split)...")
    
    rows = read_csv_rows(INPUT_CSV)
    if not rows:
        raise ValueError("Error: El archivo CSV de entrada está vacío.")

    feature_columns = [col for col in rows[0].keys() if col.startswith(FEATURE_PREFIX)]

    split_ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
    if abs(sum(split_ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Error: La suma de TRAIN, VAL y TEST debe ser exactamente 1.0")

    # Ejecución del algoritmo
    groups = build_groups(rows, feature_columns)
    split_groups, split_sizes = assign_groups(groups, split_ratios)
    rows_with_split = add_split_column(rows, split_groups)

    fieldnames = list(rows_with_split[0].keys())
    
    # Guardado de archivos
    write_csv_rows(OUTPUT_WITH_SPLIT, rows_with_split, fieldnames)
    write_csv_rows(OUTPUT_TRAIN_CSV, [r for r in rows_with_split if r["split"] == "train"], fieldnames)
    write_csv_rows(OUTPUT_VAL_CSV, [r for r in rows_with_split if r["split"] == "val"], fieldnames)
    write_csv_rows(OUTPUT_TEST_CSV, [r for r in rows_with_split if r["split"] == "test"], fieldnames)

    summary = build_summary(rows_with_split, feature_columns)
    summary["assigned_split_sizes"] = split_sizes

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nProceso completado:")
    print(f"   - Total de instancias procesadas: {len(rows_with_split)}")
    print(f"   - Archivo general guardado en: {OUTPUT_WITH_SPLIT}")
    print(f"   - Resumen de partición guardado en: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()