"""
Análisis Exploratorio de Datos (EDA) para el dataset de detección de
rasgos anatómicos en imágenes macroscópicas de madera.

Genera las siguientes figuras en la carpeta OUTPUT_DIR:
  01_distribucion_instancias_por_clase.png  — distribución total por rasgo
  02_distribucion_por_split.png             — train / val / test por rasgo
  03_distribucion_especies.png              — imágenes por especie (cola larga)
  04_tamanio_medio_por_clase.png            — ancho, alto y área media por clase
  05_distribucion_areas_por_split.png       — boxplot de áreas con umbral de ruido
  06_anotaciones_residuales.png             — histograma de áreas pequeñas

Uso:
  python eda_maderas.py --dataset /ruta/al/dataset

El dataset debe tener la estructura estándar de YOLO:
  dataset/
    train/
      images/  labels/
    valid/
      images/  labels/
    test/
      images/  labels/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CLASS_MAP = {
    0: '8',    1: '56',  2: '58',  3: '79',  4: '81',
    5: '82',   6: '83',  7: '84',  8: '85',  9: '86',
    10: '89', 11: '10', 12: 'V1', 13: 'radio'
}
CLASS_ORDER = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

IMG_WIDTH  = 1500
IMG_HEIGHT = 1500
SPLITS     = ['train', 'valid', 'test']
NOISE_THRESHOLD_PX2 = 15   # área mínima válida en píxeles²

PALETTE_SPLITS = ['#084594', '#2171b5', '#6baed6']
PALETTE_SINGLE = '#1f77b4'
PALETTE_AREA   = ['#2ca02c', '#ff7f0e', '#9467bd']

# ─────────────────────────────────────────────
# FUNCIONES DE CARGA
# ─────────────────────────────────────────────

def load_labels(dataset_path: str, splits: list) -> pd.DataFrame:
    """
    Lee todos los ficheros .txt de etiquetas YOLO y devuelve un DataFrame
    con columna: split, clase, x_center, y_center, w_norm, h_norm,
                 w_px, h_px, area_px2, filename.
    """
    rows = []
    for split in splits:
        label_dir = Path(dataset_path) / split / 'labels'
        if not label_dir.exists():
            print(f"  [AVISO] No se encuentra: {label_dir}")
            continue
        files = list(label_dir.glob('*.txt'))
        for fpath in tqdm(files, desc=f"  Cargando {split}", leave=False):
            with open(fpath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid    = int(parts[0])
                    xc     = float(parts[1])
                    yc     = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    w_px   = w_norm * IMG_WIDTH
                    h_px   = h_norm * IMG_HEIGHT
                    rows.append({
                        'split':    split,
                        'clase':    CLASS_MAP.get(cid, str(cid)),
                        'x_center': xc,
                        'y_center': yc,
                        'w_norm':   w_norm,
                        'h_norm':   h_norm,
                        'w_px':     w_px,
                        'h_px':     h_px,
                        'area_px2': w_px * h_px,
                        'filename': fpath.stem,
                    })
    return pd.DataFrame(rows)


def load_species(dataset_path: str, split: str = 'train') -> pd.Series:
    """
    Infiere el nombre de especie desde el nombre del fichero de imagen.
    Asume que el nombre sigue el patrón: ESPECIE-NUM.jpg
    Devuelve una Series con el conteo de imágenes por especie.
    """
    img_dir = Path(dataset_path) / split / 'images'
    if not img_dir.exists():
        return pd.Series(dtype=int)
    species = []
    for f in img_dir.glob('*'):
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            # toma todo lo que hay antes del último guión + número
            parts = f.stem.rsplit('-', 1)
            species.append(parts[0] if len(parts) == 2 else f.stem)
    return pd.Series(species).value_counts()


# ─────────────────────────────────────────────
# FUNCIONES DE VISUALIZACIÓN
# ─────────────────────────────────────────────

def plot_instances_per_class(df: pd.DataFrame, out_dir: Path):
    """Figura 01: distribución total de instancias por clase (log)."""
    totals = (
        df.groupby('clase')['clase']
        .count()
        .reindex(CLASS_ORDER)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(totals.index, totals.values, color=PALETTE_SINGLE,
                  edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_title('Distribución de instancias por clase anatómica\n'
                 '(escala logarítmica)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Clase anatómica (código IAWA)', fontsize=11)
    ax.set_ylabel('Número de bounding boxes (log)', fontsize=11)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # etiquetas sobre las barras
    for bar, val in zip(bars, totals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f'{val:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = out_dir / '01_distribucion_instancias_por_clase.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")


def plot_instances_by_split(df: pd.DataFrame, out_dir: Path):
    """Figura 02: instancias por clase desglosadas por split."""
    pivot = (
        df.groupby(['clase', 'split'])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_ORDER)
    )
    # asegura el orden de columnas
    split_cols = [s for s in SPLITS if s in pivot.columns]
    pivot = pivot[split_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    x     = np.arange(len(CLASS_ORDER))
    width = 0.28
    for i, (split, color) in enumerate(zip(split_cols, PALETTE_SPLITS)):
        ax.bar(x + i * width, pivot[split], width,
               label=split, color=color, edgecolor='black', linewidth=0.5)

    ax.set_title('Distribución total de bounding boxes por split',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Clase anatómica', fontsize=11)
    ax.set_ylabel('Cantidad de bounding boxes', fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_ORDER, rotation=45)
    ax.legend(title='Subconjunto')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    out = out_dir / '02_distribucion_por_split.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")


def plot_species_distribution(dataset_path: str, out_dir: Path):
    """Figura 03: imágenes por especie (cola larga)."""
    species = load_species(dataset_path, split='train')
    if species.empty:
        print("  [AVISO] No se encontraron imágenes para el gráfico de especies.")
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(species)), species.values,
           color=PALETTE_SINGLE, edgecolor='black', linewidth=0.3)
    ax.set_title('Distribución de imágenes por especie (conjunto de entrenamiento)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Especie (ordenadas por frecuencia)', fontsize=11)
    ax.set_ylabel('Número de imágenes', fontsize=11)
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(species.index, rotation=90, fontsize=7)
    ax.axhline(y=species.mean(), color='red', linestyle='--',
               linewidth=1, label=f'Media: {species.mean():.1f}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    out = out_dir / '03_distribucion_especies.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")


def plot_bbox_sizes(df: pd.DataFrame, out_dir: Path):
    """Figura 04: ancho medio, alto medio y área media por clase."""
    stats = (
        df[df['split'] == 'train']
        .groupby('clase')[['w_px', 'h_px', 'area_px2']]
        .mean()
        .reindex(CLASS_ORDER)
        .round(2)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('w_px', 'Ancho medio (px)'),
               ('h_px', 'Alto medio (px)'),
               ('area_px2', 'Área media (px²)')]

    for ax, (col, label) in zip(axes, metrics):
        ax.bar(stats.index, stats[col], color=PALETTE_SINGLE,
               edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Clase', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle('Dimensiones medias de las bounding boxes por clase anatómica\n'
                 '(conjunto de entrenamiento)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = out_dir / '04_tamanio_medio_por_clase.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")

    # También imprime la tabla en consola
    print("\n  Tabla de dimensiones medias por clase (entrenamiento):")
    print(f"  {'Clase':>8} {'Instancias':>12} {'Ancho px':>10} "
          f"{'Alto px':>10} {'Área px²':>12}")
    print("  " + "-" * 56)
    for clase in CLASS_ORDER:
        if clase not in stats.index:
            continue
        n = len(df[(df['split'] == 'train') & (df['clase'] == clase)])
        row = stats.loc[clase]
        print(f"  {clase:>8} {n:>12,} {row['w_px']:>10.2f} "
              f"{row['h_px']:>10.2f} {row['area_px2']:>12.2f}")


def plot_area_distribution(df: pd.DataFrame, out_dir: Path):
    """Figura 05: distribución de áreas por split con umbral de ruido."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for split, color in zip(SPLITS, PALETTE_AREA):
        sub = df[df['split'] == split]['area_px2']
        if sub.empty:
            continue
        ax.boxplot(sub, positions=[SPLITS.index(split)],
                   widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color='black', linewidth=2),
                   flierprops=dict(marker='.', markersize=2, alpha=0.3))

    ax.set_yscale('log')
    ax.axhline(y=NOISE_THRESHOLD_PX2, color='red', linestyle='--',
               linewidth=1.5, label=f'Umbral de ruido ({NOISE_THRESHOLD_PX2} px²)')
    ax.set_title('Distribución de áreas de anotaciones por subconjunto\n'
                 '(escala logarítmica)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels(SPLITS)
    ax.set_ylabel('Área (px²)', fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    out = out_dir / '05_distribucion_areas_por_split.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")

    # Imprime estadísticas de percentiles
    print("\n  Percentiles de área por subconjunto (px²):")
    print(f"  {'Split':>8} {'Mín':>10} {'P1':>10} {'P5':>10} {'Mediana':>10}")
    print("  " + "-" * 54)
    for split in SPLITS:
        sub = df[df['split'] == split]['area_px2']
        if sub.empty:
            continue
        print(f"  {split:>8} {sub.min():>10.2f} "
              f"{sub.quantile(0.01):>10.2f} "
              f"{sub.quantile(0.05):>10.2f} "
              f"{sub.median():>10.2f}")


def plot_residual_annotations(df: pd.DataFrame, out_dir: Path):
    """Figura 06: histograma de anotaciones por debajo del umbral de ruido."""
    residual = df[df['area_px2'] < 200]   # zoom en la zona de ruido
    if residual.empty:
        print("  No hay anotaciones residuales por debajo de 200 px².")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residual['area_px2'], bins=40, color='#d62728',
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=NOISE_THRESHOLD_PX2, color='black', linestyle='--',
               linewidth=1.5, label=f'Umbral de exclusión ({NOISE_THRESHOLD_PX2} px²)')
    ax.set_title('Distribución de anotaciones pequeñas (posibles residuales)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Área (px²)', fontsize=11)
    ax.set_ylabel('Número de anotaciones', fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    out = out_dir / '06_anotaciones_residuales.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {out.name}")

    n_residual = (df['area_px2'] < NOISE_THRESHOLD_PX2).sum()
    n_total    = len(df)
    print(f"\n  Anotaciones con área < {NOISE_THRESHOLD_PX2} px²: "
          f"{n_residual} / {n_total} ({100*n_residual/n_total:.2f}%)")


# ─────────────────────────────────────────────
# RESUMEN EN CONSOLA
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame, dataset_path: str):
    print("\n" + "="*60)
    print("  RESUMEN DEL DATASET")
    print("="*60)

    for split in SPLITS:
        sub = df[df['split'] == split]
        n_img = len(sub['filename'].unique())
        n_ann = len(sub)
        print(f"  {split:>6}: {n_img:>5} imágenes  |  {n_ann:>7,} anotaciones")

    print(f"\n  Total: {len(df['filename'].unique()):>5} imágenes  "
          f"|  {len(df):>7,} anotaciones")
    print(f"  Clases: {df['clase'].nunique()}")

    train = df[df['split'] == 'train']
    if not train.empty:
        counts = train.groupby('clase').size().reindex(CLASS_ORDER).fillna(0)
        max_c  = counts.idxmax()
        min_c  = counts.idxmin()
        ratio  = counts.max() / max(counts.min(), 1)
        print(f"\n  Clase más frecuente : {max_c} ({int(counts.max()):,} instancias)")
        print(f"  Clase más escasa    : {min_c} ({int(counts.min()):,} instancias)")
        print(f"  Ratio de desequilibrio: {ratio:.0f}:1")

    print("="*60 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EDA para el dataset de detección de rasgos en madera.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Ruta raíz del dataset (contiene train/valid/test).')
    parser.add_argument('--output', type=str, default='EDA_output',
                        help='Carpeta donde guardar las figuras (por defecto: EDA_output).')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset : {args.dataset}")
    print(f"Salida  : {out_dir.resolve()}\n")

    # 1. Cargar etiquetas
    print("Cargando etiquetas...")
    df = load_labels(args.dataset, SPLITS)

    if df.empty:
        print("ERROR: No se cargaron datos. Comprueba la ruta del dataset.")
        return

    # 2. Resumen en consola
    print_summary(df, args.dataset)

    # 3. Generar figuras
    print("Generando figuras...\n")
    plot_instances_per_class(df, out_dir)
    plot_instances_by_split(df, out_dir)
    plot_species_distribution(args.dataset, out_dir)
    plot_bbox_sizes(df, out_dir)
    plot_area_distribution(df, out_dir)
    plot_residual_annotations(df, out_dir)

    print(f"\nEDA completado. Figuras guardadas en: {out_dir.resolve()}")


if __name__ == '__main__':
    main()