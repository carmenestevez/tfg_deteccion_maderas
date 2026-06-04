"""
Generación de las bounding boxes a partir de las anotaciones del CVAT.

"""
import os

# Ruta a la carpeta donde están los .txt
carpeta_origen = r"C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\images\labels\train" 
# Aquí se guardarán los nuevos .txt limpios
carpeta_destino = r"C:\Users\prestamo_admin\Desktop\uni\maderas\proyecto-maderas-main\1parte\labels_cajas"    

os.makedirs(carpeta_destino, exist_ok=True)

for archivo in os.listdir(carpeta_origen):
    if not archivo.endswith(".txt"):
        continue
        
    ruta_origen = os.path.join(carpeta_origen, archivo)
    ruta_dest = os.path.join(carpeta_destino, archivo)
    
    with open(ruta_origen, 'r') as f_in, open(ruta_dest, 'w') as f_out:
        for linea in f_in:
            partes = linea.strip().split()
            
            # Si tiene menos de 5 elementos, no es una etiqueta válida
            if len(partes) < 5:
                continue 
                
            id_clase = partes[0]
            
            # Convertimos el resto de números a decimales
            coordenadas = [float(x) for x in partes[1:]]
            
            # Sacamos las X (posiciones pares) y las Y (posiciones impares)
            xs = coordenadas[0::2]
            ys = coordenadas[1::2]
            
            # Calculamos los límites extremos del polígono
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Calculamos los datos que pide YOLO Detection (Centro X, Centro Y, Ancho, Alto)
            centro_x = (min_x + max_x) / 2.0
            centro_y = (min_y + max_y) / 2.0
            ancho = max_x - min_x
            alto = max_y - min_y
            
            # Escribimos la nueva línea en el nuevo archivo (con 6 decimales de precisión)
            f_out.write(f"{id_clase} {centro_x:.6f} {centro_y:.6f} {ancho:.6f} {alto:.6f}\n")

print("Conversión completada.")