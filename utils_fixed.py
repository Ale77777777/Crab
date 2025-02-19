#utils_fixed.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import re
from shapely.geometry import Polygon, LineString

def load_data(file_path):
    """Carga un archivo CSV con las coordenadas necesarias."""
    data = pd.read_csv(file_path, header=[0, 1, 2])

    center_coords = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'Center', 'x')].to_frame()
    center_coords['y'] = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'Center', 'y')]
    center_coords['likelihood'] = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'Center', 'likelihood')]

    rightp_coords = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'RightP', 'x')].to_frame()
    rightp_coords['y'] = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'RightP', 'y')]

    leftp_coords = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'LeftP', 'x')].to_frame()
    leftp_coords['y'] = data[('DLC_Resnet50_Modelo5AJJan15shuffle1_snapshot_190', 'LeftP', 'y')]

    # Renombrar columnas
    center_coords.columns = ['x', 'y', 'likelihood']
    rightp_coords.columns = ['x', 'y']
    leftp_coords.columns = ['x', 'y']

    return data, center_coords, rightp_coords, leftp_coords

def parse_roi_filename(roi_filename):
    match = re.search(r'-(\d+)-(\d+)\.roi$', roi_filename)
    if match:
        center_y = int(match.group(1))  # Primero es Y
        center_x = int(match.group(2))  # Luego es X
        return center_x, center_y
    return None, None

def calculate_displacement_frame_by_frame(interval_data):
    """
    Calcula el desplazamiento cuadro por cuadro y la inmovilidad.
    """
    interval_data = interval_data.copy()
    interval_data['displacement'] = np.sqrt(
        np.square(interval_data['x'].diff()) + np.square(interval_data['y'].diff())
    )
    interval_data['displacement'] = interval_data['displacement'].fillna(0)

    # Detectar inmovilidad
    immobility = (interval_data['displacement'] < 1).astype(int)
    immobility_periods = []
    count = 0
    for val in immobility:
        if val == 1:
            count += 1
        else:
            if count >= 3:
                for _ in range(count):
                    immobility_periods.append(1)
            else:
                for _ in range(count):
                    immobility_periods.append(0)
            immobility_periods.append(0)
            count = 0
    if count >= 3:
        for _ in range(count):
            immobility_periods.append(1)
    else:
        for _ in range(count):
            immobility_periods.append(0)
    
    # Asegurar la longitud de immobility_periods
    while len(immobility_periods) < len(interval_data):
        immobility_periods.append(0)

    interval_data['immobility'] = immobility_periods[:len(interval_data)]

    return interval_data

def generate_heatmap_with_quadrants(center_coords, rightp_coords, leftp_coords, prefix, output_dir, center_x, center_y, camera_width, camera_height):
    """
    Genera mapas de calor basados en la presencia del triángulo y superpone un círculo dividido en cuatro cuadrantes a 45 grados.
    """
    if center_x is None or center_y is None:
        raise ValueError("No se pudo extraer correctamente el centro del ROI")

    heatmap = np.zeros((camera_height, camera_width))

    for i in range(len(center_coords)):
        triangle_points = np.array([
            [center_coords['x'].iloc[i], center_coords['y'].iloc[i]],
            [rightp_coords['x'].iloc[i], rightp_coords['y'].iloc[i]],
            [leftp_coords['x'].iloc[i], leftp_coords['y'].iloc[i]]
        ])

        

        grid_x, grid_y = np.meshgrid(np.arange(camera_width), np.arange(camera_height))
        points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
        path = Path(triangle_points)
        mask = path.contains_points(points).reshape(camera_height, camera_width)
        heatmap += mask

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin='lower', cmap='viridis', extent=[0, camera_width, 0, camera_height])
    plt.colorbar(label='Frecuencia')

    # Dibujar círculo dividido en cuadrantes
    radius = min(camera_width, camera_height) // 2  # Ajuste del radio del círculo
    circle = plt.Circle((center_x, center_y), radius, color='black', fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    # Dibujar líneas a 45 grados para dividir el círculo en cuadrantes
    plt.plot([center_x - radius * np.sqrt(2)/2, center_x + radius * np.sqrt(2)/2], 
             [center_y + radius * np.sqrt(2)/2, center_y - radius * np.sqrt(2)/2], color='black', linestyle='-', linewidth=2)
    plt.plot([center_x - radius * np.sqrt(2)/2, center_x + radius * np.sqrt(2)/2], 
             [center_y - radius * np.sqrt(2)/2, center_y + radius * np.sqrt(2)/2], color='black', linestyle='-', linewidth=2)

    # Etiquetas de cuadrantes correctamente posicionadas
    plt.text(center_x, center_y + radius // 2, 'Q3', color='white', fontsize=12, ha='center', va='center')  # Arriba (superior)
    plt.text(center_x + radius // 2, center_y, 'Q2', color='white', fontsize=12, ha='center', va='center')  # Derecha
    plt.text(center_x, center_y - radius // 2, 'Q1', color='white', fontsize=12, ha='center', va='center')  # Abajo
    plt.text(center_x - radius // 2, center_y, 'Q4', color='white', fontsize=12, ha='center', va='center')  # Izquierda

    # Marcar el centro indicado por los datos .roi
    plt.scatter(center_x, center_y, color='blue', s=100, label='Center ROI')

    plt.xlim(0, camera_width)
    plt.ylim(0, camera_height)
    plt.gca().invert_yaxis()  # Invertir eje Y para coincidir con la imagen
    plt.legend(loc='upper right')

    output_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Heatmap guardado en: {output_path}")
    
    return calculate_quadrant_activity(heatmap, center_x, center_y)

def calculate_quadrant_activity(heatmap, center_x, center_y):
    """
    Calcula la actividad en cada cuadrante directamente desde el heatmap,
    utilizando ángulos para cuadrantes divididos en forma de 'X'.
    """
    height, width = heatmap.shape
    quadrant_activity = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    radius = min(width, height) // 2  # Ajuste del radio del círculo

    Y, X = np.ogrid[:height, :width]
    # Invertir Y para que coincida con la visualización del heatmap
    Y = height - Y

    # Coordenadas relativas al centro
    X_rel = X - center_x
    Y_rel = Y - center_y

    # Calcular distancia desde el centro
    distance = np.sqrt(X_rel**2 + Y_rel**2)

    # Máscara para limitar al círculo
    mask_circle = distance <= radius

    # Calcular ángulo en grados
    angles = np.degrees(np.arctan2(Y_rel, X_rel))
    angles = angles % 360  # Asegurar que los ángulos estén entre 0 y 360

    # Restaurar la lógica original de los cuadrantes
    quadrant_masks = {
        'Q1': (angles >= 45) & (angles < 135),
        'Q2': ((angles >= 0) & (angles < 45)) | ((angles >= 315) & (angles < 360)),
        'Q3': (angles >= 225) & (angles < 315),
        'Q4': (angles >= 135) & (angles < 225)
    }

    for q in quadrant_activity:
        mask = mask_circle & quadrant_masks[q]
        quadrant_activity[q] = np.sum(heatmap[mask])

    return quadrant_activity

def standardize_displacement(displacement_series, rightp_coords, leftp_coords):
    """
    Estandariza el desplazamiento dividiendo por la media de la distancia entre rightp_coords y leftp_coords.
    """
    # Calcular la distancia entre los puntos de referencia en cada cuadro
    distances = np.sqrt((rightp_coords['x'] - leftp_coords['x'])**2 + (rightp_coords['y'] - leftp_coords['y'])**2)
    
    # Calcular la distancia media
    mean_body_size = np.mean(distances)
        
    # Imprimir información para depuración
    print(f"Distancias entre puntos de referencia: {distances}")
    print(f"Distancia media (mean_body_size): {mean_body_size}")
    
    # Estandarizar el desplazamiento si la distancia media es mayor que cero
    if mean_body_size > 0:
        standardized_displacement = displacement_series / mean_body_size
        print(f"Desplazamiento estandarizado: {standardized_displacement}")
        return standardized_displacement
    else:
        print("La distancia media es cero o negativa, no se puede estandarizar el desplazamiento.")
        return displacement_series
    

def format_filename(original_filename, phase, filetype):
    """
    Formatea el nombre del archivo para frame a frame y heatmaps.
    """
    base_name = original_filename.split('_DLC_')[0].strip()
    return f"{base_name}_DLC_{phase}_{filetype}"
