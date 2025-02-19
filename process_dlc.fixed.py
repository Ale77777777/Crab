#process_dlc.fixed.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_fixed import (
    load_data, calculate_displacement_frame_by_frame,
    generate_heatmap_with_quadrants, 
    calculate_quadrant_activity, parse_roi_filename
)

def detect_immobility(displacement_series):
    immobility = (displacement_series < 1).astype(int)
    immobility_periods = []
    count = 0
    for val in immobility:
        if val == 1:
            count += 1
        else:
            if count >= 3:
                immobility_periods.extend([1] * count)
            else:
                immobility_periods.extend([0] * count)
            immobility_periods.append(0)
            count = 0
    if count >= 3:
        immobility_periods.extend([1] * count)
    else:
        immobility_periods.extend([0] * count)
    
    # Asegurar la longitud de immobility_periods
    while len(immobility_periods) < len(displacement_series):
        immobility_periods.append(0)
    
    return immobility_periods[:len(displacement_series)]

def process_files(input_dir, output_dir, roi_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_data = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Procesando archivo: {file_name}")

            # Cargar datos
            data, center_coords, rightp_coords, leftp_coords = load_data(file_path)
            if center_coords.empty:
                print(f"⚠️ Archivo {file_name} no tiene datos válidos. Saltando...")
                continue

            # Buscar archivo ROI correspondiente
            animal_id = file_name.split()[0]  # Extraer ID del animal
            roi_file = [f for f in os.listdir(roi_dir) if animal_id in f]
            if not roi_file:
                print(f"⚠️ Archivo ROI no encontrado para {animal_id}. Saltando...")
                continue

            roi_path = os.path.join(roi_dir, roi_file[0])
            try:
                center_x, center_y = parse_roi_filename(roi_file[0])
                center_x = int(center_x)
                center_y = int(center_y)
            except ValueError as e:
                print(f"⚠️ Error al procesar ROI para {file_name}: {e}")
                continue

            # Identificar índices donde la media de likelihood consecutivos es menor a 0.4
            low_likelihood_indices = [
                i for i in range(len(center_coords) - 1)
                if (center_coords['likelihood'].iloc[i] + center_coords['likelihood'].iloc[i + 1]) / 2 < 0.4
            ]

            if low_likelihood_indices:
                pos_start_idx = max(0, low_likelihood_indices[0] - 91)
                pos_end_idx = low_likelihood_indices[0]
            else:
                pos_start_idx = max(0, len(center_coords) - 92)
                pos_end_idx = len(center_coords)

            pre_start_idx = max(0, pos_start_idx - 92)
            pre_end_idx = pos_start_idx

            print(f"📌 Índices Pre: {pre_start_idx} - {pre_end_idx}")
            print(f"📌 Índices Pos: {pos_start_idx} - {pos_end_idx}")

            # Extraer intervalos Pre y Pos
            pre_interval = center_coords.iloc[pre_start_idx:pre_end_idx][['x', 'y']]
            pos_interval = center_coords.iloc[pos_start_idx:pos_end_idx][['x', 'y']]

            # Calcular métricas Pre y Pos
            pre_displacement_data = calculate_displacement_frame_by_frame(pre_interval)
            pos_displacement_data = calculate_displacement_frame_by_frame(pos_interval)

            # Detectar inmovilidad
            pre_displacement_data['immobility'] = detect_immobility(pre_displacement_data['displacement'])
            pos_displacement_data['immobility'] = detect_immobility(pos_displacement_data['displacement'])

            print(f"📊 Pre-Desplazamiento Total: {pre_displacement_data['displacement'].sum()}")
            print(f"📊 Pos-Desplazamiento Total: {pos_displacement_data['displacement'].sum()}")

            # Guardar datos frame a frame
            pre_csv_path = os.path.join(output_dir, f"{file_name.split('.')[0]}_Pre_frame_data.csv")
            pos_csv_path = os.path.join(output_dir, f"{file_name.split('.')[0]}_Pos_frame_data.csv")

            pre_displacement_data.to_csv(pre_csv_path, index=False)
            pos_displacement_data.to_csv(pos_csv_path, index=False)

            print(f"✅ Guardado: {pre_csv_path}")
            print(f"✅ Guardado: {pos_csv_path}")

            # Generar mapas de calor y calcular actividad por cuadrantes desde el heatmap
            pre_quadrant_data = generate_heatmap_with_quadrants(
                center_coords.iloc[pre_start_idx:pre_end_idx],
                rightp_coords.iloc[pre_start_idx:pre_end_idx],
                leftp_coords.iloc[pre_start_idx:pre_end_idx],
                prefix=f"{file_name.split('.')[0]}_Pre",
                output_dir=output_dir,
                center_x=center_x,
                center_y=center_y,
                camera_width=640,  # Dimensiones de la cámara
                camera_height=360
            )

            pos_quadrant_data = generate_heatmap_with_quadrants(
                center_coords.iloc[pos_start_idx:pos_end_idx],
                rightp_coords.iloc[pos_start_idx:pos_end_idx],
                leftp_coords.iloc[pos_start_idx:pos_end_idx],
                prefix=f"{file_name.split('.')[0]}_Pos",
                output_dir=output_dir,
                center_x=center_x,
                center_y=center_y,
                camera_width=640,  # Dimensiones de la cámara
                camera_height=360
            )

            # Agregar al resumen global
            summary_data.append({
                "Archivo": file_name,
                "Pre_Desplazamiento_Total": pre_displacement_data["displacement"].sum(),
                "Pos_Desplazamiento_Total": pos_displacement_data["displacement"].sum(),
                "Pre_Inmovilidad_Total": pre_displacement_data["immobility"].sum(),
                "Pos_Inmovilidad_Total": pos_displacement_data["immobility"].sum(),
                "Pre_Q1": pre_quadrant_data['Q1'],
                "Pre_Q2": pre_quadrant_data['Q2'],
                "Pre_Q3": pre_quadrant_data['Q3'],
                "Pre_Q4": pre_quadrant_data['Q4'],
                "Pos_Q1": pos_quadrant_data['Q1'],
                "Pos_Q2": pos_quadrant_data['Q2'],
                "Pos_Q3": pos_quadrant_data['Q3'],
                "Pos_Q4": pos_quadrant_data['Q4']
            })

    # Guardar resumen global
    summary_file = os.path.join(output_dir, "resumen_global.csv")
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"📁 Resumen global guardado en: {summary_file}")

    print("✅ ¡Procesamiento completado!")

# Directorios
input_dir = r"F:\ALE LAB\VAL01\TS\camaratres"
output_dir = r"F:\ALE LAB\VAL01\TS\resultados"
roi_dir = r"F:\ALE LAB\VAL01\TS\camaratres\RoiSet"

process_files(input_dir, output_dir, roi_dir)
