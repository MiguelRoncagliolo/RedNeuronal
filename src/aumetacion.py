import random
import csv
import json
import os

# --- 1. Base de Datos Ficticia (Puedes expandir esto) ---
CALLES = [
    "1 Norte", "2 Norte", "5 Oriente", "Quillota", "Arlegui", "Valparaíso",
    "Libertad", "San Martín", "Alvarez", "Viana", "Von Schroeder", "Ecuador",
    "Agua Santa", "Errázuriz", "Blanco", "Prat", "Serrano", "Montaña",
    "Nueva Libertad", "Av. Marina", "Los Castaños", "Jackson"
]
NUMEROS = [str(random.randint(10, 2500)) for _ in range(100)]
PASAJEROS = [str(i) for i in range(1, 9)]
FECHAS = [
    "2025-11-25", "2025-11-26", "mañana", "hoy", "el próximo martes",
    "el 30 de diciembre", "para el 5/10/2026", "el 15/01/2026"
]
HORAS = [
    "12:00", "12:00 hrs", "8:30", "09:15", "18:00", "a las 10 de la mañana",
    "a mediodía", "para las 3 de la tarde", "a las 19.45"
]
PREFIJOS_CALLE = ["Calle", "Av.", "Avenida", "Pasaje", "Pje.", "Pj.", "Avda.", ""]
PREFIJOS_NUMERO = ["#", "N°", "nro.", "numero", "casa", ""]

# --- 2. Función de Generación de Direcciones ---
def generar_direccion_aleatoria():
    calle = random.choice(CALLES)
    numero = random.choice(NUMEROS)
    prefijo_calle = random.choice(PREFIJOS_CALLE + [""] * 3)
    prefijo_numero = random.choice(PREFIJOS_NUMERO + [""] * 2)
    partes = []
    if prefijo_calle:
        partes.append(prefijo_calle)
    partes.append(calle)
    if prefijo_numero:
        partes.append(prefijo_numero)
    partes.append(numero)
    direccion_final = " ".join(partes)
    return " ".join(direccion_final.split())

# --- 3. Plantillas de Frases ---
PLANTILLAS = [
    "Registré tu traslado desde {ORIGEN} hasta {DESTINO} para el {FECHA} a las {HORA} para {PAX} pasajeros solo ida.",
    "Hola, necesito un móvil desde {ORIGEN} para {DESTINO}, el día {FECHA} a la(s) {HORA}. Somos {PAX}.",
    "Agendado. Origen: {ORIGEN}. Destino: {DESTINO}. Fecha: {FECHA} {HORA}. Pasajeros: {PAX}.",
    "Confirmado tu viaje. Te pasamos a buscar por {ORIGEN} y te dejamos en {DESTINO} el {FECHA} a las {HORA}. ({PAX} personas).",
    "Tu cotización desde {ORIGEN} a {DESTINO} está lista. ({FECHA}, {HORA}, {PAX} pax).",
    "OK, vamos de {ORIGEN} para {DESTINO}. Fecha {FECHA} hora {HORA}, {PAX} pasajeros.",
    "Necesito ir a {DESTINO} desde {ORIGEN} el {FECHA} a las {HORA}. Seríamos {PAX}."
]

# --- 4. Función Principal de Generación y Guardado ---

def generar_y_guardar_dataset(cantidad_de_ejemplos):
    
    # Define las rutas de salida, subiendo un nivel desde 'src' a la carpeta 'data'
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Directorio de aumetacion.py (src)
    data_dir = os.path.join(base_dir, '..', 'data')       # Directorio data/
    
    # Asegurarse de que el directorio 'data' exista
    os.makedirs(data_dir, exist_ok=True)
    
    # Nombres de los archivos de salida
    csv_path = os.path.join(data_dir, 'dataset_aumentado.csv')
    jsonl_path = os.path.join(data_dir, 'dataset_aumentado.jsonl')
    
    print(f"Generando {cantidad_de_ejemplos} ejemplos...")
    
    # Usamos 'with' para asegurarnos de que los archivos se cierren correctamente
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
        
        # --- Configuración del CSV ---
        # Definimos las columnas
        columnas = ['texto', 'origen', 'destino', 'fecha', 'hora', 'pasajeros']
        csv_writer = csv.DictWriter(csvfile, fieldnames=columnas)
        csv_writer.writeheader() # Escribir la cabecera
        
        # --- Bucle de Generación ---
        for _ in range(cantidad_de_ejemplos):
            plantilla = random.choice(PLANTILLAS)
            
            # Generamos los datos de las entidades
            origen = generar_direccion_aleatoria()
            destino = generar_direccion_aleatoria()
            while origen == destino: # Evitar que origen y destino sean iguales
                destino = generar_direccion_aleatoria()
                
            fecha = random.choice(FECHAS)
            hora = random.choice(HORAS)
            pax = random.choice(PASAJEROS)
            
            # Creamos la frase final
            frase = plantilla.format(ORIGEN=origen, DESTINO=destino, FECHA=fecha, HORA=hora, PAX=pax)
            
            # --- Guardar en CSV ---
            fila_csv = {
                'texto': frase,
                'origen': origen,
                'destino': destino,
                'fecha': fecha,
                'hora': hora,
                'pasajeros': pax
            }
            csv_writer.writerow(fila_csv)
            
            # --- Guardar en JSONL ---
            # (Cada línea es un objeto JSON independiente)
            fila_jsonl = {
                'texto': frase,
                'entidades': {
                    'origen': origen,
                    'destino': destino,
                    'fecha': fecha,
                    'hora': hora,
                    'pasajeros': pax
                }
            }
            # ensure_ascii=False para guardar tildes y 'ñ' correctamente
            jsonlfile.write(json.dumps(fila_jsonl, ensure_ascii=False) + '\n')

    print("¡Generación completada! ✅")
    print(f"Datos guardados en: {csv_path}")
    print(f"Datos guardados en: {jsonl_path}")

# --- Ejecución ---
if __name__ == "__main__":
    NUEVOS_EJEMPLOS = 5000 # <-- Aumenta este número (5000, 10000, 50000)
    generar_y_guardar_dataset(NUEVOS_EJEMPLOS)