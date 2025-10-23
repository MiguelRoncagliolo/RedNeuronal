import random
import uuid # Para generar IDs únicos

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

# --- 4. Función Principal de Generación de Líneas ---

def generar_lineas_csv(cantidad_de_ejemplos):
    lineas_generadas = []
    
    # Encabezado (solo para referencia, no se imprimirá si solo quieres las líneas de datos)
    # lineas_generadas.append("id,chat_id,msg_idx,label,window_text") 
    
    for i in range(cantidad_de_ejemplos):
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
        texto_generado = plantilla.format(ORIGEN=origen, DESTINO=destino, FECHA=fecha, HORA=hora, PAX=pax)
        
        # --- Creamos los campos para el formato CSV ---
        # Usamos IDs únicos de ejemplo para id y chat_id
        id_unico = f"aug_u{uuid.uuid4().hex[:5]}" # ID corto tipo 'aug_uXXXXX'
        chat_id_unico = f"aug_chat_{i:04d}" # ID tipo 'aug_chat_0001'
        msg_idx = "1" # Índice de mensaje simple
        label = "Cotizando" # O la etiqueta que necesites para estos ejemplos
        
        # Formateamos el window_text
        # Escapamos comillas dobles dentro del texto si las hubiera (poco probable aquí)
        texto_escapado = texto_generado.replace('"', '""') 
        window_text = f'"<usr> {texto_escapado} </usr>"' # Encerrado en comillas dobles
        
        # Creamos la línea CSV final
        # Asegúrate que el orden de las columnas sea EXACTO al de tu archivo
        linea_csv = f"{id_unico},{chat_id_unico},{msg_idx},{label},{window_text}"
        
        lineas_generadas.append(linea_csv)

    return lineas_generadas

# --- Ejecución ---
if __name__ == "__main__":
    NUEVOS_EJEMPLOS = 50 # <-- Ajusta cuántas líneas quieres generar
    
    print(f"Generando {NUEVOS_EJEMPLOS} líneas en formato CSV...\n")
    
    lineas = generar_lineas_csv(NUEVOS_EJEMPLOS)
    
    # Imprimir cada línea generada
    for linea in lineas:
        print(linea)
        
    print(f"\n--- {len(lineas)} líneas generadas. Listas para copiar y pegar. ---")