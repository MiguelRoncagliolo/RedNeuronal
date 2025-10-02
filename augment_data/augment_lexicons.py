#!/usr/bin/env python3
"""
Lexicones y patrones para generación de texto sintético
Específico para clase "Potencial cliente" en dominio de transporte privado
"""

import random

# ================== LEXICONES BÁSICOS ==================

SALUDOS = [
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches",
    "hola!", "hola!", "holaaa", "buen día", "qué tal"
]

CONSULTAS_BASE = [
    "necesito info", "consulta", "me pueden orientar", "cotizan traslados",
    "hay disponibilidad", "necesito una cotización", "quiero consultar",
    "me interesa saber", "podrían ayudarme", "tienen servicio",
    "hacen traslados", "me pueden asesorar", "quisiera consultar",
    "necesito orientación", "tienen disponibilidad"
]

TERMINOS_TRANSPORTE = [
    "traslado", "viaje", "servicio", "cotización", "presupuesto",
    "transporte", "movilización", "desplazamiento", "transfer",
    "ida", "ida y vuelta", "servicio privado",
    "traslado privado", "movilización privada"
]

# ================== UBICACIONES VAGAS ==================

ORIGENES_VAGOS = [
    "en providencia", "por ñuñoa", "sector las condes", "desde santiago",
    "por el centro", "en vitacura", "sector oriente", "por maipú",
    "en la reina", "sector norte", "por san miguel", "en quilicura",
    "zona poniente", "sector sur", "por puente alto", "en huechuraba",
    "cerca del metro", "por el mall", "sector residencial"
]

DESTINOS_VAGOS = [
    "al aeropuerto", "al centro", "a la costa", "por vitacura", "a viña",
    "hacia el aeropuerto", "al mall", "a valparaíso", "por las condes",
    "al puerto", "a rancagua", "hacia la cordillera", "a temuco",
    "por el sur", "hacia antofagasta", "al norte", "a concepción",
    "por valdivia", "a la serena", "hacia puerto montt"
]

# ================== TEMPORAL VAGO ==================

TIEMPO_VAGO = [
    "mañana temprano", "esta tarde", "fin de semana", "para el domingo",
    "en la noche", "mañana en la tarde", "pasado mañana", "esta semana",
    "el próximo fin de semana", "en unos días", "para la próxima semana",
    "mañana por la mañana", "el viernes", "el sábado", "para hoy",
    "más tarde", "en la tardecita", "tempranito"
]

# ================== PERSONAS VAGAS ==================

PERSONAS_VAGAS = [
    "somos pocos", "para varios", "para un grupo", "somos 3 aprox",
    "podría ser 4-5", "somos varios", "un grupo pequeño", "pocos",
    "algunos", "para la familia", "con mi familia", "nosotros",
    "somos hartos", "varios amigos", "un grupo", "entre varios"
]

# ================== MODISMOS CHILENOS ==================

MODISMOS_CHILENOS = [
    "está bueno"
]

JERGA_INFORMAL = [
    "info", "cotiza", "x favor", "pls", "porfi",
    "grax", "q tal", "xq", "tmb", "dp", "onda"
]

# ================== CONECTORES Y RELLENO ==================

CONECTORES = [
    "y", "o", "pero", "entonces", "además", "también", "igual",
    "por cierto", "ah", "oye", "mira", "eso sí", "claro"
]

EXPRESIONES_CORTESIA = [
    "por favor", "de antemano gracias", "muchas gracias", "saludos",
    "espero respuesta", "quedo atento", "gracias!", "porfa",
    "desde ya gracias", "cualquier cosa", "ojala puedan ayudarme"
]

# ================== PLANTILLAS DE GENERACIÓN ==================

# Plantillas simples (single-turn)
TEMPLATES_SIMPLE = [
    "{saludo}",
    "{saludo} {consulta}",
    "{consulta}",
    "{saludo} {consulta} {termino_transporte}",
    "{consulta} {termino_transporte}",
    "{saludo} {consulta} {detalle_vago}",
    "{consulta} {detalle_vago}",
    "{saludo} {termino_transporte}",
    "{saludo} {consulta} {termino_transporte} {detalle_vago}",
    "{consulta} {termino_transporte} {detalle_vago}",
]

# Plantillas con contexto (multi-turn)
TEMPLATES_CONTEXT = [
    "{saludo} <sep> {consulta}",
    "{saludo} <sep> {consulta} {termino_transporte}",
    "{consulta} <sep> {detalle_vago}",
    "{saludo} {consulta} <sep> {detalle_vago}",
    "{saludo} <sep> {consulta} {detalle_vago}",
    "{consulta} {termino_transporte} <sep> {detalle_vago}",
    "{saludo} <sep> {consulta} <sep> {detalle_vago}",
]

# ================== FUNCIONES DE GENERACIÓN ==================

def get_random_element(lista):
    """Obtiene elemento aleatorio de una lista"""
    return random.choice(lista)

def get_saludo():
    """Genera saludo aleatorio"""
    return get_random_element(SALUDOS)

def get_consulta():
    """Genera consulta base aleatoria"""
    return get_random_element(CONSULTAS_BASE)

def get_termino_transporte():
    """Genera término de transporte aleatorio"""
    return get_random_element(TERMINOS_TRANSPORTE)

def get_detalle_vago():
    """
    Genera detalle vago que NO complete todos los campos críticos.
    Máximo 2 campos específicos de los 4 críticos: origen, destino, fecha/hora, personas
    """
    detalles_posibles = []
    
    # Solo origen vago
    detalles_posibles.extend([f"desde {origen}" for origen in ORIGENES_VAGOS])
    detalles_posibles.extend([f"saliendo {origen}" for origen in ORIGENES_VAGOS[:5]])
    
    # Solo destino vago
    detalles_posibles.extend([f"hacia {destino}" for destino in DESTINOS_VAGOS])
    detalles_posibles.extend([f"con destino {destino}" for destino in DESTINOS_VAGOS[:5]])
    
    # Solo tiempo vago
    detalles_posibles.extend([f"para {tiempo}" for tiempo in TIEMPO_VAGO])
    detalles_posibles.extend([f"{tiempo}" for tiempo in TIEMPO_VAGO])
    
    # Solo personas vagas
    detalles_posibles.extend([f"{personas}" for personas in PERSONAS_VAGAS])
    
    # Combinaciones de máximo 2 campos (no específicos)
    for i in range(len(ORIGENES_VAGOS[:8])):
        for j in range(len(TIEMPO_VAGO[:6])):
            detalles_posibles.append(f"desde {ORIGENES_VAGOS[i]} {TIEMPO_VAGO[j]}")
    
    for i in range(len(DESTINOS_VAGOS[:8])):
        for j in range(len(PERSONAS_VAGAS[:4])):
            detalles_posibles.append(f"hacia {DESTINOS_VAGOS[i]}, {PERSONAS_VAGAS[j]}")
    
    # Agregar expresiones genéricas
    detalles_posibles.extend([
        "para un viaje", "necesito transporte", "requiero movilización",
        "es para trabajo", "tema familiar", "por un evento",
        "viaje corporativo", "para una reunión", "asunto personal"
    ])
    
    return get_random_element(detalles_posibles)

def add_chilean_style(text, probability=0.2):
    """
    Agrega ocasionalmente modismos chilenos al texto
    """
    if random.random() < probability:
        # Agregar al final ocasionalmente
        if random.random() < 0.5:
            modismo = get_random_element(MODISMOS_CHILENOS[:3])  # Solo los más naturales
            text += f", {modismo}"
        # O reemplazar palabras ocasionalmente
        else:
            if "información" in text:
                text = text.replace("información", "info")
            elif "qué tal" in text:
                text = text.replace("qué tal", "q tal")
    
    return text

def add_informal_touches(text, probability=0.3):
    """
    Agrega toques informales ocasionales
    """
    if random.random() < probability:
        # Agregar expresiones de cortesía
        if random.random() < 0.6:
            cortesia = get_random_element(EXPRESIONES_CORTESIA)
            text += f", {cortesia}"
        
        # Ocasionalmente usar jerga
        if random.random() < 0.3:
            if "información" in text:
                text = text.replace("información", "info")
            elif "para" in text and random.random() < 0.5:
                text = text.replace("para", "p/")
    
    return text

def generate_variations(base_text, num_variations=3):
    """
    Genera variaciones del texto base con técnicas de parafraseo
    """
    variations = [base_text]
    
    for _ in range(num_variations - 1):
        variation = base_text
        
        # Variación de puntuación
        if random.random() < 0.4:
            if "." not in variation and "!" not in variation and "?" not in variation:
                punctuation = random.choice(["", ".", "!", "?"])
                variation += punctuation
        
        # Diminutivos ocasionales
        if random.random() < 0.3:
            variation = variation.replace("mañana", "mañanita")
            variation = variation.replace("tarde", "tardecita")
            variation = variation.replace("temprano", "tempranito")
        
        # Orden ligeramente diferente (muy ocasional)
        if " y " in variation and random.random() < 0.2:
            parts = variation.split(" y ")
            if len(parts) == 2:
                variation = f"{parts[1]} y {parts[0]}"
        
        variations.append(variation)
    
    return variations
