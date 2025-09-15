"""
Módulo unificado de normalización de texto para el sistema TextCNN.
Contiene todas las reglas de normalización aplicadas consistentemente 
en entrenamiento, predicción y evaluación.
"""

import re
from typing import List, Union, Sequence

# Mapeo de meses en español
MONTHS = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", 
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08", 
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
}

# Palabras que denotan intención de cotizar / consulta / viaje
INTENT_WORDS = re.compile(
    r"\b(cotizar|cotizacion|consulta|consultar|viaje|viajar|presupuesto|precio|valor|reservar|agendar|traslado)\b"
)

def _norm_time_hhmm(m):
    """Normaliza formato de hora con regex match object"""
    h = int(m.group(1))
    mm = m.group(2) or "00"
    try:
        mm = f"{int(mm):02d}"
    except Exception:
        mm = "00"
    return f"hora {h:02d}:{mm}"

def normalize_patterns(text: str) -> str:
    """
    Normaliza un texto individual aplicando todas las reglas básicas.
    Esta función se aplica ANTES de concatenar mensajes con <SEP>.
    """
    s = str(text).lower().strip()

    # Normalizar etiquetas con ":"
    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)

    # Limpiar frases de relleno
    s = re.sub(r"\bla dirección es\b", ", ", s)

    # "hasta" -> origen/destino (solo si aún no están)
    if "origen" not in s and "destino" not in s:
        s = re.sub(r"^\s*(.+?)\s+hasta\s+(.+)$", r"origen \1 destino \2", s)

    # Quitar marcador "salida el "
    s = re.sub(r"\bsalida el\s+", "", s)

    # Abreviaturas de meses (antes de procesar fechas)
    s = re.sub(r"\bset\b|\bsept\.?\b", "septiembre", s)
    s = re.sub(r"\boct\.?\b", "octubre", s)
    s = re.sub(r"\bdic\.?\b", "diciembre", s)
    s = re.sub(r"\bene\.?\b", "enero", s)
    s = re.sub(r"\bfeb\.?\b", "febrero", s)
    s = re.sub(r"\bmar\.(?!\s+(del|de))\b", "marzo", s)  # no convertir "mar del" → "marzo del"
    s = re.sub(r"\babr\.?\b", "abril", s)
    s = re.sub(r"\bjun\.?\b", "junio", s)
    s = re.sub(r"\bjul\.?\b", "julio", s)
    s = re.sub(r"\bago\.?\b", "agosto", s)
    s = re.sub(r"\bnov\.?\b", "noviembre", s)

    # Fechas con mes textual -> "fecha dd/mm(/yyyy)"
    s = re.sub(
        r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de\s+(\d{4}))?\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{MONTHS[m.group(2)]}" + (f"/{m.group(3)}" if m.group(3) else ""),
        s
    )

    # Fechas numéricas -> solo si NO está ya precedido por "fecha " y no es seguido por hora
    def _safe_date_replace(m):
        full_match = m.group(0)
        # No reemplazar si parece ser parte de una hora o si ya hay "fecha" antes
        if re.search(r"hora\s*\d{2}:\d{2}", m.string[max(0, m.end()-10):m.end()+10]):
            return full_match
        day = int(m.group(1))
        month = int(m.group(2)) 
        year = m.group(3)
        return f"fecha {day:02d}/{month:02d}" + (f"/{year}" if year else "")

    s = re.sub(r"(?<!fecha\s)\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", _safe_date_replace, s)

    # Horas: manejo de regreso
    s = re.sub(r"\bregreso a las\s+(\d{1,2})(?::(\d{1,2}))?\b",
               lambda m: f"regreso hora {int(m.group(1)):02d}:{int((m.group(2) or '0')):02d}", s)
    s = re.sub(r"\bcon\s+regreso\s+(?=hora\b)", "regreso ", s)

    # Horas: 12h/12hr/12hrs → "hora 12:00"
    s = re.sub(r"\b(\d{1,2})\s*(?:h|hr|hrs|horas)\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)

    # Horas: hh:mm o h:mm → solo si NO tiene "hora " antes
    s = re.sub(r"(?<!hora\s)\b(\d{1,2})[:h](\d{1,2})\b", _norm_time_hhmm, s)

    # "a las 12" → "hora 12:00" si NO tiene "hora " antes
    s = re.sub(r"(?<!hora\s)\ba las\s+(\d{1,2})\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)
    s = re.sub(r"\ba las\s+(?=hora\b)", "", s)

    # Función auxiliar para convertir am/pm a 24h
    def _to_24h(m):
        h = int(m.group(1))
        mm = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and h != 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        return f"hora {h:02d}:{mm:02d}"

    # Horas con am/pm: 7pm, 7:30pm, 07:30 am
    s = re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", _to_24h, s)

    # Formato hhmm pegado: 1930 → 19:30 (solo si no hay "hora " antes y está en rango válido)
    def _format_hhmm(m):
        num = int(m.group(1))
        if 300 <= num <= 2359:
            h = num // 100
            mm = num % 100
            if mm < 60:  # minutos válidos
                return f"hora {h:02d}:{mm:02d}"
        return m.group(0)  # no cambiar si no es hora válida

    def _format_hhmm_after_hora(m):
        # Para casos como "hora 1930" → "hora 19:30"
        num = int(m.group(1))
        if 300 <= num <= 2359:
            h = num // 100
            mm = num % 100
            if mm < 60:
                return f"hora {h:02d}:{mm:02d}"
        return m.group(0)

    # Formato hhmm pegado: "hora 1930" → "hora 19:30" 
    s = re.sub(r"\bhora\s+(\d{4})\b", _format_hhmm_after_hora, s)
    # Y formato solo: 1930 → "hora 19:30" (solo si no hay "hora " antes)
    s = re.sub(r"(?<!hora\s)\b(\d{4})\b", _format_hhmm, s)

    # Cantidad: "somos 4 (personas)" / "para 4" → "somos 4"
    s = re.sub(r"\b(somos|para)\s+(\d{1,3})(\s*personas?)?\b", r"somos \2", s)

    # Ordinales → cardinales: 1ro, 1º, 2do, 3ero → 1, 2, 3
    s = re.sub(r"\b(\d{1,2})\s*(?:ro|º|do|to|ero)\b", r"\1", s)

    # Sinónimos de intención
    synonyms = {
        r"\b(presupuesto|valor|precio)\b": "cotizar",
        r"\b(hola|buenas|qué tal|buen día|buenas tardes)\b": "saludo",
        r"\b(reserva|apartado|quiero agendar)\b": "reservar"
    }
    for pat, repl in synonyms.items():
        s = re.sub(pat, repl, s)

    # Stopwords / expresiones de relleno
    s = re.sub(r"\b(por favor|quisiera|me podrías|deseo saber|gracias)\b", "", s)

    # Alias de ciudades (mejorado)
    city_map = {
        r"\bsantiago( de chile| centro)?\b|\bstgo\b": "santiago",
        r"\b(cdmx|ciudad de méxico)\b": "mexico",
        r"\b(p\.?\s?montt|pto\.?\s?montt)\b": "puerto montt",
        r"\bbs\.?\s?as\.?\b|\bbsa\b": "buenos aires"
    }
    for pat, repl in city_map.items():
        s = re.sub(pat, repl, s)

    # Correcciones de errores comunes
    typo_map = {
        r"\bcotisaci[óo]n\b": "cotizacion",
        r"\bdestino+\b": "destino"
    }
    for pat, repl in typo_map.items():
        s = re.sub(pat, repl, s)

    # Manejo inteligente de expresiones (no eliminar información útil)
    s = re.sub(r"\bcon mi familia\b", "", s)  # esto sí se puede borrar
    s = re.sub(r"\bcorporativo\b", "", s)     # opcional borrar

    # Si aparece 'ida y vuelta' y no hay marcador regreso, añádelo
    if re.search(r"\bida y vuelta\b", s) and "regreso" not in s:
        s += " regreso"

    # Mantener 'traslado' como información útil (no borrar)

    # Marcador liviano para saber si este mensaje trae una fecha nueva (para lógica temporal)
    if "fecha " in s and "[_has_fecha_]" not in s:
        s += " [_has_fecha_]"

    # Limpiar espacios extra
    s = re.sub(r"\s+", " ", s).strip()
    return s

def apply_alias_rules(s: str, sep: str = "<SEP>") -> str:
    """
    Reglas complementarias (idempotentes) para capturar variantes comunes.
    Se asume que 's' ya está en minúsculas (normalize_patterns hace lower()).
    Estas reglas se aplican DESPUÉS de concatenar mensajes con separador.
    
    Args:
        s: Texto a procesar 
        sep: Separador usado entre mensajes
    """
    sep_lit = re.escape(sep.lower())
    
    # Etiquetas con ":", por si quedan
    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)

    # "desde ... hasta ..." -> origen/destino (también si aparece tras separador)
    s = re.sub(rf"(^|{sep_lit}\s*)desde\s+", r"\1origen ", s)
    s = re.sub(r"\s+hasta\s+", " destino ", s)

    # ORIGEN: frases típicas
    s = re.sub(r"\bla idea es\s+comenzar en\s+", "origen ", s)
    s = re.sub(r"\b(partir|salir)\s+desde\s+", "origen ", s)
    s = re.sub(r"\b(iniciar|inicio)\s+en\s+", "origen ", s)

    # DESTINO: frases típicas
    s = re.sub(r"\b(nuestro objetivo es\s+)?(llegar a|dirigirse a|ir a|ir hasta|terminar en|finalizar en)\s+", "destino ", s)

    # CANTIDAD: "viajamos N ..." -> "somos N"
    s = re.sub(r"\bviajamos\s+(\d{1,3})(\s*personas?)?(\s+en\s+total)?\b", r"somos \1", s)

    return s

def normalize_history_and_join(history_msgs: List[str], sep: str = "<SEP>", apply_normalizer: bool = True) -> str:
    """
    Normaliza cada mensaje con normalize_patterns, concatena con separador,
    y aplica reglas extra sobre el combinado (en minúsculas).
    Devuelve el texto final que realmente se clasifica.
    
    Args:
        history_msgs: Lista de mensajes de la conversación
        sep: Separador a usar entre mensajes
        apply_normalizer: Si aplicar normalize_patterns o no
    
    Returns:
        Texto normalizado y listo para clasificar
    """
    if apply_normalizer:
        norm_segments = [normalize_patterns(m) for m in history_msgs]
    else:
        norm_segments = history_msgs[:]

    joined = f" {sep} ".join(norm_segments)
    # Aplicar reglas extra sobre versión minúscula pasando el separador usado
    joined_final = apply_alias_rules(joined.lower(), sep=sep)
    return joined_final

# Campos obligatorios para validación
REQUIRED_FIELDS = {
    "origen": re.compile(r"\borigen\b"),
    "destino": re.compile(r"\bdestino\b"),
    "fecha": re.compile(r"\bfecha\b"),
    "hora": re.compile(r"\bhora\b"),
    "cantidad": re.compile(r"\bsomos\s+\d{1,3}\b"),
}

def _find_positions(s: str, pat: str):
    return [m.start() for m in re.compile(pat).finditer(s)]

def _get_prob(probs: Union[None, Sequence[float], dict], idx: int) -> float:
    if probs is None:
        return 0.0
    if isinstance(probs, dict):
        return float(probs.get(idx, 0.0))
    try:
        return float(probs[idx])
    except Exception:
        return 0.0

def apply_enforce_fields(
    texto_norm_joined: str,
    pred_idx: int,
    id2label: dict,
    label2id: dict,
    enforce_fields: bool = True,
    min_fields_for_quote: int = 2,        # recomendado: 2 (puedes subir a 3)
    probs: Union[None, Sequence[float], dict] = None,
    thr_gen: float = 0.6,
    require_hora_for_generada: bool = True,
    hora_must_follow_fecha: bool = True,
    intent_gate: bool = True,
    strict_traslado_exception: bool = True
) -> int:
    """
    Reglas de negocio mejoradas:
    - Generada: requiere básicos (origen, destino, fecha) y (opcional) hora válida.
    - Hora válida: si hora_must_follow_fecha, entonces 'hora' debe aparecer
      en el mismo mensaje o DESPUÉS de la última 'fecha'.
    - Potencial -> Cotizando: si hay intención + ≥1 campo (o ≥ min_fields_for_quote),
      con excepción cuando es solo 'traslado' + 1 campo.
    """
    if not enforce_fields:
        return pred_idx

    lbl_pred = id2label[int(pred_idx)]
    flags = {k: bool(rx.search(texto_norm_joined)) for k, rx in REQUIRED_FIELDS.items()}
    covered = sum(flags.values())

    # --- Orden fecha/hora ---
    pos_fecha = _find_positions(texto_norm_joined, r"\bfecha\b")
    pos_hora  = _find_positions(texto_norm_joined, r"\bhora\b")
    hora_valida = flags["hora"]

    if require_hora_for_generada:
        if not pos_hora:
            hora_valida = False
        elif hora_must_follow_fecha and pos_fecha:
            # la hora debe aparecer en o después de la última fecha
            hora_valida = any(h >= pos_fecha[-1] for h in pos_hora)

    # --- Reglas para Generada ---
    basic_ok = flags["origen"] and flags["destino"] and flags["fecha"]

    if lbl_pred == "Cotización generada":
        # si falta cualquier básico o la hora no es válida (si se requiere), baja a Cotizando
        if not basic_ok or (require_hora_for_generada and not hora_valida):
            return label2id["Cotizando"]
        # opcionalmente usa thr_gen si pasas probs
        gen_id = label2id.get("Cotización generada")
        if gen_id is not None and probs is not None:
            if _get_prob(probs, gen_id) < thr_gen:
                return label2id["Cotizando"]
        return label2id["Cotización generada"]

    # Si TODOS los campos presentes (incluida hora cuando se requiere), promueve a Generada
    if basic_ok and (not require_hora_for_generada or hora_valida):
        return label2id["Cotización generada"]

    # --- Gate por intención para subir a Cotizando ---
    has_intent = bool(INTENT_WORDS.search(texto_norm_joined))
    useful_fields = sum([
        flags["origen"], flags["destino"], flags["fecha"],
        flags["hora"], flags["cantidad"]
    ])

    if lbl_pred == "Potencial cliente":
        # regla base por cantidad de campos
        if useful_fields >= min_fields_for_quote:
            return label2id["Cotizando"]
        # gate por intención + ≥1 campo
        if intent_gate and has_intent and useful_fields >= 1:
            if strict_traslado_exception:
                # si la única intención detectada es 'traslado' y solo hay 1 campo -> mantén Potencial
                m = INTENT_WORDS.search(texto_norm_joined)
                only_traslado = (m and m.group(0) == "traslado")
                if only_traslado and useful_fields == 1:
                    return label2id["Potencial cliente"]
            return label2id["Cotizando"]

    return pred_idx
