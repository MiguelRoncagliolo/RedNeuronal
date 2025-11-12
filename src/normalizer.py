"""
Módulo unificado de normalización de texto para el sistema TextCNN.
Contiene todas las reglas de normalización aplicadas consistentemente 
en entrenamiento, predicción y evaluación.
"""

import re
from typing import List, Union, Sequence
from datetime import datetime

MONTHS = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", 
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08", 
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
}

def _get_current_month():
    """Retorna el mes actual en formato MM."""
    return datetime.now().strftime("%m")

REQUIRE_CANTIDAD_FOR_READY = True

def count_slots(text: str):
    """Cuenta slots detectados en el texto normalizado.
    Incluye los 6 campos requeridos: origen, destino, fecha, hora, cantidad, regreso.
    """
    slots = {
        "origen":   bool(re.search(r"\borigen\b", text)),
        "destino":  bool(re.search(r"\bdestino\b", text)),
        "fecha":    bool(re.search(r"\bfecha\b", text)),
        "hora":     bool(re.search(r"\bhora\b", text)),
        "cantidad": bool(re.search(r"\bsomos\s+\d+\b", text)),
        "regreso":  bool(re.search(r"\b(ida y vuelta|con regreso|sin regreso|solo ida|regreso)\b", text)),
    }
    total = sum(slots.values())
    return total, slots

def _split_segments(text: str, sep: str = "<SEP>"):
    """
    Divide el texto en segmentos. Soporta <SEP> y también formato <usr>/<sys>.
    """
    if "<SEP>" in text:
        return [t.strip() for t in re.split(r"\s*<SEP>\s*", text)]
    if "<sep>" in text:
        return [t.strip() for t in re.split(r"\s*<sep>\s*", text)]
    segments = []
    parts = re.split(r"(<usr>|<sys>)", text)
    current_seg = ""
    for part in parts:
        if part in ("<usr>", "<sys>"):
            if current_seg:
                segments.append(current_seg.strip())
            current_seg = part + " "
        else:
            current_seg += part
    if current_seg:
        segments.append(current_seg.strip())
    
    return [s for s in segments if s.strip()]

def infer_label_from_slots(
    texto_norm_joined: str,
    pred_idx: int,
    id2label: dict,
    label2id: dict
) -> int:
    """
    Infiere la etiqueta final basándose estrictamente en el conteo de slots.
    Regla de negocio estricta:
    - 0 slots → "Potencial cliente"
    - 1-5 slots → "Cotizando"
    - 6 slots (todos) → "Cotización generada"
    
    Esta regla tiene prioridad sobre la predicción cruda del modelo.
    """
    num_slots, slots = count_slots(texto_norm_joined)
    
    if num_slots == 0:
        return label2id["Potencial cliente"]
    
    if all(slots.values()):
        return label2id["Cotización generada"]
    
    return label2id["Cotizando"]

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

    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)

    s = re.sub(r"\bla dirección es\b", ", ", s)

    if "origen" not in s and "destino" not in s:
        s = re.sub(
            r"(?:^|\s)(?:desde|de)\s+([^<>]+?)\s+(?:hasta|hacia)\s+([^<>]+?)(?:\s|$)",
            r" origen \1 destino \2 ",
            s
        )
        if "origen" not in s and "destino" not in s:
            s = re.sub(
                r"(?:^|\s)(?:desde|de)\s+([^<>]+?)\s+al\s+([^<>]+?)(?:\s|$)",
                r" origen \1 destino \2 ",
                s
            )
        if "origen" not in s and "destino" not in s:
            s = re.sub(
                r"(?:^|\s)(?:desde|de)\s+([^<>]+?)\s+a\s+(?!las\s+\d)([^<>]+?)(?:\s|$)",
                r" origen \1 destino \2 ",
                s
            )
        s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\bsalida el\s+", "", s)
    s = re.sub(r"\bset\b|\bsept\.?\b", "septiembre", s)
    s = re.sub(r"\boct\.?\b", "octubre", s)
    s = re.sub(r"\bdic\.?\b", "diciembre", s)
    s = re.sub(r"\bene\.?\b", "enero", s)
    s = re.sub(r"\bfeb\.?\b", "febrero", s)
    s = re.sub(r"\bmar\.(?!\s+(del|de))\b", "marzo", s)
    s = re.sub(r"\babr\.?\b", "abril", s)
    s = re.sub(r"\bjun\.?\b", "junio", s)
    s = re.sub(r"\bjul\.?\b", "julio", s)
    s = re.sub(r"\bago\.?\b", "agosto", s)
    s = re.sub(r"\bnov\.?\b", "noviembre", s)

    s = re.sub(
        r"\bel\s+(\d{1,2})\s+de\s+este\s+mes\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{_get_current_month()}",
        s
    )
    
    s = re.sub(
        r"\bel\s+d[ií]a\s+(\d{1,2})\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{_get_current_month()}",
        s
    )
    s = re.sub(
        r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de\s+(\d{4}))?\b",
        lambda m: f"fecha {int(m.group(1)):02d}/{MONTHS[m.group(2)]}" + (f"/{m.group(3)}" if m.group(3) else ""),
        s
    )

    def _safe_date_replace(m):
        full_match = m.group(0)

        if re.search(r"hora\s*\d{1,2}:\d{2}", m.string[max(0, m.start()-10):m.end()+10]):
            return full_match
        day = int(m.group(1)); month = int(m.group(2)); year = m.group(3)
        return f"fecha {day:02d}/{month:02d}" + (f"/{year}" if year else "")

    s = re.sub(r"(?<!fecha\s)\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", _safe_date_replace, s)
    s = re.sub(r"\bregreso a las\s+(\d{1,2})(?::(\d{1,2}))?\b",
               lambda m: f"regreso hora {int(m.group(1)):02d}:{int((m.group(2) or '0')):02d}", s)
    s = re.sub(r"\bcon\s+regreso\s+(?=hora\b)", "regreso ", s)

    s = re.sub(r"\b(\d{1,2})\s*(?:h|hr|hrs|horas)\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)

    s = re.sub(r"(?<!hora\s)\b(\d{1,2})[:h](\d{1,2})\b", _norm_time_hhmm, s)
    s = re.sub(r"(?<!hora\s)\ba las\s+(\d{1,2})\b", lambda m: f"hora {int(m.group(1)):02d}:00", s)
    s = re.sub(r"\ba las\s+(?=hora\b)", "", s)

    def _to_24h(m):
        h = int(m.group(1))
        mm = int(m.group(2) or 0)
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and h != 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        return f"hora {h:02d}:{mm:02d}"

    s = re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", _to_24h, s)

    def _format_hhmm(m):
        num = int(m.group(1))
        s_local = m.string
        i0, i1 = m.start(), m.end()
        left  = s_local[max(0, i0-25):i0]
        right = s_local[i1:i1+20]
        addr_nearby = re.search(r"(av\.?|avenida|calle|camino|costanera|pasaje|psje|ruta|nº|n°|#|km)\s*$", left)
        trailing_punct = re.match(r"^[,.;-]", right)
        comma_city = re.search(r"^,\s*(santiago|valdivia|temuco|viña del mar|concepción|antofagasta|puerto montt|osorno)\b", right)
        if addr_nearby or trailing_punct or comma_city:
            return m.group(0)
        if 300 <= num <= 2359:
            h, mm = num // 100, num % 100
            if mm < 60:
                return f"hora {h:02d}:{mm:02d}"
        return m.group(0)

    def _format_hhmm_after_hora(m):
        num = int(m.group(1))
        if 300 <= num <= 2359:
            h, mm = num // 100, num % 100
            if mm < 60:
                return f"hora {h:02d}:{mm:02d}"
        return m.group(0)

    s = re.sub(r"(\b\d{1,2}[/-]\d{1,2}[/-])(\d{4}\b)", r"\1Y\2", s)

    s = re.sub(r"\bhora\s+(\d{4})\b", _format_hhmm_after_hora, s)
    s = re.sub(r"(?<!hora\s)\b(\d{4})\b", _format_hhmm, s)

    s = re.sub(r"([/-])Y(\d{4}\b)", r"\1\2", s)
    s = re.sub(
        r"\b(somos|para)\s+(\d{1,3})(\s*(personas?|pasajer[oa]s?|trabajador(?:es)?|pax))?\b",
        r"somos \2",
        s
    )

    s = re.sub(
        r"\b(vamos|viajamos|seremos)\s+(\d{1,3})(\s*(personas?|pasajer[oa]s?|trabajador(?:es)?|pax))?\b",
        r"somos \2",
        s
    )

    direction_pattern = r'\b(calle|avenida|av\.?|ruta|km|pasaje|psje|camino|costanera)\s+(\d{1,3})\s+(personas?|pasajer[oa]s?|trabajador(?:es)?|pax)\b'
    s = re.sub(direction_pattern, r'\1 \2 _DIRECTION_MARKER_ \3', s)
    
    s = re.sub(
        r"\b(\d{1,3})\s+(personas?|pasajer[oa]s?|trabajador(?:es)?|pax)\b",
        r"somos \1",
        s
    )
    
    s = re.sub(r'\b(calle|avenida|av\.?|ruta|km|pasaje|psje|camino|costanera)\s+(\d{1,3})\s+_DIRECTION_MARKER_\s+(personas?|pasajer[oa]s?|trabajador(?:es)?|pax)\b',
               r'\1 \2 \3', s)
    s = re.sub(r"\b(\d{1,2})\s*(?:ro|º|do|to|ero)\b", r"\1", s)

    synonyms = {
        r"\b(presupuesto|valor|precio)\b": "cotizar",
        r"\b(hola|buenas|qué tal|buen día|buenas tardes)\b": "saludo",
        r"\b(reserva|apartado|quiero agendar)\b": "reservar"
    }
    for pat, repl in synonyms.items():
        s = re.sub(pat, repl, s)

    s = re.sub(r"\b(por favor|quisiera|me podrías|deseo saber|gracias)\b", "", s)

    city_map = {
        r"\bsantiago( de chile| centro)?\b|\bstgo\b": "santiago",
        r"\b(cdmx|ciudad de méxico)\b": "mexico",
        r"\b(p\.?\s?montt|pto\.?\s?montt)\b": "puerto montt",
        r"\bbs\.?\s?as\.?\b|\bbsa\b": "buenos aires"
    }
    for pat, repl in city_map.items():
        s = re.sub(pat, repl, s)

    typo_map = {
        r"\bcotisaci[óo]n\b": "cotizacion",
        r"\bdestino+\b": "destino"
    }
    for pat, repl in typo_map.items():
        s = re.sub(pat, repl, s)

    s = re.sub(r"\bcon mi familia\b", "", s)
    s = re.sub(r"\bcorporativo\b", "", s)

    s = re.sub(r"\bsin\s+retorno\b", "sin regreso", s)
    s = re.sub(r"\bcon\s+retorno\b", "con regreso", s)

    if "fecha " in s and "[_has_fecha_]" not in s:
        s += " [_has_fecha_]"
    s = re.sub(r"\b(hora\s+){2,}", "hora ", s)
    s = re.sub(r"\b(fecha\s+){2,}", "fecha ", s)
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

    segments = [seg.strip() for seg in re.split(rf"\s*{sep_lit}\s*", s)]
    fixed_segments = []
    for seg in segments:
        if re.search(r"\b(?:desde|de)\s+", seg) and re.search(r"\b(?:hasta|hacia)\s+", seg):
            seg = re.sub(
                r"\b(?:desde|de)\s+(.+?)\s+(?:hasta|hacia)\s+(.+)\b",
                r"origen \1 destino \2",
                seg
            )
        if ("origen" not in seg) and ("destino" not in seg):
            seg = re.sub(
                r"^\s*(.+?)\s+(?:hasta|hacia)\s+(.+?)\s*$",
                r"origen \1 destino \2",
                seg
            )
        fixed_segments.append(seg)
    s = f" {sep} ".join(fixed_segments)

    s = re.sub(r"\borigen\s*:\s*", "origen ", s)
    s = re.sub(r"\bdestino\s*:\s*", "destino ", s)
    s = re.sub(r"\bdirección de salida\s*:\s*", "origen ", s)

    s = re.sub(r"\bla idea es\s+comenzar en\s+", "origen ", s)
    s = re.sub(r"\b(partir|salir)\s+desde\s+", "origen ", s)
    s = re.sub(r"\b(iniciar|inicio)\s+en\s+", "origen ", s)

    s = re.sub(r"\b(nuestro objetivo es\s+)?(llegar a|dirigirse a|ir a|ir hasta|terminar en|finalizar en)\s+", "destino ", s)

    s = re.sub(
        r"\b(somos|para)\s+(\d{1,3})(\s*(personas?|pasajer[oa]s?|trabajador(?:es)?|pax))?\b",
        r"somos \2",
        s
    )

    s = re.sub(
        r"\b(vamos|viajamos|seremos)\s+(\d{1,3})(\s*(personas?|pasajer[oa]s?|trabajador(?:es)?|pax))?\b",
        r"somos \2",
        s
    )

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
    joined_final = apply_alias_rules(joined.lower(), sep=sep)
    return joined_final

REQUIRED_FIELDS = {
    "origen": re.compile(r"\borigen\b"),
    "destino": re.compile(r"\bdestino\b"),
    "fecha": re.compile(r"\bfecha\b"),
    "hora": re.compile(r"\bhora\b"),
    "cantidad": re.compile(r"\bsomos\s+\d{1,3}\b"),
    "regreso": re.compile(r"\bregreso\b"),
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
    min_fields_for_quote: int = 2,
    probs: Union[None, Sequence[float], dict] = None,
    thr_gen: float = 0.6,
    require_hora_for_generada: bool = True,
    hora_must_follow_fecha: bool = True,
    intent_gate: bool = True,
    strict_traslado_exception: bool = True
) -> int:
    """
    Aplica reglas de negocio estrictas basadas en conteo de slots.
    Si enforce_fields=True, SIEMPRE usa infer_label_from_slots para obtener
    el índice final, sin dejar pasar "Cotización generada" cuando falte algún slot.
    """
    if not enforce_fields:
        return pred_idx

    return infer_label_from_slots(
        texto_norm_joined=texto_norm_joined,
        pred_idx=pred_idx,
        id2label=id2label,
        label2id=label2id,
    )
