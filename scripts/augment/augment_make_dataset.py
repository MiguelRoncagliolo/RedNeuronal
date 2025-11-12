"""
Generador de dataset sintético realista (Potencial cliente / Cotizando / Cotización generada)

- Emite conversaciones breves con <usr>/<sys> simulando WhatsApp.
- Controla los "slots" (origen, destino, fecha, hora, pax, regreso) SOLO en texto [USR].
- Valida con las MISMAS regex que usa prepare_dataset.py (copiadas aquí).
- Garantiza:
    Potencial cliente -> slots_state_count == 0
    Cotizando         -> 1 <= slots_state_count <= 5
    Cotización generada -> slots_state_count == 6

Salida:
  data/aug_potencial.jsonl
  data/aug_cotizando.jsonl
  data/aug_generada.jsonl
  data/aug_all.jsonl

Uso:
  python augment_make_dataset.py --n_pot 300 --n_cot 300 --n_gen 300 --seed 7
"""

import argparse, json, os, random, re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

RE_FECHA = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{8}|\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)\b",
    flags=re.IGNORECASE
)
RE_FECHA_COMPACTA = re.compile(r"\b\d{8}\b")
RE_HORA = re.compile(
    r"(?<!\d)(?:\b\d{1,2}[:\.]\d{2}\b|\ba\s+las\s+\d{1,2}(?::\d{2})?\b|\b\d{3,4}\b)(?!\d)",
    flags=re.IGNORECASE
)
RE_PAX = re.compile(
    r"\b(somos|para)\s+\d+\b|\b\d+\s+(?:personas|pasajeros?)\b",
    flags=re.IGNORECASE
)
RE_REGRESO = re.compile(
    r"\b(solo\s+ida|ida\s+y\s+vuelta|con\s+regreso|sin\s+regreso)\b",
    flags=re.IGNORECASE
)
RE_ORIGEN = re.compile(
    r"\b(origen)\b|(?:\bdesde\b\s+[a-z0-9áéíóúüñ][a-z0-9áéíóúüñ\s\.\-]{3,})|(?:\bsalida\s*(?:desde|:)\s*[a-z0-9])",
    flags=re.IGNORECASE
)
RE_DESTINO = re.compile(
    r"\b(destino)\b|(?:\bhasta\b\s+[a-z0-9áéíóúüñ][a-z0-9áéíóúüñ\s\.\-]{3,})|(?:\bllegada\s*(?:a|:)\s*[a-z0-9])",
    flags=re.IGNORECASE
)

def _mask_fecha_compacta(t: str) -> str:
    return RE_FECHA_COMPACTA.sub(" FECHACOMPACTA ", t)

def detect_slots_user_only(text_usr_concat: str) -> List[int]:
    """Devuelve [origen, destino, fecha, hora, pax, regreso] detectados SOLO en [USR]."""
    t = _mask_fecha_compacta(text_usr_concat.lower().strip())
    has_origen  = 1 if RE_ORIGEN.search(t)  else 0
    has_destino = 1 if RE_DESTINO.search(t) else 0
    has_fecha   = 1 if RE_FECHA.search(t)   else 0
    has_hora    = 1 if RE_HORA.search(t)    else 0
    has_pax     = 1 if RE_PAX.search(t)     else 0
    has_regreso = 1 if RE_REGRESO.search(t) else 0
    return [has_origen, has_destino, has_fecha, has_hora, has_pax, has_regreso]

def slots_state_count_from_chat(turns: List[Tuple[str,str]]) -> int:
    """Acumula SOLO [USR] y detecta slots sobre todo el acumulado."""
    usr_text = " ".join(m for r,m in turns if r=="USR")
    return sum(detect_slots_user_only(usr_text))

SALUDOS = [
    "hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hola!",
    "qué tal", "cómo estás", "cómo va todo", "buen día", "saludos", 
    "muy buenos días", "muy buenas tardes", "muy buenas noches", 
    "hola, necesito ayuda", "buenas, tengo una consulta", "hola buen día", 
    "hola buenísima tarde", "saludos cordiales", "hola transportes miramar"
]
CONSULTAS = [
    "necesito info", "quisiera consultar", "hacen traslados", "me ayudan con una cotización", 
    "tienen servicio", "hay disponibilidad", "podrían darme una tarifa",
    "me ayudan con un viaje", "cómo puedo reservar", "qué necesito para cotizar",
    "cuánto cuesta el viaje", "realizan transporte para empresas", "realizan viajes largos", 
    "tienen buses disponibles", "pueden llevarme al aeropuerto", 
    "cómo funciona el servicio", "quiero cotizar un viaje", "me cotizan por favor",
    "necesito traslado escolar", "trabajan los fines de semana"
]
CIUDADES = [
    "Santiago", "Valparaíso", "Viña del Mar", "Concepción", "La Serena", "Antofagasta", 
    "Temuco", "Puerto Montt", "Rancagua", "Talca", "Chillán", "Iquique", "Arica", 
    "Copiapó", "Ovalle", "Los Ángeles", "Curicó", "Osorno", "Coyhaique", "Punta Arenas", 
    "Quilpué", "Villa Alemana", "San Antonio", "Melipilla", "San Fernando", 
    "Linares", "Castro", "Valdivia", "Puerto Varas", "Calama", "Vallenar", 
    "Tocopilla", "Quillota", "Limache", "Rengo", "Arauco", "Coronel", "Lebu", "Angol",
    "San Felipe", "Los Andes", "Coquimbo", "Paillaco", "Frutillar", "Ancud", 
    "Chonchi", "Purranque", "Curanilahue", "Peñaflor", "Maipú", "Ñuñoa", 
    "Providencia", "Las Condes", "Lo Barnechea", "Puente Alto", "San Bernardo"
]
CALLES = [
    "Apoquindo", "Providencia", "Vicuña Mackenna", "Irarrázaval", "Pedro de Valdivia", 
    "Avenida Italia", "Los Leones", "San Pablo", "Matucana", "Pajaritos", "Alameda", 
    "Merced", "Estado", "Bandera", "Ahumada", "Lastarria", "Santa Rosa", 
    "Nueva Providencia", "Las Rejas", "Camino Melipilla", "El Bosque Norte", 
    "Costanera Norte", "Américo Vespucio", "Gran Avenida", "Los Militares", 
    "Tobalaba", "Larraín", "Santa Isabel", "Vitacura", "Los Dominicos",
    "1 Norte", "2 Norte", "3 Norte", "4 Norte", "San Martín", "Viana", "Álvarez", 
    "Libertad", "Avenida Perú", "Von Schroeders", "Quillota", "Arlegui", 
    "Valparaíso", "Recreo", "La Marina", "Agua Santa", "Los Castaños", 
    "La Torre", "Simón Bolívar", "Avenida España", "Plaza Vergara", "Etchevers"
]

def rand_dir() -> str:
    calle = random.choice(CALLES)
    num   = random.randint(100, 5000)
    ciudad= random.choice(CIUDADES)
    pre   = random.choice(["", "Av. ", "Av. ", "Camino ", "Calle "])
    return f"{pre}{calle} {num}, {ciudad}"

def rand_fecha() -> str:
    base = datetime.now() + timedelta(days=random.randint(1,120))
    form = random.choice([
        base.strftime("%d/%m/%Y"),
        base.strftime("%d-%m-%Y"),
        base.strftime("%d/%m"),
        base.strftime("%d-%m"),
        f"{base.day} de {base.strftime('%B').lower()} {base.year}",
        base.strftime("%d%m%Y")
    ])
    return form

def rand_hora() -> str:
    h  = random.randint(5,23)
    m  = random.choice([0,15,30,45])
    fm = random.choice([f"{h:02d}:{m:02d}", f"a las {h}", f"{h:02d}.{m:02d}", f"{h:02d}{m:02d}"])
    return fm

def rand_pax() -> str:
    n = random.randint(1,20)
    phr = random.choice([f"somos {n}", f"{n} personas", f"{n} pasajeros"])
    return phr

def rand_regreso() -> str:
    return random.choice(["solo ida","ida y vuelta","con regreso"])

def sys_ask_missing(missing: List[str]) -> str:
    pedidos = []
    mapping = {
        "origen": "origen",
        "destino": "destino",
        "fecha": "fecha",
        "hora": "hora",
        "pax": "cantidad de personas",
        "regreso": "si hay regreso"
    }
    for k in missing:
        pedidos.append(mapping[k])
    joined = " ".join(pedidos)
    return f"<sys> falta {joined} por favor"

def usr_slot(k: str) -> str:
    if k=="origen":  return f"origen {rand_dir()}"
    if k=="destino": return f"destino {rand_dir()}"
    if k=="fecha":   return f"el {rand_fecha()}"
    if k=="hora":    return f"a las {rand_hora().replace('a las ', '')}" if random.random()<0.5 else rand_hora()
    if k=="pax":     return rand_pax()
    if k=="regreso": return random.choice(["solo ida","ida y vuelta","con regreso"])
    return ""

def compose_text(turns: List[Tuple[str,str]]) -> str:
    parts=[]
    for r,m in turns:
        tag = "<usr>" if r=="USR" else "<sys>"
        parts.append(f"{tag} {m}")
    return " ".join(parts)

ALL_SLOTS = ["origen","destino","fecha","hora","pax","regreso"]

def gen_potencial_cliente() -> List[Tuple[str,str]]:
    saludo = random.choice(SALUDOS)
    consulta = random.choice(CONSULTAS)
    turns = [("USR", f"{saludo} {consulta}")]
    if random.random() < 0.5:
        turns.append(("SYS", "falta origen destino fecha hora cantidad de personas si hay regreso por favor"))
        turns.append(("USR", random.choice(["me orientan porfa","solo quiero info", "pueden indicarme cómo cotizar"])) )
    if slots_state_count_from_chat(turns) != 0:
        return gen_potencial_cliente()  # reintentar
    return turns

def gen_cotizando() -> List[Tuple[str,str]]:
    target = random.randint(1,5)
    chosen = random.sample(ALL_SLOTS, k=target)
    turns: List[Tuple[str,str]] = []

    turns.append(("USR", f"{random.choice(SALUDOS)} {random.choice(CONSULTAS)}"))
    missing = chosen.copy()
    turns.append(("SYS", sys_ask_missing(missing + ([] if random.random()<0.5 else random.sample(ALL_SLOTS, k=random.randint(0,2))))))

    random.shuffle(missing)
    split = random.randint(1, max(1, len(missing)-1)) if len(missing)>=2 else 1
    chunk1 = missing[:split]
    chunk2 = missing[split:]

    turns.append(("USR", " ".join(usr_slot(k) for k in chunk1)))
    if chunk2:
        turns.append(("USR", " ".join(usr_slot(k) for k in chunk2)))

    ssc = slots_state_count_from_chat(turns)
    if not (1 <= ssc <= 5):
        return gen_cotizando()
    return turns

def gen_generada() -> List[Tuple[str,str]]:
    turns: List[Tuple[str,str]] = []
    turns.append(("USR", f"{random.choice(SALUDOS)} {random.choice(['quiero cotizar un traslado', 'necesito una cotización'])}"))
    turns.append(("SYS", sys_ask_missing(ALL_SLOTS)))
    order = ALL_SLOTS.copy()
    random.shuffle(order)
    c1 = order[:2]; c2 = order[2:4]; c3 = order[4:]
    turns.append(("USR", " ".join(usr_slot(k) for k in c1)))
    turns.append(("USR", " ".join(usr_slot(k) for k in c2)))
    turns.append(("USR", " ".join(usr_slot(k) for k in c3)))
    if slots_state_count_from_chat(turns) != 6:
        return gen_generada()
    return turns

def gen_examples(n: int, klass: str, seed: int, base_chat_idx: int=0) -> List[Dict]:
    random.seed(seed)
    out=[]
    for i in range(n):
        if klass=="Potencial cliente":
            turns = gen_potencial_cliente()
        elif klass=="Cotizando":
            turns = gen_cotizando()
        else:
            turns = gen_generada()

        text = compose_text(turns)
        chat_id = f"chat_aug_{base_chat_idx + i:05d}"
        out.append({
            "id": f"usr_{base_chat_idx + i:06d}",
            "chat_id": chat_id,
            "text": text,
            "label": klass
        })
    return out

def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pot", type=int, default=300)
    ap.add_argument("--n_cot", type=int, default=300)
    ap.add_argument("--n_gen", type=int, default=300)
    ap.add_argument("--seed",  type=int, default=7)
    ap.add_argument("--outdir",type=str,  default="data")
    args = ap.parse_args()

    pot = gen_examples(args.n_pot, "Potencial cliente", args.seed, base_chat_idx=0)
    cot = gen_examples(args.n_cot, "Cotizando",         args.seed+1, base_chat_idx=args.n_pot)
    gen = gen_examples(args.n_gen, "Cotización generada", args.seed+2, base_chat_idx=args.n_pot+args.n_cot)

    p_pot = os.path.join(args.outdir,"aug_potencial.jsonl")
    p_cot = os.path.join(args.outdir,"aug_cotizando.jsonl")
    p_gen = os.path.join(args.outdir,"aug_generada.jsonl")
    p_all = os.path.join(args.outdir,"aug_all.jsonl")

    save_jsonl(p_pot, pot)
    save_jsonl(p_cot, cot)
    save_jsonl(p_gen, gen)

    all_rows = pot + cot + gen
    save_jsonl(p_all, all_rows)

    print(f"[aug] Potencial: {len(pot)}  -> {p_pot}")
    print(f"[aug] Cotizando: {len(cot)}  -> {p_cot}")
    print(f"[aug] Generada : {len(gen)}  -> {p_gen}")
    print(f"[aug] ALL      : {len(all_rows)}  -> {p_all}")

if __name__ == "__main__":
    main()
