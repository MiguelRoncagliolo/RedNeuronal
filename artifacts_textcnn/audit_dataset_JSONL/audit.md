# Auditoría de Dataset - 2025-10-01T18:37:46.790902

## 📊 Información General

- **Dataset**: `data/final.jsonl`
- **Total de filas**: 4,616
- **Total de columnas**: 5
- **Fecha de análisis**: 2025-10-01T18:37:46.790902

## 🔍 Columnas Detectadas

**Todas las columnas disponibles:**
`id`, `chat_id`, `msg_idx`, `text`, `label`

**Columnas inferidas automáticamente:**
- **Text Col**: `text`
- **Label Col**: `label`
- **Chat Col**: `chat_id`
- **Id Col**: `id`
- **Msg Idx Col**: `msg_idx`

## 🎯 Distribución de Clases

**Total de clases**: 3

| Clase | Cantidad | Porcentaje |
|-------|----------|------------|
| Cotizando | 2,200 | 47.66% |
| Cotización generada | 1,750 | 37.91% |
| Potencial cliente | 666 | 14.43% |

**Balance de clases**: 3.30:1 (mayor:menor)
⚠️ **ADVERTENCIA**: Dataset significativamente desbalanceado

## 🔄 Análisis de Duplicados

- **Duplicados exactos** (texto + etiqueta): 0 (0.0%)
- **Duplicados de texto** (diferentes etiquetas): 0 (0.0%)

✅ No se encontraron duplicados

## 📏 Estadísticas de Longitud de Texto

| Estadística | Valor |
|-------------|-------|
| Mínimo | 29 caracteres |
| Percentil 25 | 143 caracteres |
| Mediana (P50) | 267 caracteres |
| Percentil 75 | 306 caracteres |
| Percentil 95 | 351 caracteres |
| Máximo | 402 caracteres |
| Promedio | 230.97 caracteres |
| Desviación estándar | 98.12 caracteres |

### 📝 Ejemplos de Textos

**Texto más corto** (29 caracteres):
```
<usr> necesito una cotizacion
```

**Texto más largo** (402 caracteres):
```
<sys> falta cantidad de personas destino por favor <usr> destino vicuña mackenna 4224 valparaíso <usr> origen apoquindo 3881 la serena <sys> falta cantidad de personas hora por favor <usr> a las 2015 
```


## 💬 Análisis de Conversaciones

- **Conversaciones únicas**: 625
- **Mensajes promedio por conversación**: 7.39
- **Conversación más corta**: 1 mensajes
- **Conversación más larga**: 10 mensajes
- **Mediana de mensajes**: 8 mensajes


## 🚫 Análisis de Valores Nulos

| Columna | Nulos | Porcentaje |
|---------|-------|------------|
| id | 0 | 0.0% |
| chat_id | 0 | 0.0% |
| msg_idx | 0 | 0.0% |
| text | 0 | 0.0% |
| label | 0 | 0.0% |

✅ No se encontraron valores nulos

## 🔑 Hallazgos Clave

- Dataset con 4,616 filas y 5 columnas
- Problema de clasificación con 3 clases
- ✅ No hay duplicados exactos
- Longitud mediana de texto: 267 caracteres
- 625 conversaciones únicas con 7.4 mensajes promedio

## 💡 Recomendaciones

- Considerar técnicas de balanceo de clases (oversampling, undersampling)
