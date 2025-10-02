# Auditoría de Dataset - 2025-09-29T16:07:30.392083

## 📊 Información General

- **Dataset**: `data\dataset.csv`
- **Total de filas**: 4,512
- **Total de columnas**: 5
- **Fecha de análisis**: 2025-09-29T16:07:30.392083

## 🔍 Columnas Detectadas

**Todas las columnas disponibles:**
`id`, `chat_id`, `msg_idx`, `label`, `window_text`

**Columnas inferidas automáticamente:**
- **Text Col**: `window_text`
- **Label Col**: `label`
- **Chat Col**: `chat_id`
- **Id Col**: `id`
- **Msg Idx Col**: `msg_idx`

## 🎯 Distribución de Clases

**Total de clases**: 3

| Clase | Cantidad | Porcentaje |
|-------|----------|------------|
| Cotizando | 2,498 | 55.36% |
| Cotización generada | 1,750 | 38.79% |
| Potencial cliente | 264 | 5.85% |

**Balance de clases**: 9.46:1 (mayor:menor)
⚠️ **ADVERTENCIA**: Dataset significativamente desbalanceado

## 🔄 Análisis de Duplicados

- **Duplicados exactos** (texto + etiqueta): 218 (4.83%)
- **Duplicados de texto** (diferentes etiquetas): 218 (4.83%)

⚠️ **RECOMENDACIÓN**: Revisar y eliminar duplicados exactos

## 📏 Estadísticas de Longitud de Texto

| Estadística | Valor |
|-------------|-------|
| Mínimo | 29 caracteres |
| Percentil 25 | 188 caracteres |
| Mediana (P50) | 273 caracteres |
| Percentil 75 | 309 caracteres |
| Percentil 95 | 352 caracteres |
| Máximo | 402 caracteres |
| Promedio | 245.21 caracteres |
| Desviación estándar | 87.17 caracteres |

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

- **Conversaciones únicas**: 606
- **Mensajes promedio por conversación**: 7.45
- **Conversación más corta**: 1 mensajes
- **Conversación más larga**: 11 mensajes
- **Mediana de mensajes**: 8 mensajes


## 🚫 Análisis de Valores Nulos

| Columna | Nulos | Porcentaje |
|---------|-------|------------|
| id | 0 | 0.0% |
| chat_id | 0 | 0.0% |
| msg_idx | 0 | 0.0% |
| label | 0 | 0.0% |
| window_text | 0 | 0.0% |

✅ No se encontraron valores nulos

## 🔑 Hallazgos Clave

- Dataset con 4,512 filas y 5 columnas
- Problema de clasificación con 3 clases
- ⚠️ 218 duplicados exactos requieren atención
- Longitud mediana de texto: 273 caracteres
- 606 conversaciones únicas con 7.5 mensajes promedio

## 💡 Recomendaciones

- Eliminar duplicados exactos antes del entrenamiento
- Considerar técnicas de balanceo de clases (oversampling, undersampling)
