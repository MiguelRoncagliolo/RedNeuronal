#!/usr/bin/env python3
"""
Data Audit Script - Auditoría automatizada del dataset

Genera reportes de calidad de datos incluyendo:
- Distribución de clases
- Duplicados por texto + label
- Estadísticas de longitud de texto
- Resumen de hallazgos en markdown

Uso:
    python src/data_audit.py --in data/dataset.csv --out artifacts_textcnn/audit/
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def detect_dataset(data_dir="data"):
    """
    Detecta automáticamente el dataset principal.
    Prioriza data/dataset.csv, luego busca otros CSV/JSONL en /data.
    """
    data_path = Path(data_dir)
    
    # Prioridad 1: dataset.csv
    priority_file = data_path / "dataset.csv"
    if priority_file.exists():
        return str(priority_file)
    
    # Prioridad 2: cualquier CSV en /data
    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        return str(csv_files[0])  # toma el primero
    
    # Prioridad 3: cualquier JSONL en /data
    jsonl_files = list(data_path.glob("*.jsonl"))
    if jsonl_files:
        return str(jsonl_files[0])
    
    return None


def infer_columns(df):
    """
    Infiere las columnas importantes del dataset basado en nombres comunes.
    Retorna un dict con las columnas detectadas.
    """
    columns = df.columns.tolist()
    detected = {}
    
    # Buscar columna de texto principal
    text_candidates = ['window_text', 'text', 'message', 'content', 'input', 'query']
    for col in text_candidates:
        if col in columns:
            detected['text_col'] = col
            break
    
    # Buscar columna de etiquetas
    label_candidates = ['label', 'class', 'target', 'category', 'classification']
    for col in label_candidates:
        if col in columns:
            detected['label_col'] = col
            break
    
    # Buscar columna de ID de conversación
    chat_candidates = ['chat_id', 'conversation_id', 'session_id', 'conv_id']
    for col in chat_candidates:
        if col in columns:
            detected['chat_col'] = col
            break
    
    # Buscar columna de ID único
    id_candidates = ['id', 'message_id', 'unique_id', 'idx']
    for col in id_candidates:
        if col in columns:
            detected['id_col'] = col
            break
    
    # Buscar columna de índice de mensaje
    msg_idx_candidates = ['msg_idx', 'message_idx', 'turn', 'step']
    for col in msg_idx_candidates:
        if col in columns:
            detected['msg_idx_col'] = col
            break
    
    return detected


def analyze_dataset(df, detected_cols, dataset_path):
    """
    Realiza el análisis completo del dataset.
    Retorna un dict con todos los resultados.
    """
    analysis = {
        'meta': {
            'dataset_path': dataset_path,
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'columns': df.columns.tolist(),
            'detected_columns': detected_cols,
            'analysis_timestamp': datetime.now().isoformat()
        }
    }
    
    # ===== ANÁLISIS DE NULOS =====
    null_analysis = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        null_analysis[col] = {
            'null_count': int(null_count),
            'null_percentage': round(null_pct, 2)
        }
    analysis['nulls'] = null_analysis
    
    # ===== ANÁLISIS DE DISTRIBUCIÓN DE CLASES =====
    if 'label_col' in detected_cols:
        label_col = detected_cols['label_col']
        class_dist = df[label_col].value_counts().to_dict()
        class_pct = (df[label_col].value_counts(normalize=True) * 100).round(2).to_dict()
        
        analysis['class_distribution'] = {
            'counts': class_dist,
            'percentages': class_pct,
            'num_classes': len(class_dist)
        }
    else:
        analysis['class_distribution'] = {'error': 'No se detectó columna de etiquetas'}
    
    # ===== ANÁLISIS DE DUPLICADOS =====
    if 'text_col' in detected_cols and 'label_col' in detected_cols:
        text_col = detected_cols['text_col']
        label_col = detected_cols['label_col']
        
        # Duplicados exactos (texto + label)
        exact_dupes = df.duplicated(subset=[text_col, label_col], keep=False)
        num_exact_dupes = exact_dupes.sum()
        
        # Duplicados solo por texto (diferentes labels)
        text_dupes = df.duplicated(subset=[text_col], keep=False)
        num_text_dupes = text_dupes.sum()
        
        analysis['duplicates'] = {
            'exact_duplicates': int(num_exact_dupes),
            'text_only_duplicates': int(num_text_dupes),
            'exact_duplicate_percentage': round((num_exact_dupes / len(df)) * 100, 2),
            'text_duplicate_percentage': round((num_text_dupes / len(df)) * 100, 2)
        }
        
        # Guardar duplicados para reporte
        if num_exact_dupes > 0:
            analysis['duplicate_examples'] = df[exact_dupes].to_dict('records')
        
    else:
        analysis['duplicates'] = {'error': 'No se detectaron columnas de texto y/o etiquetas'}
    
    # ===== ANÁLISIS DE LONGITUDES DE TEXTO =====
    if 'text_col' in detected_cols:
        text_col = detected_cols['text_col']
        text_lengths = df[text_col].astype(str).str.len()
        
        analysis['text_lengths'] = {
            'min': int(text_lengths.min()),
            'p25': int(text_lengths.quantile(0.25)),
            'p50': int(text_lengths.quantile(0.50)),
            'p75': int(text_lengths.quantile(0.75)),
            'p95': int(text_lengths.quantile(0.95)),
            'max': int(text_lengths.max()),
            'mean': round(text_lengths.mean(), 2),
            'std': round(text_lengths.std(), 2)
        }
        
        # Ejemplos de textos extremos
        min_idx = text_lengths.idxmin()
        max_idx = text_lengths.idxmax()
        analysis['text_examples'] = {
            'shortest': {
                'length': int(text_lengths[min_idx]),
                'text': str(df.loc[min_idx, text_col])[:200]  # truncar si es muy largo
            },
            'longest': {
                'length': int(text_lengths[max_idx]),
                'text': str(df.loc[max_idx, text_col])[:200]  # truncar si es muy largo
            }
        }
    else:
        analysis['text_lengths'] = {'error': 'No se detectó columna de texto'}
    
    # ===== ANÁLISIS DE CONVERSACIONES =====
    if 'chat_col' in detected_cols:
        chat_col = detected_cols['chat_col']
        unique_chats = df[chat_col].nunique()
        avg_msgs_per_chat = len(df) / unique_chats
        
        chat_lengths = df[chat_col].value_counts()
        
        analysis['conversations'] = {
            'unique_conversations': int(unique_chats),
            'avg_messages_per_conversation': round(avg_msgs_per_chat, 2),
            'min_messages_in_chat': int(chat_lengths.min()),
            'max_messages_in_chat': int(chat_lengths.max()),
            'median_messages_in_chat': int(chat_lengths.median())
        }
    else:
        analysis['conversations'] = {'info': 'No se detectó columna de ID de conversación'}
    
    return analysis


def generate_reports(analysis, output_dir):
    """
    Genera los archivos CSV y markdown con los resultados del análisis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== class_dist.csv =====
    if 'class_distribution' in analysis and 'counts' in analysis['class_distribution']:
        class_data = []
        counts = analysis['class_distribution']['counts']
        percentages = analysis['class_distribution']['percentages']
        
        for label, count in counts.items():
            class_data.append({
                'class': label,
                'count': count,
                'percentage': percentages.get(label, 0)
            })
        
        class_df = pd.DataFrame(class_data)
        class_df.to_csv(os.path.join(output_dir, 'class_dist.csv'), index=False)
    
    # ===== dupes.csv =====
    if 'duplicate_examples' in analysis:
        dupes_df = pd.DataFrame(analysis['duplicate_examples'])
        dupes_df.to_csv(os.path.join(output_dir, 'dupes.csv'), index=False)
    else:
        # Crear CSV vacío si no hay duplicados
        empty_dupes = pd.DataFrame(columns=['info'])
        empty_dupes.loc[0] = ['No se encontraron duplicados exactos']
        empty_dupes.to_csv(os.path.join(output_dir, 'dupes.csv'), index=False)
    
    # ===== len_stats.csv =====
    if 'text_lengths' in analysis and 'min' in analysis['text_lengths']:
        len_stats = []
        stats = analysis['text_lengths']
        for stat_name, value in stats.items():
            if stat_name not in ['shortest', 'longest']:  # excluir ejemplos
                len_stats.append({'statistic': stat_name, 'value': value})
        
        len_df = pd.DataFrame(len_stats)
        len_df.to_csv(os.path.join(output_dir, 'len_stats.csv'), index=False)
    
    # ===== audit.md =====
    generate_audit_markdown(analysis, output_dir)


def generate_audit_markdown(analysis, output_dir):
    """
    Genera el reporte markdown con hallazgos clave.
    """
    md_content = f"""# Auditoría de Dataset - {analysis['meta']['analysis_timestamp']}

## 📊 Información General

- **Dataset**: `{analysis['meta']['dataset_path']}`
- **Total de filas**: {analysis['meta']['total_rows']:,}
- **Total de columnas**: {analysis['meta']['total_cols']}
- **Fecha de análisis**: {analysis['meta']['analysis_timestamp']}

## 🔍 Columnas Detectadas

**Todas las columnas disponibles:**
{', '.join([f'`{col}`' for col in analysis['meta']['columns']])}

**Columnas inferidas automáticamente:**
"""
    
    detected = analysis['meta']['detected_columns']
    if detected:
        for key, col in detected.items():
            md_content += f"- **{key.replace('_', ' ').title()}**: `{col}`\n"
    else:
        md_content += "- ⚠️ No se pudieron inferir columnas automáticamente\n"
    
    # ===== DISTRIBUCIÓN DE CLASES =====
    md_content += "\n## 🎯 Distribución de Clases\n\n"
    
    if 'class_distribution' in analysis and 'counts' in analysis['class_distribution']:
        class_dist = analysis['class_distribution']
        md_content += f"**Total de clases**: {class_dist['num_classes']}\n\n"
        md_content += "| Clase | Cantidad | Porcentaje |\n"
        md_content += "|-------|----------|------------|\n"
        
        for label, count in class_dist['counts'].items():
            pct = class_dist['percentages'].get(label, 0)
            md_content += f"| {label} | {count:,} | {pct}% |\n"
        
        # Análisis de balance
        percentages = list(class_dist['percentages'].values())
        max_pct = max(percentages)
        min_pct = min(percentages)
        balance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        md_content += f"\n**Balance de clases**: {balance_ratio:.2f}:1 (mayor:menor)\n"
        
        if balance_ratio > 3:
            md_content += "⚠️ **ADVERTENCIA**: Dataset significativamente desbalanceado\n"
        elif balance_ratio > 1.5:
            md_content += "⚡ Dataset moderadamente desbalanceado\n"
        else:
            md_content += "✅ Dataset balanceado\n"
    else:
        md_content += "❌ No se pudo analizar distribución de clases\n"
    
    # ===== DUPLICADOS =====
    md_content += "\n## 🔄 Análisis de Duplicados\n\n"
    
    if 'duplicates' in analysis and 'exact_duplicates' in analysis['duplicates']:
        dupes = analysis['duplicates']
        md_content += f"- **Duplicados exactos** (texto + etiqueta): {dupes['exact_duplicates']:,} ({dupes['exact_duplicate_percentage']}%)\n"
        md_content += f"- **Duplicados de texto** (diferentes etiquetas): {dupes['text_only_duplicates']:,} ({dupes['text_duplicate_percentage']}%)\n\n"
        
        if dupes['exact_duplicates'] > 0:
            md_content += "⚠️ **RECOMENDACIÓN**: Revisar y eliminar duplicados exactos\n"
        
        if dupes['text_only_duplicates'] > dupes['exact_duplicates']:
            md_content += "🔍 **ATENCIÓN**: Textos iguales con etiquetas diferentes - posible inconsistencia en anotación\n"
        
        if dupes['exact_duplicates'] == 0 and dupes['text_only_duplicates'] == 0:
            md_content += "✅ No se encontraron duplicados\n"
    else:
        md_content += "❌ No se pudo analizar duplicados\n"
    
    # ===== LONGITUDES DE TEXTO =====
    md_content += "\n## 📏 Estadísticas de Longitud de Texto\n\n"
    
    if 'text_lengths' in analysis and 'min' in analysis['text_lengths']:
        lengths = analysis['text_lengths']
        md_content += "| Estadística | Valor |\n"
        md_content += "|-------------|-------|\n"
        md_content += f"| Mínimo | {lengths['min']} caracteres |\n"
        md_content += f"| Percentil 25 | {lengths['p25']} caracteres |\n"
        md_content += f"| Mediana (P50) | {lengths['p50']} caracteres |\n"
        md_content += f"| Percentil 75 | {lengths['p75']} caracteres |\n"
        md_content += f"| Percentil 95 | {lengths['p95']} caracteres |\n"
        md_content += f"| Máximo | {lengths['max']} caracteres |\n"
        md_content += f"| Promedio | {lengths['mean']} caracteres |\n"
        md_content += f"| Desviación estándar | {lengths['std']} caracteres |\n\n"
        
        # Ejemplos
        if 'text_examples' in analysis:
            examples = analysis['text_examples']
            md_content += "### 📝 Ejemplos de Textos\n\n"
            md_content += f"**Texto más corto** ({examples['shortest']['length']} caracteres):\n"
            md_content += f"```\n{examples['shortest']['text']}\n```\n\n"
            md_content += f"**Texto más largo** ({examples['longest']['length']} caracteres):\n"
            md_content += f"```\n{examples['longest']['text']}\n```\n\n"
        
        # Alertas
        if lengths['min'] < 10:
            md_content += "⚠️ **ADVERTENCIA**: Textos muy cortos detectados (< 10 caracteres)\n"
        
        if lengths['max'] > 1000:
            md_content += "⚠️ **ADVERTENCIA**: Textos muy largos detectados (> 1000 caracteres)\n"
    else:
        md_content += "❌ No se pudo analizar longitudes de texto\n"
    
    # ===== CONVERSACIONES =====
    if 'conversations' in analysis and 'unique_conversations' in analysis['conversations']:
        convs = analysis['conversations']
        md_content += "\n## 💬 Análisis de Conversaciones\n\n"
        md_content += f"- **Conversaciones únicas**: {convs['unique_conversations']:,}\n"
        md_content += f"- **Mensajes promedio por conversación**: {convs['avg_messages_per_conversation']}\n"
        md_content += f"- **Conversación más corta**: {convs['min_messages_in_chat']} mensajes\n"
        md_content += f"- **Conversación más larga**: {convs['max_messages_in_chat']} mensajes\n"
        md_content += f"- **Mediana de mensajes**: {convs['median_messages_in_chat']} mensajes\n\n"
        
        if convs['avg_messages_per_conversation'] < 2:
            md_content += "⚠️ **OBSERVACIÓN**: Conversaciones muy cortas en promedio\n"
    
    # ===== VALORES NULOS =====
    md_content += "\n## 🚫 Análisis de Valores Nulos\n\n"
    md_content += "| Columna | Nulos | Porcentaje |\n"
    md_content += "|---------|-------|------------|\n"
    
    has_nulls = False
    for col, null_info in analysis['nulls'].items():
        null_count = null_info['null_count']
        null_pct = null_info['null_percentage']
        md_content += f"| {col} | {null_count:,} | {null_pct}% |\n"
        if null_count > 0:
            has_nulls = True
    
    if not has_nulls:
        md_content += "\n✅ No se encontraron valores nulos\n"
    else:
        md_content += "\n⚠️ **RECOMENDACIÓN**: Revisar y manejar valores nulos antes del entrenamiento\n"
    
    # ===== HALLAZGOS CLAVE =====
    md_content += "\n## 🔑 Hallazgos Clave\n\n"
    
    key_findings = []
    
    # Sobre el dataset
    key_findings.append(f"Dataset con {analysis['meta']['total_rows']:,} filas y {analysis['meta']['total_cols']} columnas")
    
    # Sobre clases
    if 'class_distribution' in analysis and 'counts' in analysis['class_distribution']:
        num_classes = analysis['class_distribution']['num_classes']
        key_findings.append(f"Problema de clasificación con {num_classes} clases")
    
    # Sobre duplicados
    if 'duplicates' in analysis and 'exact_duplicates' in analysis['duplicates']:
        exact_dupes = analysis['duplicates']['exact_duplicates']
        if exact_dupes > 0:
            key_findings.append(f"⚠️ {exact_dupes:,} duplicados exactos requieren atención")
        else:
            key_findings.append("✅ No hay duplicados exactos")
    
    # Sobre longitudes
    if 'text_lengths' in analysis and 'min' in analysis['text_lengths']:
        p50 = analysis['text_lengths']['p50']
        key_findings.append(f"Longitud mediana de texto: {p50} caracteres")
    
    # Sobre conversaciones
    if 'conversations' in analysis and 'unique_conversations' in analysis['conversations']:
        chats = analysis['conversations']['unique_conversations']
        avg_msgs = analysis['conversations']['avg_messages_per_conversation']
        key_findings.append(f"{chats:,} conversaciones únicas con {avg_msgs:.1f} mensajes promedio")
    
    for finding in key_findings:
        md_content += f"- {finding}\n"
    
    # ===== RECOMENDACIONES =====
    md_content += "\n## 💡 Recomendaciones\n\n"
    
    recommendations = []
    
    if 'duplicates' in analysis and analysis['duplicates'].get('exact_duplicates', 0) > 0:
        recommendations.append("Eliminar duplicados exactos antes del entrenamiento")
    
    if 'class_distribution' in analysis and 'percentages' in analysis['class_distribution']:
        percentages = list(analysis['class_distribution']['percentages'].values())
        if len(percentages) > 1:
            max_pct, min_pct = max(percentages), min(percentages)
            if max_pct / min_pct > 3:
                recommendations.append("Considerar técnicas de balanceo de clases (oversampling, undersampling)")
    
    if 'text_lengths' in analysis and 'min' in analysis['text_lengths']:
        if analysis['text_lengths']['min'] < 10:
            recommendations.append("Revisar textos muy cortos (< 10 caracteres) para posible limpieza")
        if analysis['text_lengths']['max'] > 1000:
            recommendations.append("Considerar truncar textos muy largos (> 1000 caracteres)")
    
    if not recommendations:
        recommendations.append("✅ Dataset en buena condición para entrenamiento")
    
    for rec in recommendations:
        md_content += f"- {rec}\n"
    
    # Guardar archivo
    with open(os.path.join(output_dir, 'audit.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)


def main():
    parser = argparse.ArgumentParser(description='Auditoría automatizada del dataset')
    parser.add_argument('--in', dest='input_path', type=str, 
                       help='Ruta al dataset (si no se especifica, se detecta automáticamente)')
    parser.add_argument('--out', dest='output_dir', type=str, default='artifacts_textcnn/audit/',
                       help='Directorio de salida para reportes (default: artifacts_textcnn/audit/)')
    
    args = parser.parse_args()
    
    # Detectar dataset automáticamente si no se especifica
    if args.input_path:
        dataset_path = args.input_path
        if not os.path.exists(dataset_path):
            print(f"ERROR: File {dataset_path} does not exist")
            sys.exit(1)
    else:
        print("Detecting dataset automatically...")
        dataset_path = detect_dataset()
        if not dataset_path:
            print("ERROR: Could not detect any dataset in /data")
            print("TIP: Specify path with --in /path/to/dataset.csv")
            sys.exit(1)
        print(f"Dataset detected: {dataset_path}")
    
    # Cargar dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            df = pd.read_json(dataset_path, lines=True)
        else:
            print(f"Error: Unsupported file format: {dataset_path}")
            print("Supported formats: .csv, .jsonl")
            sys.exit(1)
        
        print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)
    
    # Inferir columnas
    print("Inferring dataset columns...")
    detected_cols = infer_columns(df)
    print(f"Detected columns: {detected_cols}")
    
    # Realizar análisis
    print("Performing complete analysis...")
    analysis = analyze_dataset(df, detected_cols, dataset_path)
    
    # Generar reportes
    print(f"Generating reports in: {args.output_dir}")
    generate_reports(analysis, args.output_dir)
    
    print("\nAudit completed!")
    print(f"Files generated in: {args.output_dir}")
    print("   - class_dist.csv")
    print("   - dupes.csv") 
    print("   - len_stats.csv")
    print("   - audit.md")
    print(f"\nSee full summary in: {args.output_dir}/audit.md")


if __name__ == "__main__":
    main()
