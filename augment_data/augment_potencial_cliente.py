#!/usr/bin/env python3
"""
Generador de ejemplos sintéticos para clase "Potencial cliente"
Dominio: transporte privado con restricciones de sobre-especificación

Uso:
    python src/augment_potencial_cliente.py --n 300 --with_context_ratio 0.3
"""

import argparse
import json
import os
import re
import random
from datetime import datetime
from collections import Counter

# Importar lexicones
from augment_lexicons import (
    TEMPLATES_SIMPLE, TEMPLATES_CONTEXT,
    get_saludo, get_consulta, get_termino_transporte, get_detalle_vago,
    add_chilean_style, add_informal_touches, generate_variations
)


class PotencialClienteGenerator:
    """
    Generador de ejemplos sintéticos para clase "Potencial cliente"
    """
    
    def __init__(self, seed=42):
        """
        Inicializa el generador con seed para reproducibilidad
        """
        random.seed(seed)
        self.generated_texts = set()  # Para evitar duplicados exactos
        self.discarded_count = 0
        self.over_specification_count = 0
        self.duplicate_count = 0
        
        # Patrones críticos para detectar sobre-especificación
        self.critical_patterns = {
            'origen_especifico': re.compile(r'\b(origen|desde|saliendo|dirección).{1,50}?\b(\d{3,4}|\w+\s+\d{3,4}|\w+\s+\#?\d+)', re.IGNORECASE),
            'destino_especifico': re.compile(r'\b(destino|hacia|hasta|llegar).{1,50}?\b(\d{3,4}|\w+\s+\d{3,4}|\w+\s+\#?\d+)', re.IGNORECASE),
            'fecha_especifica': re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]?\d{2,4}|\d{1,2}\s+de\s+\w+)', re.IGNORECASE),
            'hora_especifica': re.compile(r'\b\d{1,2}:\d{2}\b', re.IGNORECASE),
            'personas_especificas': re.compile(r'\bsomos\s+\d{1,2}\b|\b\d{1,2}\s+personas?\b', re.IGNORECASE)
        }
    
    def is_over_specified(self, text):
        """
        Detecta si el texto tiene demasiados campos críticos especificados.
        Regla: NO debe tener simultáneamente 4+ campos críticos específicos.
        """
        critical_count = 0
        
        # Contar campos críticos específicos
        if self.critical_patterns['origen_especifico'].search(text):
            critical_count += 1
        
        if self.critical_patterns['destino_especifico'].search(text):
            critical_count += 1
        
        if (self.critical_patterns['fecha_especifica'].search(text) or 
            self.critical_patterns['hora_especifica'].search(text)):
            critical_count += 1
        
        if self.critical_patterns['personas_especificas'].search(text):
            critical_count += 1
        
        # Si tiene 3+ campos específicos, es sobre-especificado para "Potencial cliente"
        return critical_count >= 3
    
    def is_valid_length(self, text, min_len=20, max_len=320):
        """
        Valida que el texto esté en el rango de longitud apropiado
        """
        return min_len <= len(text) <= max_len
    
    def is_duplicate(self, text):
        """
        Verifica si el texto ya fue generado
        """
        normalized = text.lower().strip()
        return normalized in self.generated_texts
    
    def add_generated_text(self, text):
        """
        Añade texto a la lista de generados
        """
        normalized = text.lower().strip()
        self.generated_texts.add(normalized)
    
    def generate_single_turn(self, style='chileno'):
        """
        Genera un ejemplo de un solo turno usando plantillas
        """
        template = random.choice(TEMPLATES_SIMPLE)
        
        # Completar template con slots
        text = template.format(
            saludo=get_saludo(),
            consulta=get_consulta(),
            termino_transporte=get_termino_transporte(),
            detalle_vago=get_detalle_vago() if '{detalle_vago}' in template else ''
        ).strip()
        
        # Limpiar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Agregar etiqueta <usr>
        text = f"<usr> {text}"
        
        # Aplicar estilo si es chileno
        if style == 'chileno':
            text = add_chilean_style(text)
        
        # Agregar toques informales
        text = add_informal_touches(text)
        
        return text
    
    def generate_multi_turn(self, style='chileno'):
        """
        Genera un ejemplo de múltiples turnos con contexto
        """
        template = random.choice(TEMPLATES_CONTEXT)
        
        # Completar template base
        base_text = template.format(
            saludo=get_saludo(),
            consulta=get_consulta(),
            termino_transporte=get_termino_transporte(),
            detalle_vago=get_detalle_vago() if '{detalle_vago}' in template else ''
        )
        
        # Separar por <sep> y agregar <usr> a cada parte
        parts = [part.strip() for part in base_text.split('<sep>')]
        usr_parts = [f"<usr> {part}" for part in parts if part]
        
        # Unir con separador apropiado
        text = ' <sep> '.join(usr_parts)
        
        # Limpiar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'<sep>\s+', ' <sep> ', text)
        text = re.sub(r'\s+<sep>', ' <sep>', text)
        
        # Aplicar estilo
        if style == 'chileno':
            text = add_chilean_style(text)
        
        text = add_informal_touches(text)
        
        return text
    
    def generate_example(self, with_context=False, style='chileno'):
        """
        Genera un ejemplo sintético válido
        """
        max_attempts = 10
        
        for attempt in range(max_attempts):
            if with_context:
                text = self.generate_multi_turn(style)
            else:
                text = self.generate_single_turn(style)
            
            # Validaciones
            if not self.is_valid_length(text):
                continue
            
            if self.is_over_specified(text):
                self.over_specification_count += 1
                continue
            
            if self.is_duplicate(text):
                self.duplicate_count += 1
                continue
            
            # Si pasa todas las validaciones, es válido
            self.add_generated_text(text)
            return text
        
        # Si no se pudo generar después de max_attempts
        self.discarded_count += 1
        return None
    
    def generate_dataset(self, n, with_context_ratio=0.3, style='chileno'):
        """
        Genera dataset completo de ejemplos sintéticos
        """
        examples = []
        
        print(f"Generating {n} synthetic examples...")
        print(f"Context ratio: {with_context_ratio}")
        print(f"Style: {style}")
        
        generated_count = 0
        total_attempts = 0
        
        while generated_count < n and total_attempts < n * 3:  # Límite de intentos
            total_attempts += 1
            
            # Decidir si usar contexto
            use_context = random.random() < with_context_ratio
            
            # Generar ejemplo
            text = self.generate_example(with_context=use_context, style=style)
            
            if text is not None:
                # Crear registro JSONL
                example = {
                    "id": f"usr_{generated_count:05d}",
                    "chat_id": f"chat_{generated_count // 10:04d}",  # 10 mensajes por chat
                    "msg_idx": generated_count % 10,
                    "text": text,
                    "label": "Potencial cliente"
                }
                
                examples.append(example)
                generated_count += 1
                
                # Progreso cada 50 ejemplos
                if generated_count % 50 == 0:
                    print(f"Generated {generated_count}/{n} examples...")
        
        print(f"Generation completed: {generated_count} valid examples")
        print(f"Total attempts: {total_attempts}")
        print(f"Discarded for over-specification: {self.over_specification_count}")
        print(f"Discarded for duplicates: {self.duplicate_count}")
        print(f"Other discards: {self.discarded_count}")
        
        return examples
    
    def get_statistics(self, examples):
        """
        Calcula estadísticas del dataset generado
        """
        lengths = [len(ex['text']) for ex in examples]
        
        stats = {
            'count': len(examples),
            'mean_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'p95_length': sorted(lengths)[int(0.95 * len(lengths))] if lengths else 0,
            'context_examples': len([ex for ex in examples if ' <sep> ' in ex['text']]),
            'single_turn_examples': len([ex for ex in examples if ' <sep> ' not in ex['text']])
        }
        
        return stats


def save_jsonl(examples, output_path):
    """
    Guarda ejemplos en formato JSONL
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")


def generate_summary_report(examples, stats, generator, output_dir):
    """
    Genera reporte de resumen en markdown
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'aug_summary.md')
    
    # Ejemplos de muestra
    sample_examples = random.sample(examples, min(10, len(examples)))
    
    content = f"""# Synthetic Data Generation Report - {datetime.now().isoformat()}

## 📊 Generation Summary

- **Total examples generated**: {stats['count']:,}
- **Target count**: {stats['count']:,}
- **Generation success rate**: {stats['count'] / (stats['count'] + generator.discarded_count + generator.over_specification_count + generator.duplicate_count) * 100:.1f}%

## 🚫 Discarded Examples

- **Over-specification discards**: {generator.over_specification_count:,}
- **Duplicate discards**: {generator.duplicate_count:,}
- **Other discards**: {generator.discarded_count:,}
- **Total discards**: {generator.over_specification_count + generator.duplicate_count + generator.discarded_count:,}

## 📏 Length Statistics

- **Mean length**: {stats['mean_length']:.1f} characters
- **Min length**: {stats['min_length']} characters
- **Max length**: {stats['max_length']} characters
- **P95 length**: {stats['p95_length']} characters

## 🔄 Context Distribution

- **Single-turn examples**: {stats['single_turn_examples']:,} ({stats['single_turn_examples']/stats['count']*100:.1f}%)
- **Multi-turn examples**: {stats['context_examples']:,} ({stats['context_examples']/stats['count']*100:.1f}%)

## 📝 Sample Examples

"""
    
    for i, example in enumerate(sample_examples, 1):
        content += f"""### Example {i}
```
{example['text']}
```

"""
    
    content += f"""## 🔧 Generation Rules Applied

1. **Field Restriction**: Maximum 2 critical fields specified simultaneously
2. **Length Bounds**: 20-320 characters
3. **Duplicate Prevention**: Exact text deduplication within generation batch
4. **Domain Vocabulary**: Transport/travel specific terminology
5. **Chilean Style**: Informal WhatsApp-style Spanish from Chile/LatAm
6. **Context Format**: Multi-turn examples use ` <sep> ` separator

## ✅ Quality Validation

- ✓ No examples with 3+ critical fields (origin + destination + datetime + persons)
- ✓ All examples within length bounds
- ✓ No exact duplicates in generated set
- ✓ All examples tagged as "Potencial cliente"
- ✓ Proper JSONL canonical format

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic examples for "Potencial cliente" class')
    
    parser.add_argument('--n', type=int, default=300,
                       help='Number of examples to generate (default: 300)')
    parser.add_argument('--style', type=str, default='chileno', choices=['chileno', 'standard'],
                       help='Text style (default: chileno)')
    parser.add_argument('--with_context_ratio', type=float, default=0.3,
                       help='Ratio of multi-turn examples (default: 0.3)')
    parser.add_argument('--output', type=str, default='data/aug_potencial_cliente.jsonl',
                       help='Output JSONL file (default: data/aug_potencial_cliente.jsonl)')
    parser.add_argument('--summary_dir', type=str, default='artifacts_textcnn/audit_aug',
                       help='Summary report directory (default: artifacts_textcnn/audit_aug)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SYNTHETIC DATA GENERATION - POTENCIAL CLIENTE")
    print("="*70)
    
    # Inicializar generador
    generator = PotencialClienteGenerator(seed=args.seed)
    
    # Generar ejemplos
    examples = generator.generate_dataset(
        n=args.n,
        with_context_ratio=args.with_context_ratio,
        style=args.style
    )
    
    if not examples:
        print("ERROR: No valid examples generated")
        return
    
    # Calcular estadísticas
    stats = generator.get_statistics(examples)
    
    # Guardar JSONL
    save_jsonl(examples, args.output)
    
    # Generar reporte
    generate_summary_report(examples, stats, generator, args.summary_dir)
    
    # Imprimir resumen final
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Generated examples: {stats['count']:,}")
    print(f"Mean length: {stats['mean_length']:.1f} chars")
    print(f"P95 length: {stats['p95_length']} chars")
    print(f"Multi-turn ratio: {stats['context_examples']/stats['count']*100:.1f}%")
    print(f"Over-specification discards: {generator.over_specification_count:,}")
    print(f"Duplicate discards: {generator.duplicate_count:,}")
    print(f"Success rate: {stats['count'] / (stats['count'] + generator.discarded_count + generator.over_specification_count + generator.duplicate_count) * 100:.1f}%")
    print("="*70)
    
    print(f"\nFiles generated:")
    print(f"  Dataset: {args.output}")
    print(f"  Summary: {args.summary_dir}/aug_summary.md")


if __name__ == "__main__":
    main()
