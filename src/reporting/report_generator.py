"""
Módulo para geração de relatórios finais
"""

from typing import Dict


def generate_final_report(stats: Dict, output_path: str):
    """
    Gera relatório final de análise em formato texto

    Args:
        stats: Dicionário com estatísticas
            {
                'frames_analisados': int,
                'pessoas_unicas': set,
                'anomalias_total': int,
                'emocoes': dict,
                'atividades': dict
            }
        output_path: Caminho para salvar o relatório
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("RELATORIO FINAL DE ANALISE\n")
        f.write("==========================\n")
        f.write(f"Total Frames: {stats['frames_analisados']}\n")
        f.write(f"Total Pessoas: {len(stats['pessoas_unicas'])}\n")
        f.write(f"Total Anomalias: {stats['anomalias_total']}\n")

        f.write("\nEMOCOES:\n")
        for emotion, count in stats['emocoes'].items():
            f.write(f"- {emotion}: {count}\n")

        f.write("\nATIVIDADES:\n")
        for activity, count in stats['atividades'].items():
            f.write(f"- {activity}: {count}\n")
