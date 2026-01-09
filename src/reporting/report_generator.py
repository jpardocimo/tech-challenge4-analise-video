"""
Gerador de Relatório Analítico de Vídeo
Lê CSV de análise e gera relatório com insights automáticos
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter


class CSVReportGenerator:
    """
    Gerador de relatórios analíticos baseado em CSV
    Sem dependência de LLM - usa análise estatística
    """
    
    def __init__(self, csv_path: str):
        """
        Inicializa o gerador de relatórios
        
        Args:
            csv_path: Caminho para o arquivo CSV de análise
        """
        self.csv_path = csv_path
        self.df = None
        self.video_name = None
        self.total_frames = 0
        self.duration_seconds = 0
        self.fps = 30  # Padrão
        
    def load_csv(self):
        """Carrega e processa o CSV"""
        try:
            self.df = pd.read_csv(self.csv_path, sep=';')
            
            # Calcula métricas básicas
            self.total_frames = self.df['frame'].max() if not self.df.empty else 0
            self.duration_seconds = self.total_frames / self.fps
            
            # Remove linhas vazias (sem detecções)
            self.df = self.df[self.df['activity'].notna() & (self.df['activity'] != '')]
            
            print(f"✅ CSV carregado: {len(self.df)} registros")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar CSV: {e}")
            return False
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Gera relatório completo com insights
        
        Args:
            output_path: Caminho para salvar o relatório (opcional)
            
        Returns:
            Texto do relatório em markdown
        """
        if self.df is None or self.df.empty:
            return "# Erro\n\nNenhum dado disponível para análise."
        
        # Gera seções
        header = self._generate_header()
        summary = self._generate_executive_summary()
        statistics = self._generate_statistics()
        insights = self._generate_insights()
        activities = self._generate_activities_analysis()
        emotions = self._generate_emotions_analysis()
        anomalies = self._generate_anomalies_analysis()
        postures = self._generate_postures_analysis()
        timeline = self._generate_timeline_highlights()
        
        # Monta relatório
        report = f"""{header}

{summary}

{statistics}

{insights}

{activities}

{emotions}

{postures}

{anomalies}

{timeline}

---
*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
*Fonte: {Path(self.csv_path).name}*
"""
        
        # Salva se caminho fornecido
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            print(f"📄 Relatório salvo em: {output_path}")
        
        return report
    
    def _generate_header(self) -> str:
        """Gera cabeçalho do relatório"""
        return f"""# 📊 Relatório Analítico de Vídeo
## Tech Challenge - Fase 4

**Arquivo CSV**: `{Path(self.csv_path).name}`  
**Data da Análise**: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
**Duração Estimada**: {self.duration_seconds:.1f} segundos ({self.total_frames} frames)
"""
    
    def _generate_executive_summary(self) -> str:
        """Gera resumo executivo com principais descobertas"""
        unique_people = self.df['face_id'].nunique()
        total_detections = len(self.df)
        
        # Atividade mais comum (excluindo "Atividade nao detectada")
        activities_filtered = self.df[self.df['activity'] != 'Atividade nao detectada']
        
        if not activities_filtered.empty:
            top_activity = activities_filtered['activity'].value_counts().head(1)
            top_activity_name = top_activity.index[0] if not top_activity.empty else "N/A"
            top_activity_count = top_activity.values[0] if not top_activity.empty else 0
        else:
            top_activity_name = "N/A"
            top_activity_count = 0
        
        # Anomalias
        anomalies_df = self.df[self.df['anomalies'].notna() & (self.df['anomalies'] != '')]
        total_anomalies = len(anomalies_df)
        
        # Monta resumo
        activity_text = f'A atividade predominante foi **"{top_activity_name}"**, observada em **{top_activity_count} detecções**.' if top_activity_name != "N/A" else "Nenhuma atividade específica foi predominante."
        
        return f"""## 📋 Resumo Executivo

Durante a análise do vídeo, foram detectadas **{unique_people} rostos** em **{total_detections} frames processados**.

{activity_text} 

{f"⚠️ Foram identificadas **{total_anomalies} anomalias** comportamentais que requerem atenção." if total_anomalies > 0 else "✅ Nenhuma anomalia comportamental significativa foi detectada."}
"""
    
    def _generate_statistics(self) -> str:
        """Gera tabela de estatísticas gerais"""
        unique_people = self.df['face_id'].nunique()
        total_detections = len(self.df)
        frames_with_detection = self.df['frame'].nunique()
        anomalies_count = len(self.df[self.df['anomalies'].notna() & (self.df['anomalies'] != '')])
        
        # Confiança média de postura
        avg_confidence = self.df['body_pose_confidence'].astype(float).mean() if 'body_pose_confidence' in self.df.columns else 0
        
        return f"""## 📈 Estatísticas Gerais

| Métrica | Valor |
|---------|-------|
| **Pessoas Únicas** | {unique_people} |
| **Total de Detecções** | {total_detections} |
| **Frames com Detecção** | {frames_with_detection} / {self.total_frames} ({frames_with_detection/max(self.total_frames,1)*100:.1f}%) |
| **Anomalias Detectadas** | {anomalies_count} |
| **Confiança Média de Postura** | {avg_confidence:.2%} |
"""
    
    def _generate_insights(self) -> str:
        """Gera insights automáticos baseados nos dados"""
        insights = []
        
        # Insight 1: Pessoas lendo/usando celular
        reading_activities = self.df[self.df['activity'].str.contains('Lendo|Celular', case=False, na=False)]
        if not reading_activities.empty:
            unique_readers = reading_activities['face_id'].nunique()
            insights.append(f"📖 **{unique_readers} pessoa(s)** foram observadas lendo ou usando celular")
        
        # Insight 2: Movimentos abruptos
        abrupt_movements = self.df[self.df['anomalies'].str.contains('MOVIMENTO ABRUPTO', case=False, na=False)]
        if not abrupt_movements.empty:
            unique_movers = abrupt_movements['tid_used'].nunique()
            insights.append(f"⚡ **{unique_movers} pessoa(s)** apresentaram movimentos abruptos ou aleatórios")
        
        # Insight 3: Quedas detectadas
        falls = self.df[self.df['anomalies'].str.contains('QUEDA', case=False, na=False)]
        if not falls.empty:
            insights.append(f"🚨 **{len(falls)} possível(is) queda(s)** detectada(s) - requer atenção!")
        
        # Insight 4: Instabilidade emocional
        emotional_instability = self.df[self.df['anomalies'].str.contains('Instabilidade Emocional|CARETA', case=False, na=False)]
        if not emotional_instability.empty:
            unique_emotional = emotional_instability['face_id'].nunique()
            insights.append(f"😟 **{unique_emotional} pessoa(s)** apresentaram instabilidade emocional ou caretas")
        
        # Insight 5: Pessoas acenando
        waving = self.df[self.df['activity'].str.contains('Acenando', case=False, na=False)]
        if not waving.empty:
            unique_wavers = waving['face_id'].nunique()
            insights.append(f"👋 **{unique_wavers} pessoa(s)** foram vistas acenando")
        
        # Insight 6: Trabalho com laptop
        laptop_work = self.df[self.df['activity'].str.contains('Laptop', case=False, na=False)]
        if not laptop_work.empty:
            unique_workers = laptop_work['face_id'].nunique()
            insights.append(f"💻 **{unique_workers} pessoa(s)** trabalhando com laptop")
        
        # Insight 7: Objetos mais detectados
        if 'objects_detected' in self.df.columns:
            all_objects = []
            for objects_str in self.df['objects_detected'].dropna():
                if objects_str:
                    all_objects.extend([obj.strip() for obj in str(objects_str).split(',')])
            
            if all_objects:
                object_counts = Counter(all_objects)
                top_object = object_counts.most_common(1)[0]
                insights.append(f"📦 Objeto mais detectado: **{top_object[0]}** ({top_object[1]} vezes)")
        
        if not insights:
            return "## 💡 Insights\n\n*Nenhum insight específico identificado.*"
        
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        return f"""## 💡 Insights Principais

{insights_text}
"""
    
    def _generate_activities_analysis(self) -> str:
        """Gera análise detalhada de atividades"""
        activity_counts = self.df['activity'].value_counts().head(10)
        
        if activity_counts.empty:
            return "## 🎯 Análise de Atividades\n\n*Nenhuma atividade detectada.*"
        
        # Tabela
        table = "| Atividade | Frequência | Percentual |\n|-----------|------------|------------|\n"
        total = len(self.df)
        for activity, count in activity_counts.items():
            percentage = (count / total) * 100
            table += f"| {activity} | {count} | {percentage:.1f}% |\n"
        
        return f"""## 🎯 Análise de Atividades

**Top 10 Atividades Detectadas**

{table}
"""
    
    def _generate_emotions_analysis(self) -> str:
        """Gera análise de emoções"""
        if 'emotion' not in self.df.columns:
            return ""
        
        emotions_df = self.df[self.df['emotion'].notna() & (self.df['emotion'] != '')]
        
        if emotions_df.empty:
            return "## 😊 Análise de Emoções\n\n*Nenhuma emoção detectada.*"
        
        emotion_counts = emotions_df['emotion'].value_counts()
        
        table = "| Emoção | Frequência | Percentual |\n|--------|------------|------------|\n"
        total = len(emotions_df)
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100
            table += f"| {emotion} | {count} | {percentage:.1f}% |\n"
        
        dominant_emotion = emotion_counts.index[0] if not emotion_counts.empty else "N/A"
        
        return f"""## 😊 Análise de Emoções

**Emoção Predominante**: {dominant_emotion}

{table}
"""
    
    def _generate_postures_analysis(self) -> str:
        """Gera análise de posturas"""
        if 'posture' not in self.df.columns:
            return ""
        
        postures_df = self.df[self.df['posture'].notna() & (self.df['posture'] != '')]
        
        if postures_df.empty:
            return ""
        
        posture_counts = postures_df['posture'].value_counts()
        
        # Tradução
        posture_labels = {
            'sitting': 'Sentado',
            'standing': 'Em pé',
            'lying_down': 'Deitado'
        }
        
        table = "| Postura | Frequência | Percentual |\n|---------|------------|------------|\n"
        total = len(postures_df)
        for posture, count in posture_counts.items():
            percentage = (count / total) * 100
            posture_pt = posture_labels.get(posture, posture)
            table += f"| {posture_pt} | {count} | {percentage:.1f}% |\n"
        
        return f"""## 🧍 Análise de Posturas

{table}
"""
    
    def _generate_anomalies_analysis(self) -> str:
        """Gera análise detalhada de anomalias"""
        anomalies_df = self.df[self.df['anomalies'].notna() & (self.df['anomalies'] != '')]
        
        if anomalies_df.empty:
            return "## ⚠️ Anomalias Detectadas\n\n✅ **Nenhuma anomalia detectada** - Comportamento normal durante toda a análise."
        
        # Conta tipos de anomalias
        all_anomalies = []
        for anomalies_str in anomalies_df['anomalies']:
            if anomalies_str and str(anomalies_str) != 'nan':
                all_anomalies.extend([a.strip() for a in str(anomalies_str).split('|')])
        
        anomaly_counts = Counter(all_anomalies)
        
        # Frames com anomalias
        anomaly_frames = anomalies_df['frame'].unique()
        
        # Conta tipos gerais (sem IDs de tracking)
        movement_count = sum(1 for a in all_anomalies if 'MOVIMENTO ABRUPTO' in a.upper())
        fall_count = sum(1 for a in all_anomalies if 'QUEDA' in a.upper())
        emotional_count = sum(1 for a in all_anomalies if 'INSTABILIDADE' in a.upper() or 'CARETA' in a.upper())
        
        summary_parts = []
        if movement_count > 0:
            summary_parts.append(f"**{movement_count}** movimentos abruptos")
        if fall_count > 0:
            summary_parts.append(f"**{fall_count}** possíveis quedas")
        if emotional_count > 0:
            summary_parts.append(f"**{emotional_count}** instabilidades emocionais")
        
        summary_text = ", ".join(summary_parts) if summary_parts else "anomalias diversas"
        
        return f"""## ⚠️ Anomalias Detectadas

**Total de Anomalias**: {len(all_anomalies)}  
**Frames Afetados**: {len(anomaly_frames)}  
**Resumo**: {summary_text}
"""
    
    def _generate_anomaly_observations(self, anomaly_counts: Counter) -> str:
        """Gera observações sobre anomalias"""
        # Método mantido para compatibilidade, mas não usado mais
        return ""
    
    def _generate_timeline_highlights(self) -> str:
        """Gera destaques da linha do tempo"""
        # Pega primeiros e últimos eventos interessantes
        anomalies_df = self.df[self.df['anomalies'].notna() & (self.df['anomalies'] != '')]
        
        if anomalies_df.empty:
            return ""
        
        # Primeiras 5 anomalias
        first_anomalies = anomalies_df.head(5)
        
        timeline = "| Frame | Timestamp | Evento |\n|-------|-----------|--------|\n"
        
        for _, row in first_anomalies.iterrows():
            frame = row['frame']
            timestamp_ms = row.get('timestamp_ms', frame * 33)  # Aproximado
            timestamp_sec = timestamp_ms / 1000
            anomaly = row['anomalies']
            
            timeline += f"| {frame} | {timestamp_sec:.1f}s | {anomaly} |\n"
        
        return f"""## ⏱️ Linha do Tempo - Eventos Principais

**Primeiros Eventos Detectados**

{timeline}

*Para análise completa, consulte o arquivo CSV.*
"""


# Função auxiliar para uso direto
def generate_report_from_csv(csv_path: str, output_path: str = None) -> str:
    """
    Gera relatório a partir de um arquivo CSV
    
    Args:
        csv_path: Caminho do CSV de entrada
        output_path: Caminho do relatório de saída (opcional)
        
    Returns:
        Texto do relatório
    """
    generator = CSVReportGenerator(csv_path)
    
    if not generator.load_csv():
        return "Erro ao carregar CSV"
    
    return generator.generate_report(output_path)


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python report_generator.py <caminho_csv> [caminho_saida]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not output_file:
        # Gera nome automático
        csv_path = Path(csv_file)
        output_file = csv_path.parent / f"relatorio_analitico_{csv_path.stem}.md"
    
    print(f"📊 Gerando relatório de: {csv_file}")
    report = generate_report_from_csv(csv_file, str(output_file))
    print(f"✅ Relatório gerado com sucesso!")


def generate_final_report(stats_summary: dict, output_path: str):
    """
    Gera relatório final de texto simples (compatibilidade com main.py)
    
    Args:
        stats_summary: Dicionário com estatísticas do StatsManager
        output_path: Caminho para salvar o relatório
    """
    report = f"""# Relatório de Análise de Vídeo

## Estatísticas Gerais

- Frames Processados: {stats_summary.get('frames_processados', 0)}
- Pessoas Únicas: {stats_summary.get('pessoas_unicas', 0)}
- Anomalias Detectadas: {stats_summary.get('anomalias', 0)}

## Atividades Detectadas

"""
    
    atividades = stats_summary.get('atividades', {})
    if atividades:
        for atividade, count in sorted(atividades.items(), key=lambda x: -x[1]):
            report += f"- {atividade}: {count}\n"
    else:
        report += "Nenhuma atividade detectada.\n"
    
    report += f"\n## Emoções Detectadas\n\n"
    
    emocoes = stats_summary.get('emocoes', {})
    if emocoes:
        for emocao, count in sorted(emocoes.items(), key=lambda x: -x[1]):
            report += f"- {emocao}: {count}\n"
    else:
        report += "Nenhuma emoção detectada.\n"
    
    # Salva relatório
    Path(output_path).write_text(report, encoding='utf-8')
    print(f"📄 Relatório salvo em: {output_path}")
