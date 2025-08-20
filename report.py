"""
Report generation module for BandMatch
Handles JSON, CSV, and PDF report creation
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from comparison import BandComparison
from reference import ReferenceWarning


class ReportGenerator:
    """Generates analysis reports in various formats"""
    
    def __init__(self):
        """Initialize report generator"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_json_report(self,
                            comparisons: List[BandComparison],
                            metadata: Dict,
                            output_path: Optional[Path] = None) -> str:
        """
        Generate JSON report
        
        Args:
            comparisons: List of band comparisons
            metadata: Additional metadata
            output_path: Optional output file path
            
        Returns:
            JSON string
        """
        report_data = {
            'timestamp': self.timestamp,
            'metadata': metadata,
            'bands': [c.band_name for c in comparisons],
            'baseline_db': [round(c.baseline_db, 1) for c in comparisons],
            'target_db': [round(c.target_db, 1) for c in comparisons],
            'delta_db': [round(c.delta_db, 1) for c in comparisons],
            'judgement': [c.judgment.value for c in comparisons],
            'eq_suggestions': [c.eq_suggestion for c in comparisons],
            'confidence': [round(c.confidence, 2) for c in comparisons],
            'lufs': metadata.get('lufs', {}),
            'warnings': metadata.get('warnings', [])
        }
        
        json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json_str, encoding='utf-8')
        
        return json_str
    
    def generate_csv_report(self,
                           comparisons: List[BandComparison],
                           output_path: Optional[Path] = None) -> str:
        """
        Generate CSV report
        
        Args:
            comparisons: List of band comparisons
            output_path: Optional output file path
            
        Returns:
            CSV string
        """
        rows = []
        headers = ['Band', 'Baseline_dB', 'Target_dB', 'Delta_dB', 'Judgement', 'EQ_Suggestion']
        
        for comp in comparisons:
            rows.append([
                comp.band_name,
                round(comp.baseline_db, 1),
                round(comp.target_db, 1),
                round(comp.delta_db, 1),
                comp.judgment.value,
                comp.eq_suggestion
            ])
        
        # Generate CSV string
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        csv_str = output.getvalue()
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(csv_str, encoding='utf-8')
        
        return csv_str
    
    def generate_pdf_report(self,
                           comparisons: List[BandComparison],
                           metadata: Dict,
                           charts: Optional[Dict[str, Path]] = None,
                           output_path: Optional[Path] = None):
        """
        Generate PDF report
        
        Args:
            comparisons: List of band comparisons
            metadata: Additional metadata
            charts: Dictionary of chart image paths
            output_path: Output file path
        """
        if output_path is None:
            output_path = Path(f"bandmatch_report_{self.timestamp}.pdf")
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86C1'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("BandMatch Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata section
        story.append(Paragraph("Analysis Details", styles['Heading2']))
        metadata_data = [
            ['Date:', datetime.now().strftime("%Y-%m-%d %H:%M")],
            ['Reference A:', metadata.get('ref_a_file', 'N/A')],
            ['Reference B:', metadata.get('ref_b_file', 'N/A')],
            ['Target:', metadata.get('target_file', 'N/A')],
            ['Sample Rate:', f"{metadata.get('sample_rate', 48000)} Hz"],
            ['Target LUFS:', f"{metadata.get('target_lufs', -14)} LUFS"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 30))
        
        # Results table
        story.append(Paragraph("Band Analysis Results", styles['Heading2']))
        
        table_data = [['Band', 'Baseline (dB)', 'Target (dB)', 'Delta (dB)', 'Judgment']]
        for comp in comparisons:
            row = [
                comp.band_name,
                f"{comp.baseline_db:.1f}",
                f"{comp.target_db:.1f}",
                f"{comp.delta_db:+.1f}",
                comp.judgment.value
            ]
            table_data.append(row)
        
        results_table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        
        # Color code based on judgment
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        # Add color coding for delta values
        for i, comp in enumerate(comparisons, 1):
            if abs(comp.delta_db) < 1.0:
                color = colors.green
            elif abs(comp.delta_db) < 3.0:
                color = colors.yellow
            elif abs(comp.delta_db) < 6.0:
                color = colors.orange
            else:
                color = colors.red
            table_style.append(('BACKGROUND', (3, i), (3, i), color))
        
        results_table.setStyle(TableStyle(table_style))
        story.append(results_table)
        story.append(Spacer(1, 30))
        
        # Add charts if provided
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis", styles['Heading2']))
            
            for chart_name, chart_path in charts.items():
                if chart_path and Path(chart_path).exists():
                    img = Image(str(chart_path), width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
        
        # EQ Recommendations
        story.append(Paragraph("EQ Recommendations", styles['Heading2']))
        for comp in comparisons:
            if comp.judgment.value != "적정":
                para = Paragraph(f"<b>{comp.band_name}:</b> {comp.eq_suggestion}", styles['Normal'])
                story.append(para)
        
        # Warnings if any
        if metadata.get('warnings'):
            story.append(Spacer(1, 20))
            story.append(Paragraph("Warnings", styles['Heading2']))
            for warning in metadata['warnings']:
                para = Paragraph(f"• {warning}", styles['Normal'])
                story.append(para)
        
        # Build PDF
        doc.build(story)


class ChartGenerator:
    """Generates visualization charts"""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize chart generator
        
        Args:
            style: Matplotlib style to use
        """
        if style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
    
    def generate_bar_chart(self,
                          comparisons: List[BandComparison],
                          output_path: Optional[Path] = None) -> Path:
        """
        Generate bar chart of delta dB values
        
        Args:
            comparisons: List of band comparisons
            output_path: Optional output path
            
        Returns:
            Path to generated chart
        """
        if output_path is None:
            output_path = Path(f"bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bands = [c.band_name for c in comparisons]
        deltas = [c.delta_db for c in comparisons]
        
        # Color based on value
        colors_list = []
        for delta in deltas:
            if abs(delta) < 1.0:
                colors_list.append('#2ECC71')  # Green
            elif abs(delta) < 3.0:
                colors_list.append('#F39C12')  # Orange
            elif abs(delta) < 6.0:
                colors_list.append('#E74C3C')  # Red
            else:
                colors_list.append('#8B0000')  # Dark red
        
        bars = ax.bar(bands, deltas, color=colors_list, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{delta:+.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delta (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Target vs Baseline Frequency Response', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([min(deltas) - 2, max(deltas) + 2])
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#2ECC71', label='Optimal (< 1 dB)'),
            mpatches.Patch(color='#F39C12', label='Slight (1-3 dB)'),
            mpatches.Patch(color='#E74C3C', label='Moderate (3-6 dB)'),
            mpatches.Patch(color='#8B0000', label='Extreme (> 6 dB)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_radar_chart(self,
                           comparisons: List[BandComparison],
                           output_path: Optional[Path] = None) -> Path:
        """
        Generate radar chart comparing baseline and target
        
        Args:
            comparisons: List of band comparisons
            output_path: Optional output path
            
        Returns:
            Path to generated chart
        """
        if output_path is None:
            output_path = Path(f"radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        bands = [c.band_name for c in comparisons]
        baseline_db = [c.baseline_db for c in comparisons]
        target_db = [c.target_db for c in comparisons]
        
        # Normalize to 0-1 range for better visualization
        all_values = baseline_db + target_db
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        baseline_norm = [(v - min_val) / range_val for v in baseline_db]
        target_norm = [(v - min_val) / range_val for v in target_db]
        
        # Compute angle for each band
        angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
        
        # Close the plot
        baseline_norm += baseline_norm[:1]
        target_norm += target_norm[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='#3498DB')
        ax.fill(angles, baseline_norm, alpha=0.25, color='#3498DB')
        
        ax.plot(angles, target_norm, 'o-', linewidth=2, label='Target', color='#E74C3C')
        ax.fill(angles, target_norm, alpha=0.25, color='#E74C3C')
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(bands)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Spectral Balance Comparison', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path