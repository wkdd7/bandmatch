"""
Main GUI window for BandMatch
PyQt6-based graphical interface
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog, QComboBox,
    QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QProgressBar, QMessageBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QAction, QIcon, QPixmap, QPalette, QColor

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from audio_io import AudioLoader
from loudness import LoudnessProcessor
from bands import BandDefinition
from spectrum import SpectrumAnalyzer
from reference import ReferenceCombiner
from comparison import BandComparator
from report import ReportGenerator, ChartGenerator


class FileDropZone(QFrame):
    """Drop zone widget for audio files"""
    
    fileDropped = pyqtSignal(str)
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.file_path = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setMinimumHeight(100)
        
        layout = QVBoxLayout()
        
        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        
        # File info
        self.file_label = QLabel("Drop audio file here or click to browse")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)
        
        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_btn)
        
        # File details
        self.details_label = QLabel("")
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.details_label)
        
        self.setLayout(layout)
        
        # Styling
        self.update_style()
    
    def update_style(self):
        """Update visual style based on state"""
        if self.file_path:
            self.setStyleSheet("""
                FileDropZone {
                    border: 2px solid #27AE60;
                    border-radius: 5px;
                    background-color: #E8F8F5;
                }
            """)
        else:
            self.setStyleSheet("""
                FileDropZone {
                    border: 2px dashed #BDC3C7;
                    border-radius: 5px;
                    background-color: #ECF0F1;
                }
                FileDropZone:hover {
                    border-color: #3498DB;
                    background-color: #EBF5FB;
                }
            """)
    
    def dragEnterEvent(self, event):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                FileDropZone {
                    border: 2px solid #3498DB;
                    border-radius: 5px;
                    background-color: #EBF5FB;
                }
            """)
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.update_style()
    
    def dropEvent(self, event):
        """Handle file drop"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.set_file(files[0])
    
    def browse_file(self):
        """Open file browser"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.title}",
            "",
            "Audio Files (*.wav *.aiff *.aif *.flac *.mp3 *.m4a);;All Files (*.*)"
        )
        if file_path:
            self.set_file(file_path)
    
    def set_file(self, file_path: str):
        """Set the file path"""
        self.file_path = file_path
        path = Path(file_path)
        self.file_label.setText(path.name)
        self.file_label.setStyleSheet("color: black;")
        
        # Try to get file info
        try:
            loader = AudioLoader()
            info = loader.get_audio_info(file_path)
            self.details_label.setText(
                f"Duration: {info['duration']:.1f}s | "
                f"SR: {info['sample_rate']} Hz | "
                f"Ch: {info['channels']}"
            )
        except:
            self.details_label.setText("")
        
        self.update_style()
        self.fileDropped.emit(file_path)
    
    def clear(self):
        """Clear the file"""
        self.file_path = None
        self.file_label.setText("Drop audio file here or click to browse")
        self.file_label.setStyleSheet("color: gray;")
        self.details_label.setText("")
        self.update_style()


class AnalysisWorker(QThread):
    """Worker thread for analysis"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, ref_a: str, ref_b: str, target: str, settings: dict):
        super().__init__()
        self.ref_a = ref_a
        self.ref_b = ref_b
        self.target = target
        self.settings = settings
    
    def run(self):
        """Run analysis"""
        try:
            self.progress.emit(10, "Loading audio files...")
            
            # Initialize components
            loader = AudioLoader(target_sr=self.settings['sample_rate'])
            loudness_processor = LoudnessProcessor(target_lufs=self.settings['target_lufs'])
            
            # Load audio
            ref_a_audio, _, ref_a_meta = loader.load_audio(self.ref_a, mono=True)
            self.progress.emit(20, "Loaded Reference A")
            
            ref_b_audio, _, ref_b_meta = loader.load_audio(self.ref_b, mono=True)
            self.progress.emit(30, "Loaded Reference B")
            
            target_audio, _, target_meta = loader.load_audio(self.target, mono=True)
            self.progress.emit(40, "Loaded Target")
            
            # Normalize
            self.progress.emit(50, "Normalizing loudness...")
            sr = self.settings['sample_rate']
            lufs = self.settings['target_lufs']
            
            ref_a_norm, _ = loudness_processor.normalize_to_target(ref_a_audio, sr, lufs)
            ref_b_norm, _ = loudness_processor.normalize_to_target(ref_b_audio, sr, lufs)
            target_norm, _ = loudness_processor.normalize_to_target(target_audio, sr, lufs)
            
            # Analyze
            self.progress.emit(60, "Analyzing spectra...")
            
            band_definition = BandDefinition(preset=self.settings['band_preset'])
            analyzer = SpectrumAnalyzer(
                n_fft=self.settings['n_fft'],
                band_definition=band_definition
            )
            
            ref_a_bands = analyzer.analyze_audio(ref_a_norm, sr, self.settings['aggregate'])
            ref_b_bands = analyzer.analyze_audio(ref_b_norm, sr, self.settings['aggregate'])
            target_bands = analyzer.analyze_audio(target_norm, sr, self.settings['aggregate'])
            
            # Combine references
            self.progress.emit(70, "Combining references...")
            combiner = ReferenceCombiner()
            baseline_bands = combiner.combine_references(ref_a_bands, ref_b_bands)
            warnings = combiner.get_warnings()
            
            # Compare
            self.progress.emit(80, "Comparing target to baseline...")
            comparator = BandComparator()
            comparisons = comparator.compare_bands(baseline_bands, target_bands)
            
            # Generate charts
            self.progress.emit(90, "Generating visualizations...")
            chart_gen = ChartGenerator()
            bar_chart = chart_gen.generate_bar_chart(comparisons)
            radar_chart = chart_gen.generate_radar_chart(comparisons)
            
            # Package results
            results = {
                'comparisons': comparisons,
                'warnings': warnings,
                'summary': comparator.generate_overall_summary(comparisons),
                'match_score': comparator.calculate_overall_match_score(comparisons),
                'bar_chart': str(bar_chart),
                'radar_chart': str(radar_chart),
                'metadata': {
                    'ref_a': ref_a_meta,
                    'ref_b': ref_b_meta,
                    'target': target_meta
                }
            }
            
            self.progress.emit(100, "Analysis complete!")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class BandMatchMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_results = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("BandMatch - Frequency Band Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # File input section
        input_group = QGroupBox("Audio Files")
        input_layout = QHBoxLayout()
        
        self.ref_a_drop = FileDropZone("Reference A")
        self.ref_b_drop = FileDropZone("Reference B")
        self.target_drop = FileDropZone("Target")
        
        input_layout.addWidget(self.ref_a_drop)
        input_layout.addWidget(self.ref_b_drop)
        input_layout.addWidget(self.target_drop)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Settings section
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout()
        
        # LUFS setting
        settings_layout.addWidget(QLabel("Target LUFS:"))
        self.lufs_combo = QComboBox()
        self.lufs_combo.addItems(["-23", "-18", "-16", "-14", "-12", "-9"])
        self.lufs_combo.setCurrentText("-14")
        settings_layout.addWidget(self.lufs_combo)
        
        # Band preset
        settings_layout.addWidget(QLabel("Band Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["default", "mastering", "podcast", "edm", "voice"])
        settings_layout.addWidget(self.preset_combo)
        
        # Aggregation method
        settings_layout.addWidget(QLabel("Aggregation:"))
        self.aggregate_combo = QComboBox()
        self.aggregate_combo.addItems(["median", "mean", "percentile_95"])
        settings_layout.addWidget(self.aggregate_combo)
        
        settings_layout.addStretch()
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                font-weight: bold;
                padding: 10px 30px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        settings_layout.addWidget(self.analyze_btn)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        main_layout.addWidget(self.progress_label)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        
        # Table tab
        self.results_table = QTableWidget()
        self.results_tabs.addTab(self.results_table, "Results Table")
        
        # Charts tab
        self.charts_widget = QWidget()
        self.results_tabs.addTab(self.charts_widget, "Visualizations")
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "Summary")
        
        main_layout.addWidget(self.results_tabs)
        
        # Menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        export_json = QAction("Export JSON...", self)
        export_json.triggered.connect(self.export_json)
        file_menu.addAction(export_json)
        
        export_csv = QAction("Export CSV...", self)
        export_csv.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv)
        
        export_pdf = QAction("Export PDF...", self)
        export_pdf.triggered.connect(self.export_pdf)
        file_menu.addAction(export_pdf)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
    
    def run_analysis(self):
        """Run the analysis"""
        # Check files are selected
        if not all([self.ref_a_drop.file_path, 
                   self.ref_b_drop.file_path,
                   self.target_drop.file_path]):
            QMessageBox.warning(self, "Missing Files", 
                              "Please select all three audio files.")
            return
        
        # Prepare settings
        settings = {
            'sample_rate': 48000,
            'target_lufs': float(self.lufs_combo.currentText()),
            'band_preset': self.preset_combo.currentText(),
            'aggregate': self.aggregate_combo.currentText(),
            'n_fft': 4096
        }
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.analyze_btn.setEnabled(False)
        self.results_tabs.setVisible(False)
        
        # Create and start worker
        self.worker = AnalysisWorker(
            self.ref_a_drop.file_path,
            self.ref_b_drop.file_path,
            self.target_drop.file_path,
            settings
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        
        self.worker.start()
    
    def update_progress(self, value: int, message: str):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def show_results(self, results: dict):
        """Show analysis results"""
        self.current_results = results
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Populate table
        comparisons = results['comparisons']
        self.results_table.setRowCount(len(comparisons))
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Band", "Baseline (dB)", "Target (dB)", "Delta (dB)", "Judgment"]
        )
        
        for i, comp in enumerate(comparisons):
            self.results_table.setItem(i, 0, QTableWidgetItem(comp.band_name))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{comp.baseline_db:.1f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{comp.target_db:.1f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{comp.delta_db:+.1f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(comp.judgment.value))
            
            # Color code delta
            delta_item = self.results_table.item(i, 3)
            if abs(comp.delta_db) < 1.0:
                delta_item.setBackground(QColor(46, 204, 113))  # Green
            elif abs(comp.delta_db) < 3.0:
                delta_item.setBackground(QColor(241, 196, 15))  # Yellow
            else:
                delta_item.setBackground(QColor(231, 76, 60))   # Red
        
        self.results_table.resizeColumnsToContents()
        
        # Show charts
        charts_layout = QHBoxLayout()
        self.charts_widget.setLayout(charts_layout)
        
        if 'bar_chart' in results:
            bar_label = QLabel()
            bar_pixmap = QPixmap(results['bar_chart'])
            bar_label.setPixmap(bar_pixmap.scaled(500, 400, Qt.AspectRatioMode.KeepAspectRatio))
            charts_layout.addWidget(bar_label)
        
        if 'radar_chart' in results:
            radar_label = QLabel()
            radar_pixmap = QPixmap(results['radar_chart'])
            radar_label.setPixmap(radar_pixmap.scaled(500, 400, Qt.AspectRatioMode.KeepAspectRatio))
            charts_layout.addWidget(radar_label)
        
        # Show summary
        summary_text = f"""
<h2>Analysis Summary</h2>
<p><b>Overall:</b> {results['summary']}</p>
<p><b>Match Score:</b> {results['match_score']:.1f}%</p>

<h3>EQ Recommendations:</h3>
<ul>
"""
        for comp in comparisons:
            if comp.judgment.value != "적정":
                summary_text += f"<li><b>{comp.band_name}:</b> {comp.eq_suggestion}</li>"
        
        summary_text += "</ul>"
        
        if results['warnings']:
            summary_text += "<h3>Warnings:</h3><ul>"
            for warning in results['warnings']:
                summary_text += f"<li>{warning}</li>"
            summary_text += "</ul>"
        
        self.summary_text.setHtml(summary_text)
        
        # Show results
        self.results_tabs.setVisible(True)
    
    def show_error(self, error_msg: str):
        """Show error message"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{error_msg}")
    
    def export_json(self):
        """Export results as JSON"""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "", "JSON Files (*.json)"
        )
        
        if file_path:
            report_gen = ReportGenerator()
            report_gen.generate_json_report(
                self.current_results['comparisons'],
                {'warnings': [str(w) for w in self.current_results['warnings']]},
                Path(file_path)
            )
            QMessageBox.information(self, "Export Complete", f"JSON exported to {file_path}")
    
    def export_csv(self):
        """Export results as CSV"""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            report_gen = ReportGenerator()
            report_gen.generate_csv_report(
                self.current_results['comparisons'],
                Path(file_path)
            )
            QMessageBox.information(self, "Export Complete", f"CSV exported to {file_path}")
    
    def export_pdf(self):
        """Export results as PDF"""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            report_gen = ReportGenerator()
            charts = {
                'bar_chart': Path(self.current_results.get('bar_chart', '')),
                'radar_chart': Path(self.current_results.get('radar_chart', ''))
            }
            report_gen.generate_pdf_report(
                self.current_results['comparisons'],
                {'warnings': [str(w) for w in self.current_results['warnings']]},
                charts,
                Path(file_path)
            )
            QMessageBox.information(self, "Export Complete", f"PDF exported to {file_path}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("BandMatch")
    
    # Set application style
    app.setStyle("Fusion")
    
    window = BandMatchMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()