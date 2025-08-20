#!/usr/bin/env python3
"""
BandMatch CLI interface
Command-line tool for frequency band comparison analysis
"""

import click
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Tuple

# Import BandMatch modules
from audio_io import AudioLoader
from loudness import LoudnessProcessor
from bands import BandDefinition
from spectrum import SpectrumAnalyzer
from reference import ReferenceCombiner
from comparison import BandComparator
from report import ReportGenerator, ChartGenerator


@click.command()
@click.option('--ref-a', required=True, type=click.Path(exists=True), 
              help='Path to reference A audio file')
@click.option('--ref-b', required=True, type=click.Path(exists=True),
              help='Path to reference B audio file')
@click.option('--target', required=True, type=click.Path(exists=True),
              help='Path to target audio file')
@click.option('--sr', default=48000, type=int,
              help='Target sample rate (default: 48000)')
@click.option('--lufs', default=-14.0, type=float,
              help='Target LUFS for normalization (default: -14)')
@click.option('--bands', default=None, type=str,
              help='Custom band definition (e.g., "20-80,80-250,250-2000,2000-6000,6000-20000")')
@click.option('--preset', default='default', 
              type=click.Choice(['default', 'mastering', 'podcast', 'edm', 'voice']),
              help='Band preset to use')
@click.option('--aggregate', default='median',
              type=click.Choice(['median', 'mean', 'percentile_95']),
              help='Time aggregation method (default: median)')
@click.option('--weights', default='1,1', type=str,
              help='Reference weights as comma-separated values (default: "1,1")')
@click.option('--n-fft', default=4096, type=int,
              help='FFT size for analysis (default: 4096)')
@click.option('--hop-length', default=None, type=int,
              help='Hop length for STFT (default: n_fft/4)')
@click.option('--json', 'json_out', default=None, type=click.Path(),
              help='Output JSON report to file')
@click.option('--csv', 'csv_out', default=None, type=click.Path(),
              help='Output CSV report to file')
@click.option('--pdf', 'pdf_out', default=None, type=click.Path(),
              help='Output PDF report to file')
@click.option('--charts/--no-charts', default=True,
              help='Generate visualization charts')
@click.option('--verbose', is_flag=True,
              help='Verbose output')
def main(ref_a: str, ref_b: str, target: str, sr: int, lufs: float,
         bands: Optional[str], preset: str, aggregate: str, weights: str,
         n_fft: int, hop_length: Optional[int], json_out: Optional[str],
         csv_out: Optional[str], pdf_out: Optional[str], charts: bool,
         verbose: bool):
    """
    BandMatch - Frequency band comparison tool
    
    Analyzes target audio against two reference tracks and reports
    band-by-band energy differences.
    """
    
    try:
        # Parse weights
        weight_values = [float(w.strip()) for w in weights.split(',')]
        if len(weight_values) != 2:
            raise ValueError("Weights must be two comma-separated values")
        
        if verbose:
            click.echo("BandMatch Analysis Starting...")
            click.echo(f"Reference A: {ref_a}")
            click.echo(f"Reference B: {ref_b}")
            click.echo(f"Target: {target}")
            click.echo(f"Target LUFS: {lufs}")
            click.echo(f"Sample Rate: {sr} Hz")
            click.echo(f"FFT Size: {n_fft}")
            click.echo("")
        
        # Initialize components
        loader = AudioLoader(target_sr=sr)
        loudness_processor = LoudnessProcessor(target_lufs=lufs)
        
        # Set up bands
        if bands:
            band_definition = BandDefinition.from_string(bands)
        else:
            band_definition = BandDefinition(preset=preset)
        
        # Initialize analyzer
        analyzer = SpectrumAnalyzer(
            n_fft=n_fft,
            hop_length=hop_length,
            band_definition=band_definition
        )
        
        # Load audio files
        if verbose:
            click.echo("Loading audio files...")
        
        ref_a_audio, _, ref_a_meta = loader.load_audio(ref_a, mono=True)
        ref_b_audio, _, ref_b_meta = loader.load_audio(ref_b, mono=True)
        target_audio, _, target_meta = loader.load_audio(target, mono=True)
        
        # Check audio lengths
        min_duration = 10.0  # seconds
        for audio, meta, name in [(ref_a_audio, ref_a_meta, "Reference A"),
                                  (ref_b_audio, ref_b_meta, "Reference B"),
                                  (target_audio, target_meta, "Target")]:
            if meta['duration'] < min_duration:
                click.echo(f"Warning: {name} is short ({meta['duration']:.1f}s), "
                          f"results may be less reliable", err=True)
        
        # Normalize loudness
        if verbose:
            click.echo(f"Normalizing to {lufs} LUFS...")
        
        ref_a_norm, ref_a_gain = loudness_processor.normalize_to_target(ref_a_audio, sr, lufs)
        ref_b_norm, ref_b_gain = loudness_processor.normalize_to_target(ref_b_audio, sr, lufs)
        target_norm, target_gain = loudness_processor.normalize_to_target(target_audio, sr, lufs)
        
        if verbose:
            click.echo(f"  Reference A gain: {ref_a_gain:+.1f} dB")
            click.echo(f"  Reference B gain: {ref_b_gain:+.1f} dB")
            click.echo(f"  Target gain: {target_gain:+.1f} dB")
            click.echo("")
        
        # Analyze spectra
        if verbose:
            click.echo("Analyzing frequency spectra...")
        
        ref_a_bands = analyzer.analyze_audio(ref_a_norm, sr, aggregate)
        ref_b_bands = analyzer.analyze_audio(ref_b_norm, sr, aggregate)
        target_bands = analyzer.analyze_audio(target_norm, sr, aggregate)
        
        # Combine references
        if verbose:
            click.echo("Combining references...")
        
        combiner = ReferenceCombiner()
        baseline_bands = combiner.combine_references(
            ref_a_bands, ref_b_bands, tuple(weight_values)
        )
        
        # Check for warnings
        warnings = combiner.get_warnings()
        if warnings and verbose:
            click.echo("\nReference warnings:")
            for warning in warnings:
                click.echo(f"  - {warning}", err=True)
            click.echo("")
        
        # Compare target to baseline
        if verbose:
            click.echo("Comparing target to baseline...")
        
        comparator = BandComparator()
        comparisons = comparator.compare_bands(baseline_bands, target_bands)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("ANALYSIS RESULTS")
        click.echo("="*60)
        click.echo(f"{'Band':<12} {'Baseline':>10} {'Target':>10} {'Delta':>10} {'Judgment':<15}")
        click.echo("-"*60)
        
        for comp in comparisons:
            # Color code based on judgment
            if abs(comp.delta_db) < 1.0:
                judgment_str = click.style(comp.judgment.value, fg='green')
            elif abs(comp.delta_db) < 3.0:
                judgment_str = click.style(comp.judgment.value, fg='yellow')
            else:
                judgment_str = click.style(comp.judgment.value, fg='red')
            
            click.echo(f"{comp.band_name:<12} {comp.baseline_db:>10.1f} "
                      f"{comp.target_db:>10.1f} {comp.delta_db:>+10.1f} {judgment_str}")
        
        # Overall summary
        click.echo("-"*60)
        summary = comparator.generate_overall_summary(comparisons)
        match_score = comparator.calculate_overall_match_score(comparisons)
        click.echo(f"\nOverall: {summary}")
        click.echo(f"Match Score: {match_score:.1f}%")
        
        # Generate reports
        metadata = {
            'ref_a_file': Path(ref_a).name,
            'ref_b_file': Path(ref_b).name,
            'target_file': Path(target).name,
            'sample_rate': sr,
            'target_lufs': lufs,
            'weights': weight_values,
            'warnings': [str(w) for w in warnings]
        }
        
        report_gen = ReportGenerator()
        
        # Generate charts if requested
        chart_paths = {}
        if charts:
            if verbose:
                click.echo("\nGenerating charts...")
            
            chart_gen = ChartGenerator()
            bar_path = chart_gen.generate_bar_chart(comparisons)
            radar_path = chart_gen.generate_radar_chart(comparisons)
            chart_paths = {'bar_chart': bar_path, 'radar_chart': radar_path}
            
            if verbose:
                click.echo(f"  Bar chart: {bar_path}")
                click.echo(f"  Radar chart: {radar_path}")
        
        # Save reports
        if json_out:
            report_gen.generate_json_report(comparisons, metadata, Path(json_out))
            click.echo(f"\nJSON report saved: {json_out}")
        
        if csv_out:
            report_gen.generate_csv_report(comparisons, Path(csv_out))
            click.echo(f"CSV report saved: {csv_out}")
        
        if pdf_out:
            report_gen.generate_pdf_report(comparisons, metadata, chart_paths, Path(pdf_out))
            click.echo(f"PDF report saved: {pdf_out}")
        
        # EQ suggestions
        click.echo("\n" + "="*60)
        click.echo("EQ RECOMMENDATIONS")
        click.echo("="*60)
        
        needs_adjustment = [c for c in comparisons if c.judgment.value != "적정"]
        if needs_adjustment:
            for comp in sorted(needs_adjustment, key=lambda x: abs(x.delta_db), reverse=True)[:5]:
                click.echo(f"• {comp.band_name}: {comp.eq_suggestion}")
        else:
            click.echo("No adjustments needed - target matches baseline well!")
        
        click.echo("\nAnalysis complete!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()