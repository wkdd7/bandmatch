#!/usr/bin/env python3
"""
BandMatch - Main application entry point
Frequency band comparison tool for audio analysis
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main entry point for BandMatch"""
    parser = argparse.ArgumentParser(
        description="BandMatch - Audio frequency band comparison tool",
        epilog="Run 'bandmatch gui' for graphical interface or 'bandmatch cli --help' for command-line options"
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Application mode')
    
    # GUI mode
    gui_parser = subparsers.add_parser('gui', help='Launch graphical interface')
    
    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Command-line interface')
    
    # Parse arguments
    args, remaining = parser.parse_known_args()
    
    if args.mode == 'gui':
        # Launch GUI
        from ui.main_window import main as gui_main
        gui_main()
    elif args.mode == 'cli':
        # Launch CLI with remaining arguments
        from cli import main as cli_main
        # Pass remaining arguments to CLI
        sys.argv = ['bandmatch-cli'] + remaining
        cli_main()
    else:
        # Default to GUI if no mode specified
        print("Launching BandMatch GUI...")
        print("Use 'bandmatch cli --help' for command-line interface")
        from ui.main_window import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()