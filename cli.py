"""Command-line interface for CosmoNet."""

import argparse
import sys
import pandas as pd
from pathlib import Path
from .cosmonet_classifier import CosmoNetClassifier


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CosmoNet: Astronomical Light Curve Classification"
    )
    
    parser.add_argument(
        "--train-metadata",
        required=True,
        help="Path to training metadata CSV file"
    )
    
    parser.add_argument(
        "--train-lightcurves",
        required=True,
        help="Path to training light curves CSV file"
    )
    
    parser.add_argument(
        "--test-metadata",
        help="Path to test metadata CSV file (optional)"
    )
    
    parser.add_argument(
        "--test-lightcurves",
        help="Path to test light curves CSV file (optional)"
    )
    
    parser.add_argument(
        "--output",
        default="cosmonet_results",
        help="Output directory for results (default: cosmonet_results)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for sequence preparation (default: 1000)"
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    train_meta_path = Path(args.train_metadata)
    train_lc_path = Path(args.train_lightcurves)
    
    if not train_meta_path.exists():
        print(f"Error: Training metadata file not found: {train_meta_path}")
        sys.exit(1)
    
    if not train_lc_path.exists():
        print(f"Error: Training light curves file not found: {train_lc_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Run analysis
    print("Starting CosmoNet analysis...")
    print(f"Output will be saved to: {output_dir}")
    
    try:
        # Initialize classifier
        classifier = CosmoNetClassifier(random_state=42)
        
        # Load data
        classifier.load_data(str(train_meta_path), str(train_lc_path))
        
        # Load test data if provided
        test_meta = None
        test_lc = None
        if args.test_metadata and args.test_lightcurves:
            test_meta_path = Path(args.test_metadata)
            test_lc_path = Path(args.test_lightcurves)
            
            if test_meta_path.exists() and test_lc_path.exists():
                test_meta = pd.read_csv(test_meta_path)
                test_lc = pd.read_csv(test_lc_path)
                print(f"Loaded test data: {test_meta.shape[0]} objects")
        
        # Run pipeline
        classifier.define_classes()
        classifier.engineer_features()
        sequences, targets, object_ids = classifier.prepare_sequences(
            sample_size=args.sample_size
        )
        classifier.train_models(n_folds=args.n_folds)
        results = classifier.evaluate_models()
        
        # Save results
        classifier.save_results(output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Overall accuracy: {results.get('accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()