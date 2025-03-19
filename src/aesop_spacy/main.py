from pathlib import Path
import json


from .analysis.fable_analyzer import FableAnalyzer

def main():
    """Main entry point for fable analysis."""
    # Set up paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "processed"

    # Initialize analyzer
    analyzer = FableAnalyzer(data_dir)

    # Process all languages
    analyzer.process_all_languages()

    # Run analyses
    for lang in analyzer.fables_by_language:
        pos_dist = analyzer.analyze_pos_distribution(lang)
        print(f"\nPOS distribution for {lang}:")
        for pos, percentage in sorted(pos_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pos}: {percentage:.2f}%")

    # Compare the wolf and lamb fable across languages
    comparison = analyzer.compare_fable_across_languages("1")
    print("\nWolf and Lamb comparison:")
    print(json.dumps(comparison, indent=2))

if __name__ == "__main__":
    main()
