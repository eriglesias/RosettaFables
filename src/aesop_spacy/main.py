#main.py
"""
Aesop Fable Analysis Tool - Main Entry Point

This module serves as the main entry point for processing and analyzing multilingual
Aesop fables using NLP techniques. It orchestrates the pipeline components to extract,
process, and analyze fable texts across different languages.

Features:
- Processes markdown-formatted fable files into structured JSON
- Analyzes linguistic features across languages and fables
- Generates comparative analysis between different language versions
- Provides command-line configuration options

Usage:
    python main.py [OPTIONS]

Options:
    --data-dir PATH       Path to data directory containing fables
    --output-dir PATH     Path to save processed and analyzed data
    --only-process        Only process raw fables to JSON (skip analysis)
    --only-analyze        Only analyze previously processed fables
    --debug               Enable detailed debug logging

Example:
    python main.py --data-dir ./my_fables --output-dir ./results --debug


Dependencies:
    - pathlib: For cross-platform path handling
    - logging: For application logging
    - argparse: For command-line argument parsing
    - json: For reading/writing structured data
"""

import logging
import argparse
import json
import sys
from pathlib import Path
from aesop_spacy.pipeline.pipeline import FablePipeline
from aesop_spacy.visualization.plots.pos_comparison import POSDistributionPlot
from aesop_spacy.visualization.plots.syntax_comparison import  SyntaxAnalysisPlot
from aesop_spacy.visualization.plots.clustering_plot import ClusteringPlot
from aesop_spacy.visualization.plots.nlp_techniques_plot import NLPTechniquesPlot


# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(description="Process and analyze Aesop's fables")

    parser.add_argument('--data-dir', type=str, help='Data directory path')
    parser.add_argument('--output-dir', type=str, help='Output directory path')
    parser.add_argument('--only-process', action='store_true', help='Only process fables')
    parser.add_argument('--only-analyze', action='store_true', help='Only analyze processed fables')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


def setup_logging(debug_mode=False):
    """Configure application logging based on verbosity level"""
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger("main")


def setup_directories(data_dir, output_dir, logger):
    """Ensure required directories exist and return validated paths"""
    # Create absolute paths if relative paths provided
    base_dir = Path(data_dir) if data_dir else project_root / "data"
    
    # Define input and output paths based on your structure
    input_dir = base_dir / "data_raw"
    output_dir = Path(output_dir) if output_dir else base_dir / "data_handled"

    logger.info("Using input directory: %s", input_dir)
    logger.info("Using output directory: %s", output_dir)

    # Ensure directories exist
    (input_dir / "fables").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)

    # Verify fable source file
    fable_md_path = input_dir / "fables" / "initial_fables.md"
    logger.info("Looking for fables at: %s (exists: %s)",fable_md_path, fable_md_path.exists())

    # Check processed files
    json_files = list((output_dir / "processed").glob("*.json"))
    logger.info("Found %d JSON files in processed directory", len(json_files))

    # Return both input and output paths
    return input_dir, output_dir




def save_analysis_summary(output_dir, results):
    """Save analysis summary to JSON file"""
    summary_file = output_dir / "analysis" / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'languages_processed': list(results.get('pos_distribution', {}).keys()),
            'fables_compared': list(results.get('fable_comparisons', {}).keys()),
            'analysis_types': list(results.keys())
        }, f, indent=2)


def run_pipeline(args, input_dir, output_dir, logger):
    """Run the fable processing pipeline based on command line arguments"""
    pipeline = FablePipeline(input_dir, output_dir)

    if args.only_analyze:
        logger.info("Running analysis only")
        return pipeline.analyze(), False

    elif args.only_process:
        logger.info("Running processing only")
        pipeline.run(use_processed=False)
        return None, False

    else:
        logger.info("Running full pipeline")
        pipeline.run()
        results = pipeline.analyze()
        return results, True

def run_visualizations(output_dir, logger):
    """Run visualizations on analysis results"""    
    logger.info("Running visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(exist_ok=True)
    
    # Create POS distribution visualizations
    pos_plotter = POSDistributionPlot(output_dir=vis_dir)
    
    # Create single language visualizations
    for lang in ['en', 'de', 'nl', 'es', 'grc']:
        try:
            logger.info("Creating POS distribution plot for %s", lang)
            fig, ax = pos_plotter.plot_single_language(lang, top_n=10)
            pos_plotter.save_figure(fig, f'pos_distribution_{lang}.png')
        except Exception as e:
            logger.error("Error creating POS plot for %s: %s", lang, e)
    
    # Create comparison visualizations
    try:
        logger.info("Creating POS comparison plot")
        fig, ax = pos_plotter.plot_language_comparison()
        pos_plotter.save_figure(fig, 'pos_comparison_all_languages.png')
    except Exception as e:
        logger.error("Error creating POS comparison plot: %s", e)
    
    # Create heatmap visualization
    try:
        logger.info("Creating POS heatmap")
        fig, ax = pos_plotter.plot_pos_heatmap()
        pos_plotter.save_figure(fig, 'pos_heatmap_all_languages.png')
    except Exception as e:
        logger.error("Error creating POS heatmap: %s", e)
    
    logger.info("Visualizations complete")

def run_syntax_visualizations(output_dir, logger):
    """Run syntax analysis visualizations"""
    logger.info("Running syntax analysis visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create syntax visualization plotter
    syntax_plotter = SyntaxAnalysisPlot(output_dir=vis_dir)
    
    # Define which fable IDs to visualize
    fable_ids = ['1']  # You can expand this list as needed
    
    # Create single language visualizations
    for fable_id in fable_ids:
        for lang in ['en', 'de', 'nl', 'es', 'grc']:
            try:
                logger.info("Creating dependency plot for fable %s in %s", fable_id, lang)
                fig, ax = syntax_plotter.plot_dependency_frequencies(fable_id, lang)
                syntax_plotter.save_figure(fig, f'dependencies_{fable_id}_{lang}.png')
                
                logger.info("Creating tree shapes plot for fable %s in %s", fable_id, lang)
                fig, ax = syntax_plotter.plot_tree_shapes(fable_id, lang)
                syntax_plotter.save_figure(fig, f'tree_shapes_{fable_id}_{lang}.png')
            except Exception as e:
                logger.error("Error creating syntax plots for %s in %s: %s", fable_id, lang, e)
    
    # Create comparison visualizations
    for fable_id in fable_ids:
        try:
            logger.info("Creating dependency comparison for fable %s", fable_id)
            fig, ax = syntax_plotter.plot_dependency_comparison(fable_id)
            syntax_plotter.save_figure(fig, f'dependency_comparison_{fable_id}.png')
            
            logger.info("Creating tree shapes comparison for fable %s", fable_id)
            fig, ax = syntax_plotter.plot_tree_shapes_comparison(fable_id)
            syntax_plotter.save_figure(fig, f'tree_shapes_comparison_{fable_id}.png')
            
            logger.info("Creating dependency heatmap for fable %s", fable_id)
            fig, ax = syntax_plotter.plot_dependency_heatmap(fable_id)
            syntax_plotter.save_figure(fig, f'dependency_heatmap_{fable_id}.png')
        except Exception as e:
            logger.error("Error creating syntax comparison for fable %s: %s", fable_id, e)
    
    logger.info("Syntax visualizations complete")

def run_clustering_visualizations(output_dir, logger):
    """Run clustering analysis visualizations"""
    logger.info("Running clustering analysis visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create clustering visualization plotter       
    clustering_plotter = ClusteringPlot(output_dir=vis_dir)
    
    # Methods to visualize
    methods = ['kmeans', 'hierarchical', 'dbscan']
    
    for method in methods:
        try:
            logger.info(f"Creating cluster distribution plot for {method}")
            fig, ax = clustering_plotter.plot_cluster_distribution(method)
            clustering_plotter.save_figure(fig, f'cluster_distribution_{method}.png')
            
            logger.info(f"Creating language distribution plot for {method}")
            fig, ax = clustering_plotter.plot_language_distribution(method)
            clustering_plotter.save_figure(fig, f'language_distribution_{method}.png')
            
            logger.info(f"Creating feature importance plot for {method}")
            fig, ax = clustering_plotter.plot_feature_importance(method)
            clustering_plotter.save_figure(fig, f'feature_importance_{method}.png')
            
            logger.info(f"Creating 2D cluster projection for {method}")
            fig, ax = clustering_plotter.plot_2d_cluster_projection(method)
            clustering_plotter.save_figure(fig, f'cluster_projection_{method}.png')
            
            if method == 'kmeans' or method == 'hierarchical':
                logger.info(f"Creating cluster tendency plot for {method}")
                fig, ax = clustering_plotter.plot_cluster_tendency(method)
                clustering_plotter.save_figure(fig, f'cluster_tendency_{method}.png')
        except Exception as e:
            logger.error(f"Error creating clustering plots for {method}: {e}")
    
    # Special plots for hierarchical clustering
    try:
        logger.info("Creating dendrogram plot")
        fig, ax = clustering_plotter.plot_dendrogram()
        clustering_plotter.save_figure(fig, 'dendrogram.png')
    except Exception as e:
        logger.error(f"Error creating dendrogram plot: {e}")
    
    # Compare metrics across methods
    try:
        logger.info("Creating clustering metrics comparison plot")
        fig, ax = clustering_plotter.plot_clustering_metrics()
        clustering_plotter.save_figure(fig, 'clustering_metrics_comparison.png')
    except Exception as e:
        logger.error(f"Error creating clustering metrics plot: {e}")
    
    logger.info("Clustering visualizations complete")

def run_nlp_visualizations(output_dir, logger):
    """Run NLP techniques visualizations"""
    logger.info("Running NLP analysis visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create NLP visualization plotter
    nlp_plotter = NLPTechniquesPlot(output_dir=vis_dir)
    
    # TF-IDF visualizations
    try:
        logger.info("Creating TF-IDF top terms plot")
        fig, ax = nlp_plotter.plot_tfidf_top_terms()
        nlp_plotter.save_figure(fig, 'tfidf_top_terms.png')
        
        logger.info("Creating language-term heatmap")
        fig, ax = nlp_plotter.plot_language_term_heatmap()
        nlp_plotter.save_figure(fig, 'language_term_heatmap.png')
        
        # Create document-term matrices for each language
        for lang in ['en', 'de', 'nl', 'es', 'grc']:
            logger.info(f"Creating document-term matrix for {lang}")
            fig, ax = nlp_plotter.plot_document_term_matrix(language=lang)
            nlp_plotter.save_figure(fig, f'document_term_matrix_{lang}.png')
    except Exception as e:
        logger.error(f"Error creating TF-IDF visualizations: {e}")
    
    # Topic modeling visualizations
    try:
        # Plot each topic's term distribution
        n_topics = 5  # Adjust based on your data
        for topic_id in range(n_topics):
            logger.info(f"Creating term distribution for topic {topic_id}")
            fig, ax = nlp_plotter.plot_topic_term_distribution(topic_id=topic_id)
            nlp_plotter.save_figure(fig, f'topic_{topic_id}_terms.png')
        
        logger.info("Creating document-topic distribution plot")
        fig, ax = nlp_plotter.plot_document_topic_distribution()
        nlp_plotter.save_figure(fig, 'document_topic_distribution.png')
        
        logger.info("Creating language-topic distribution plot")
        fig, ax = nlp_plotter.plot_language_topic_distribution()
        nlp_plotter.save_figure(fig, 'language_topic_distribution.png')
        
        logger.info("Creating topic similarity heatmap")
        fig, ax = nlp_plotter.plot_topic_similarity_heatmap()
        nlp_plotter.save_figure(fig, 'topic_similarity_heatmap.png')
    except Exception as e:
        logger.error(f"Error creating topic modeling visualizations: {e}")
    
    logger.info("NLP visualizations complete")

def main():
    """Main entry point for Aesop fable analysis"""
    # Parse arguments and set up logging
    args = parse_arguments()
    logger = setup_logging(args.debug)

    try:
        # Set up directories
        data_dir, output_dir = setup_directories(args.data_dir, args.output_dir, logger)

        # Run the pipeline
        analysis_results, save_results = run_pipeline(args, data_dir, output_dir, logger)

        if save_results:
            save_analysis_summary(output_dir, analysis_results)
            #run_visualizations(output_dir, logger)
            logger.info("Aesop fable processing complete")

        run_visualizations(output_dir, logger)
        run_syntax_visualizations(output_dir, logger)
        run_clustering_visualizations(output_dir, logger)
        run_nlp_visualizations(output_dir, logger)
   
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
    except ImportError as e:
        logger.error("Module import error: %s", e)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON data: %s", e)
    except ValueError as e:
        logger.error("Data format error: %s", e)
    except PermissionError as e:
        logger.error("Permission denied accessing files: %s", e)
    except OSError as e:
        logger.error("OS error occurred: %s", e)
    except Exception as e:  # Fallback for unexpected errors
        logger.error("Unexpected error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()