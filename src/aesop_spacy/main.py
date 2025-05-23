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
from aesop_spacy.visualization.plots.word_frequency_plot import WordFrequencyPlot
from aesop_spacy.visualization.plots.entity_analysis_plot import EntityAnalysisPlot
from aesop_spacy.visualization.plots.moral_analysis_plot import MoralAnalysisPlot

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
    parser.add_argument('--verify-models', action='store_true', help='Verify required models')
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
    """
    Ensure required directories exist with clean separation of concerns.
    
    Directory structure philosophy:
    - data_raw/     : Contains only raw, unprocessed input files
    - data_handled/ : Contains ALL processed outputs (JSON, analysis, figures)
    
    This creates a clear data flow: raw -> handled
    """
    # Create absolute paths if relative paths provided
    base_dir = Path(data_dir) if data_dir else project_root / "data"

    # Define clean directory structure
     # Raw input files only
    input_dir = base_dir / "data_raw"
    # All processed outputs
    output_dir = Path(output_dir) if output_dir else base_dir / "data_handled"

    logger.info("Using input directory: %s", input_dir)
    logger.info("Using output directory: %s", output_dir)

    # Create input structure (for raw files only)
    (input_dir / "fables").mkdir(parents=True, exist_ok=True)
    # Create output structure (for all processed data)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)  # Processed fables JSON
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)   # Analysis results
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)    # Visualizations
    # Verify input files exist
    fable_md_path = input_dir / "fables" / "initial_fables.md"
    logger.info("Looking for raw fables at: %s (exists: %s)", fable_md_path, fable_md_path.exists())
    # Check existing processed files (in the correct location)
    json_files = list((output_dir / "processed").glob("*.json"))
    logger.info("Found %d processed JSON files in output directory", len(json_files))
    # Log the clean directory structure
    logger.debug("Directory structure:")
    logger.debug("  Raw input:  %s", input_dir)
    logger.debug("  Processed:  %s", output_dir / "processed")
    logger.debug("  Analysis:   %s", output_dir / "analysis")
    logger.debug("  Figures:    %s", output_dir / "figures")

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

def run_word_frequency_visualizations(output_dir, logger):
    """Run word frequency visualizations"""
    logger.info("Running word frequency visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create word frequency plotter
    word_freq_plotter = WordFrequencyPlot(output_dir=vis_dir)
    
    # Define which fable IDs to visualize
    fable_ids = ['1', '2', '3', '4', '5']  # Adjust based on your data
    
    # Create visualizations for each fable
    for fable_id in fable_ids:
        try:
            logger.info(f"Creating top words plot for fable {fable_id}")
            fig, ax = word_freq_plotter.plot_top_words(fable_id)
            word_freq_plotter.save_figure(fig, f'top_words_fable_{fable_id}.png')
            
            logger.info(f"Creating word frequency heatmap for fable {fable_id}")
            fig, ax = word_freq_plotter.plot_word_frequency_heatmap(fable_id)
            word_freq_plotter.save_figure(fig, f'word_freq_heatmap_fable_{fable_id}.png')
            
            logger.info(f"Creating shared vocabulary plot for fable {fable_id}")
            fig, ax = word_freq_plotter.plot_shared_vocabulary(fable_id)
            word_freq_plotter.save_figure(fig, f'shared_vocab_fable_{fable_id}.png')
            
            logger.info(f"Creating word distribution comparison for fable {fable_id}")
            fig, ax = word_freq_plotter.plot_word_distribution_comparison(fable_id)
            word_freq_plotter.save_figure(fig, f'word_distrib_fable_{fable_id}.png')
        except Exception as e:
            logger.error(f"Error creating word frequency visualizations for fable {fable_id}: {e}")
    
    # Create lexical richness comparison across all fables
    try:
        logger.info("Creating lexical richness comparison")
        fig, ax = word_freq_plotter.plot_lexical_richness_comparison(fable_ids)
        word_freq_plotter.save_figure(fig, 'lexical_richness_comparison.png')
    except Exception as e:
        logger.error(f"Error creating lexical richness comparison: {e}")
    
    logger.info("Word frequency visualizations complete")

def run_entity_visualizations(output_dir, logger):
    """Run entity analysis visualizations"""
    logger.info("Running entity analysis visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create entity plotter
    entity_plotter = EntityAnalysisPlot(output_dir=vis_dir)
    
    # Create entity distribution plots for each language
    for lang in ['en', 'de', 'nl', 'es', 'grc']:
        try:
            logger.info(f"Creating entity distribution plot for {lang}")
            fig, ax = entity_plotter.plot_entity_distribution(lang)
            entity_plotter.save_figure(fig, f'entity_distribution_{lang}.png')
            
            logger.info(f"Creating entity examples visualization for {lang}")
            fig, ax = entity_plotter.plot_entity_examples(lang)
            entity_plotter.save_figure(fig, f'entity_examples_{lang}.png')
        except Exception as e:
            logger.error(f"Error creating entity plots for {lang}: {e}")
    
    # Create comparison visualizations
    try:
        logger.info("Creating entity comparison plot")
        fig, ax = entity_plotter.plot_entity_comparison()
        entity_plotter.save_figure(fig, 'entity_comparison_all_languages.png')
        
        logger.info("Creating entity heatmap")
        fig, ax = entity_plotter.plot_entity_heatmap()
        entity_plotter.save_figure(fig, 'entity_heatmap_all_languages.png')
        
        logger.info("Creating top entities plot")
        fig, ax = entity_plotter.plot_top_entities(n=15)
        entity_plotter.save_figure(fig, 'top_entities_all_languages.png')
    except Exception as e:
        logger.error(f"Error creating entity comparison plots: {e}")
    
    logger.info("Entity visualizations complete")

def run_moral_visualizations(output_dir, logger):
    """Run moral analysis visualizations"""
    logger.info("Running moral analysis visualizations...")
    
    # Create visualization directory if it doesn't exist
    vis_dir = output_dir / "figures"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create moral plotter
    moral_plotter = MoralAnalysisPlot(output_dir=vis_dir)
    
    # Create moral theme plots for each language
    for lang in ['en', 'de', 'nl', 'es', 'grc']:
        try:
            logger.info(f"Creating moral themes plot for {lang}")
            fig, ax = moral_plotter.plot_moral_themes(lang)
            moral_plotter.save_figure(fig, f'moral_themes_{lang}.png')
            
            logger.info(f"Creating moral inferences visualization for {lang}")
            fig, ax = moral_plotter.plot_moral_inferences(lang)
            moral_plotter.save_figure(fig, f'moral_inferences_{lang}.png')
            
            logger.info(f"Creating keywords heatmap for {lang}")
            fig, ax = moral_plotter.plot_keywords_heatmap(lang)
            moral_plotter.save_figure(fig, f'moral_keywords_{lang}.png')
        except Exception as e:
            logger.error(f"Error creating moral plots for {lang}: {e}")
    
    # Create comparison visualizations
    try:
        logger.info("Creating moral comparison plot")
        fig, ax = moral_plotter.plot_moral_comparison()
        moral_plotter.save_figure(fig, 'moral_comparison_all_languages.png')
        
        logger.info("Creating moral similarity heatmap")
        fig, ax = moral_plotter.plot_moral_similarity_heatmap()
        moral_plotter.save_figure(fig, 'moral_similarity_heatmap.png')
        
        logger.info("Creating theme consistency plot")
        fig, ax = moral_plotter.plot_theme_consistency()
        moral_plotter.save_figure(fig, 'moral_theme_consistency.png')
    except Exception as e:
        logger.error(f"Error creating moral comparison plots: {e}")
    
    # Create individual fable moral comparisons
    for fable_id in ['1', '2', '3', '4', '5']:  # Adjust based on your data
        try:
            logger.info(f"Creating moral comparison for fable {fable_id}")
            fig, ax = moral_plotter.plot_moral_comparison(fable_id)
            moral_plotter.save_figure(fig, f'moral_comparison_fable_{fable_id}.png')
        except Exception as e:
            logger.error(f"Error creating moral comparison for fable {fable_id}: {e}")
    
    logger.info("Moral visualizations complete")




def main():
    """Main entry point for Aesop fable analysis"""
    # Parse arguments and set up logging
    args = parse_arguments()
    logger = setup_logging(args.debug)

    try:
        if args.verify_models:
            from aesop_spacy.models.model_manager import verify_models
            logger.info("Verifying required models...")
            verification = verify_models(check_optional=True)

            if verification['missing']:
                logger.warning("Missing models detected. Please install with:")
                for cmd in verification['install_commands']:
                    logger.warning("  %s", cmd)
            else:
                logger.info("All required models are installed.")
            return 
        
        
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
        run_word_frequency_visualizations(output_dir, logger)
        run_entity_visualizations(output_dir, logger)
        run_moral_visualizations(output_dir, logger)
   
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