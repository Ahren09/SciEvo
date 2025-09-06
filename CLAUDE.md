# SciEvo Analysis Documentation

This document provides comprehensive documentation of the analysis capabilities available in the SciEvo project and their corresponding Python commands.

## 📊 Analysis Commands

### Citation Analysis

#### Basic Citation Analysis
```bash
# Run citation analysis with default settings
python analysis/analyze_citation.py

# Run citation analysis with title focus
python analysis/analyze_citation.py --feature_name title
```

**Purpose**: Analyzes citation patterns, age of citations, and citation diversity across scientific papers.

**Key Features**:
- Calculates Age of Citations (AoC) for papers
- Measures citation diversity using Simpson's, Shannon's, and Gini indices
- Analyzes temporal patterns in citation behavior
- Generates subject-specific and topic-specific metrics

### Keyword Analysis

#### Keyword Extraction
```bash
# Basic keyword extraction
python analysis/keyword_extraction.py

# LLM-based keyword extraction
python analysis/keyword_extraction_llm.py

# LLM-based extraction for specific years
python analysis/keyword_extraction_llm.py --year 2010
python analysis/keyword_extraction_llm.py --year 2020
python analysis/keyword_extraction_llm.py --year 2023

# LLM-based extraction with title focus
python analysis/keyword_extraction_llm.py --feature_name title

# LLM-based extraction with title and abstract
python analysis/keyword_extraction_llm.py --feature_name title --abstract

# N-gram based keyword extraction
python analysis/keyword_extraction_ngram.py
```

**Purpose**: Extracts and analyzes keywords from scientific papers using various methods.

**Key Features**:
- Traditional keyword extraction methods
- LLM-based extraction for improved accuracy
- Year-specific analysis for temporal trends
- Title-only and title+abstract extraction modes
- N-gram based extraction for phrase identification

#### Keyword Ranking
```bash
# Analyze keyword ranks and occurrences
python analysis/rank_keywords_by_number_of_occurrences.py
```

**Purpose**: Ranks keywords by their frequency and importance across the dataset.

## 📁 Dataset Processing Commands

### Citation Graph Construction
```bash
# Construct citation graph from references
python dataset/construct_citation_graph.py
```

**Purpose**: Builds citation networks from paper references for network analysis.

### Keyword Hypergraph Construction
```bash
# Construct keyword hypergraph
python dataset/construct_keyword_hypergraph.py

# Construct keyword hypergraph using NetworkX
python dataset/construct_keyword_hypergraph.py --networkx
```

**Purpose**: Creates hypergraphs based on keyword co-occurrence patterns.

### Data Management
```bash
# Create Spark table for data processing
python dataset/create_spark_table.py

# Download arXiv papers
python dataset/download_arxiv_paper.py

# Index table by tags
python dataset/index_table_by_tags.py

# Merge keywords from different sources
python dataset/merge_keywords.py
```

**Purpose**: Various data processing and management tasks for the SciEvo dataset.

## 🤖 Model Commands

### Graph Neural Networks
```bash
# Run GCN (Graph Convolutional Network)
python model/gcn.py

# Run GCN with title focus
python model/gcn.py --feature_name title

# Run GConvGRU model
python model/gconvgru.py
```

**Purpose**: Implements and runs graph neural network models for citation and keyword analysis.

### Word Embeddings
```bash
# Train word2vec model
python model/word2vec.py

# Train word2vec with title focus
python model/word2vec.py --feature_name title

# Train word2vec with summary focus
python model/word2vec.py --summary
```

**Purpose**: Trains word embedding models for text analysis and similarity computation.

### Alignment Analysis
```bash
# Run Procrustes analysis
python model/procrustes.py
```

**Purpose**: Performs alignment analysis between different embedding spaces.

## 🎨 Visualization Commands

### Age of Citation (AoC) Visualizations
```bash
# Plot AoC by subjects
python visualization/plot_aoc_by_subjects.py

# Plot AoC by subjects with title focus
python visualization/plot_aoc_by_subjects.py --feature_name title

# Plot AoC for all subjects
python visualization/plot_aoc_all_subjects.py

# Plot AoC for computer science specifically
python visualization/plot_aoc_cs.py
```

**Purpose**: Visualizes age of citation patterns across different subject areas.

### Citation Analysis Visualizations
```bash
# Plot citation diversity
python visualization/plot_citation_diversity.py

# Plot citation diversity with title focus
python visualization/plot_citation_diversity.py --feature_name title

# Plot citation flow using Sankey diagrams
python visualization/plot_citation_flow_sankey_chord.py

# Plot citation flow with Sankey diagram focus
python visualization/plot_citation_flow_sankey_chord.py --sankey

# Plot citation graph
python visualization/plot_citation_graph.py

# Plot citation graph for Blender
python visualization/plot_citation_graph_blender.py
```

**Purpose**: Visualizes citation patterns, diversity, and flow between different fields.

### Keyword Trajectory Visualizations
```bash
# Plot keyword trajectories
python visualization/plot_keyword_traj.py

# Plot keyword trajectories with GCN model
python visualization/plot_keyword_traj.py --gcn --trajectory 1

# Get keyword trajectory coordinates
python visualization/get_keyword_trajectory_coords.py

# Get keyword trajectory coordinates with GCN
python visualization/get_keyword_trajectory_coords.py --gcn

# Plot single trajectory
python visualization/plot_one_trajectory.py

# Plot with 7B model
python visualization/plot_keyword_traj.py --model 7b
```

**Purpose**: Visualizes how keywords evolve and move through different topics over time.

### Statistical Visualizations
```bash
# Plot keyword occurrences
python visualization/plot_keywords_occurrences.py

# Plot number of papers and keywords
python visualization/plot_number_of_papers_and_keywords.py

# Plot paper trends
python visualization/plot_paper_trend.py

# Plot keyword ranks
python visualization/plot_ranks_of_keywords.py

# Plot trend of keyword occurrences
python visualization/plot_trend_of_keywords_occurrence.py

# Plot top-k word ratio
python visualization/plot_topk_word_ratio.py
```

**Purpose**: Creates statistical visualizations of keyword and paper trends.

## 🔧 Utility Commands

### Data Processing Utilities
```bash
# Metadata extraction
python utility/metadataextractionsec.py

# Time utilities
python utility/utils_time.py
```

**Purpose**: Provides utility functions for data processing and time-related operations.

## 📝 Script Commands

### Main Analysis Runner
```bash
# Run main analysis pipeline
python scripts/run_analysis.py
```

**Purpose**: Executes the complete analysis pipeline with all components.

## 🐍 General Python Debugging

```bash
# Debug current file with arguments
# Use VS Code debugger with argument picker
```

**Purpose**: General debugging configuration for any Python file with argument support.

## Usage Notes

### Environment Setup
All commands are configured with the proper Python path (`PYTHONPATH`) to ensure modules can be imported correctly. The working directory is set to the project root for all commands.

### Command Line Arguments
Most commands support various command-line arguments for customization:
- `--feature_name title`: Focus analysis on paper titles only
- `--abstract`: Include abstract text in analysis
- `--year`: Specify year for temporal analysis
- `--gcn`: Use Graph Convolutional Network models
- `--model`: Specify model type (e.g., "7b")
- `--networkx`: Use NetworkX for graph operations
- `--sankey`: Generate Sankey diagram visualizations

### Output Files
Analysis results are typically saved to:
- `outputs/stats/`: Statistical analysis results
- `outputs/figures/`: Generated visualizations
- `dataset/`: Processed data files

### Data Dependencies
Most analysis commands depend on:
- arXiv paper data
- Semantic Scholar references
- Keyword extraction results
- Citation graph data

Ensure all required data files are present before running analysis commands.
