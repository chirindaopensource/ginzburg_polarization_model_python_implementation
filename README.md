# README.md

# A Flexible Measure of Voter Polarization: A Python Implementation

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%233776AB.svg?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.07770-b31b1b.svg)](https://arxiv.org/abs/2507.07770)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2507.07770-blue)](https://doi.org/10.48550/arXiv.2507.07770)
[![Research](https://img.shields.io/badge/Research-Computational%20Social%20Science-green)](https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation)
[![Discipline](https://img.shields.io/badge/Discipline-Political%20Economy-blue)](https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation)
[![Methodology](https://img.shields.io/badge/Methodology-Distributional%20Analysis-orange)](https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation)
[![Data Source](https://img.shields.io/badge/Data%20Source-ANES-lightgrey)](https://electionstudies.org/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation)

**Repository:** https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent** implementation of the research methodology from the 2025 paper entitled **"A Flexible Measure of Voter Polarization"** by:

*   Boris Ginzburg

The project provides a robust, end-to-end Python pipeline for computing and analyzing the flexible polarization index, `P(F, x*)`. This measure moves beyond traditional, mean-centric metrics like variance to provide a high-resolution diagnostic tool. It allows an analyst to measure polarization around any specified point in the ideological spectrum, enabling the precise identification and tracking of specific fault lines within an electorate.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: execute_full_research_project](#key-callable-execute_full_research_project)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "A Flexible Measure of Voter Polarization." The core of this repository is the iPython Notebook `ginzburg_polarization_index_implementation_draft.ipynb`, which contains a comprehensive suite of functions to compute the `P(F, x*)` index, analyze its dynamics, and compare it against traditional measures.

Traditional measures of polarization, such as variance, describe the dispersion of an electorate around a single point—the mean. This can mask crucial asymmetric dynamics. For instance, the center-right may be polarizing while the center-left is coalescing, an effect that a single variance number might miss. The Ginzburg index solves this by making the reference point `x*` a flexible parameter, effectively allowing an analyst to "scan" the entire ideological spectrum for polarization.

This codebase enables researchers, political analysts, and data scientists to:
-   Rigorously compute the `P(F, x*)` index for any ideological distribution.
-   Analyze the evolution of polarization at different points on the political spectrum over time.
-   Systematically identify the ideological "cleavage points" where polarization has increased the most.
-   Compare the insights from this flexible measure against traditional metrics.
-   Replicate and extend the findings of the original research paper.

## Theoretical Background

The implemented methods are grounded in distributional analysis and integral calculus, applied to survey data.

**A Flexible Definition of Polarization:** The framework begins by defining what it means for one distribution `F̂` to be more polarized than another `F` around a specific point `x*`. This occurs if the probability mass in *any* interval containing `x*` is smaller under `F̂`. This leads to a practical single-crossing condition on the Cumulative Distribution Functions (CDFs).

**The Polarization Index `P(F, x*)`:** To provide a scalar measure, the paper defines the polarization index. The formula is designed to increase as probability mass shifts away from the central point `x*` towards both tails of the distribution.
$P(F, x^*) := \frac{\int_{\min\{X\}}^{x^*} F(x) dx}{x^* - \min\{X\}} - \frac{\int_{x^*}^{\max\{X\}} F(x) dx}{\max\{X\} - x^*} + 1$
For discrete survey data, the integrals are calculated exactly as the sum of areas of rectangles under the empirical CDF's step function.

**Cleavage Point Finder:** The framework can be inverted. Instead of specifying `x*` and measuring polarization, the pipeline can perform a grid search across all possible `x*` values to find the one where the percentage increase in `P(F, x*)` between two time periods is maximized. This identifies the most significant emerging ideological fault line.

## Features

The provided iPython Notebook (`ginzburg_polarization_index_implementation_draft.ipynb`) implements the full research pipeline, including:

-   **Parameter Validation:** Rigorous checks for all input data and configurations to ensure methodological compliance.
-   **Data Cleansing:** Robust handling of survey-specific missing value codes.
-   **Weighted Statistics:** Correct application of survey weights for all calculations, including CDF construction and traditional measures.
-   **Exact `P(F, x*)` Calculation:** A numerically stable and mathematically precise implementation of the polarization index for discrete data.
-   **Automated Analysis Suite:** Functions to systematically analyze temporal trends, election-year effects, and comparative performance against traditional metrics.
-   **Cleavage Point Finder:** An algorithm to scan the ideological spectrum and identify points of maximum polarization increase.
-   **Theoretical Extensions:** Computational models for affective polarization and the effects of issue salience.
-   **Robustness Checks:** A framework for testing the sensitivity of results to key methodological choices (e.g., weighted vs. unweighted analysis).
-   **Publication-Quality Visualization:** A suite of functions to generate the key figures from the paper.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preparation (Tasks 1-2):** The pipeline ingests raw ANES data, cleans it by handling missing value codes, and constructs a weighted empirical Cumulative Distribution Function (CDF) for each survey year and ideological scale.
2.  **Polarization Calculation (Tasks 3-4):** It computes the `P(F, x*)` index for every specified `(year, scale, x*)` combination using the formula from Definition 2.
3.  **Temporal and Event Analysis (Tasks 5-6):** The pipeline analyzes the evolution of `P(F, x*)` over time and its short-term changes around elections, replicating the analysis in Figures 1, 2, and 3 of the paper.
4.  **Cleavage Point Identification (Tasks 7-8):** It implements the grid search algorithm to find the `x*` that maximizes the percentage increase in polarization between two periods.
5.  **Comparative Analysis (Tasks 9-10):** The pipeline computes traditional measures (e.g., variance) and systematically compares their trends to those of the `P(F, x*)` index at various points, quantifying where the measures converge and diverge.

## Core Components (Notebook Structure)

The `ginzburg_polarization_index_implementation_draft.ipynb` notebook is structured as a logical pipeline with modular functions for each task:

-   **Task 0: `validate_parameters`**: The initial quality gate for all inputs.
-   **Task 1: `clean_anes_data`**: Handles data quality and missing codes.
-   **Task 2: `preprocess_for_polarization`**: The core CDF generation engine.
-   **Task 3-4: `calculate_polarization_index`, `compute_all_polarization_measures`**: The main polarization index calculation.
-   **Task 5-10**: A suite of analysis functions for temporal, election, cleavage, and comparative analysis.
-   **Task 11: `calculate_affective_polarization`, `simulate_issue_salience_effect`**: Implementation of the theoretical models.
-   **Task 12-14**: High-level orchestrators (`run_polarization_pipeline`, `run_robustness_analysis`, `execute_full_research_project`) that run the entire workflow.

## Key Callable: execute_full_research_project

The central function in this project is `execute_full_research_project`. It orchestrates the entire analytical workflow from raw data to final results, including robustness checks and report generation.

```python
def execute_full_research_project(
    anes_df: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes the complete, end-to-end polarization research project.

    This master orchestrator function serves as the single entry point to run
    the entire analysis suite, from raw data to final report assets. It encapsulates
    the full research workflow, including the baseline analysis, robustness checks,
    and the generation of all tables and visualizations.

    Args:
        anes_df (pd.DataFrame): The raw ANES survey data.
        params (Dict[str, Any]): A comprehensive dictionary containing all
            parameters required for every stage of the analysis.

    Returns:
        Dict[str, Any]: A master dictionary containing the complete project results.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   Core dependencies as listed in `requirements.txt`: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation.git
    cd ginzburg_polarization_model_python_implementation
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies from `requirements.txt`:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The primary input is a `pandas.DataFrame` with the following required columns:
-   `respondent_id`: A unique identifier for each respondent.
-   `year`: A string/object representing the survey wave (e.g., `'2004a'`, `'2016b'`).
-   `left_right`: The respondent's self-placement on the 0-10 left-right scale.
-   `liberal_conservative`: The respondent's self-placement on the 1-7 liberal-conservative scale.
-   `weight`: The full-sample survey weight for that respondent.

## Usage

### **User Guide: Deploying the End-to-End Polarization Analysis Pipeline**

This section provides a practical, step-by-step guide to utilizing the `execute_full_research_project` function. This function is the single entry point to the entire analytical library, designed for robust, reproducible, and comprehensive analysis of voter polarization.

#### **Step 1: Data Acquisition and Preparation**

The pipeline requires a single, consolidated `pandas.DataFrame` as its primary data input. This DataFrame must be prepared *before* calling the main function and must adhere to a strict schema.

*   **Action:** The analyst must first acquire the necessary survey data (e.g., from the ANES Data Center). The data from multiple years and waves should be merged into one file.
*   **Schema:** The resulting DataFrame must contain the following columns with these exact names:
    *   `respondent_id`: A unique identifier for each respondent.
    *   `year`: A string or object representing the survey wave (e.g., `'2004a'`, `'2016b'`).
    *   `left_right`: The respondent's self-placement on the 0-10 left-right scale.
    *   `liberal_conservative`: The respondent's self-placement on the 1-7 liberal-conservative scale.
    *   `weight`: The full-sample survey weight provided by ANES for that specific survey wave.

**Code Snippet: Creating a Mock DataFrame**

For demonstration purposes, we will construct a small, mock `DataFrame` that conforms to this schema. In a real-world scenario, this `anes_df` would be the result of loading and merging actual ANES `.dta` or `.csv` files.

```python
import pandas as pd
import numpy as np

# In a real application, this DataFrame would be loaded from a file.
# df = pd.read_csv("path/to/your/merged_anes_data.csv")
# For this example, we create a mock DataFrame.

data = {
    'respondent_id': [f'R_{i}' for i in range(100)],
    'year': np.random.choice(['2012', '2016a', '2016b', '2020'], 100),
    'left_right': np.random.choice(list(range(11)) + [98], 100), # Include a missing code
    'liberal_conservative': np.random.choice(list(range(1, 8)) + [99], 100), # Include a missing code
    'weight': np.random.uniform(0.5, 2.5, 100)
}
anes_df = pd.DataFrame(data)

print("Sample of the prepared input DataFrame:")
print(anes_df.head())
```

#### **Step 2: Constructing the Master Parameters Dictionary**

The `execute_full_research_project` function is controlled by a single, comprehensive parameters dictionary. This object encapsulates every tunable setting for the entire analysis, ensuring perfect reproducibility. We will construct this dictionary section by section, using the default values specified in the original project brief.

**Code Snippet: Defining the `params` Dictionary**

```python
# Initialize the master parameters dictionary.
params = {}

# --- Central Points of Interest (x*) ---
# These are the points around which polarization will be measured for each scale.
params['central_points_params'] = {
    "left_right_x_stars": [1, 2, 3, 4, 5, 6, 7, 8, 9], # Note: Excludes boundaries 0 and 10
    "liberal_conservative_x_stars": [2, 3, 4, 5, 6] # Note: Excludes boundaries 1 and 7
}

# --- Policy Space Boundaries ---
# These define the theoretical min and max of each ideological scale.
params['boundaries_params'] = {
    "left_right_boundaries": {'min': 0, 'max': 10},
    "liberal_conservative_boundaries": {'min': 1, 'max': 7}
}

# --- Integration Parameters ---
# Defines the numerical method for the P(F,x*) calculation.
# Note: The current implementation is optimized for 'trapezoidal' on discrete data.
params['integration_params'] = {
    'method': 'trapezoidal',
    'num_points': 1000
}

# --- Cleavage Finder Parameters ---
# Defines the time periods to scan for the maximum increase in polarization.
params['cleavage_finder_params'] = {
    'time_points': [('2016a', '2020')], # Using years available in our mock data
    'potential_x_stars': params['central_points_params']
}

# --- Analysis-Specific Parameters ---
# Defines midpoints for partitioning temporal plots and centrist definitions.
params['midpoints'] = {
    'left_right': 5,
    'liberal_conservative': 4
}
params['centrist_definitions'] = {
    'left_right': [4, 5, 6],
    'liberal_conservative': [3, 4, 5]
}
params['election_years_for_analysis'] = [2016] # Using year available in mock data

# --- Theoretical Extension Parameters ---
# Defines parameters for the issue salience simulation.
params['salience_simulation_params'] = {
    'salience_alphas': [0.1, 0.3, 0.5, 0.7, 0.9],
    'common_value_dist': {'low': 4.8, 'high': 5.2},
    'divisive_issue_dist': {'low': 0, 'high': 10},
    'polarization_params': {
        'x_star': 5, # The x* to calculate polarization around in the simulation
        'boundaries': params['boundaries_params']['left_right_boundaries']
    },
    'random_seed': 42 # For reproducibility
}

# --- Optional: ANES Missing Value Codes ---
# This can be customized if different codes are used in the data.
params['missing_value_map'] = {
    'left_right': [98, 99],
    'liberal_conservative': [98, 99]
}

print("\nMaster parameters dictionary constructed successfully.")
```

#### **Step 3: Executing the Pipeline and Inspecting Results**

With the input `DataFrame` and the `params` dictionary prepared, the entire research project can be executed with a single function call. The function will print its progress through the various stages of the analysis.

**Code Snippet: Running the Master Orchestrator**

```python
# First, ensure the full library of functions is loaded in your environment.
# from ginzburg_polarization_model_python_implementation import execute_full_research_project

# Execute the entire research project.
# This single call runs validation, cleaning, all calculations, analyses,
# robustness checks, and reporting.
project_results = execute_full_research_project(
    anes_df=anes_df,
    params=params
)
```

The returned `project_results` object is a deeply nested dictionary containing every artifact generated during the run. An analyst can now programmatically access any piece of the analysis for inspection, custom plotting, or further work.

**Code Snippet: Accessing and Inspecting Key Outputs**

```python
# --- Inspecting the Main Analysis Results ---
print("\n--- Example: Inspecting Key Outputs from the Main Analysis ---")

# Access the main results dictionary
main_analysis_results = project_results['main_analysis']['results']

# 1. View the summary of the cleavage point analysis
print("\nCleavage Point Analysis Summary:")
cleavage_summary = main_analysis_results['cleavage_analysis']
print(cleavage_summary)

# 2. View the summary of the comparative analysis (correlation)
print("\nCorrelation Summary (Traditional vs. Flexible Measures):")
correlation_summary = main_analysis_results['comparative_framework']['correlation_summary']
print(correlation_summary)

# 3. Access a specific plot (e.g., temporal trends for the left-right scale)
report_assets = project_results['main_analysis']['report_assets']
temporal_plot_lr = report_assets['plots']['temporal_trends_left_right']
# To display the plot in a Jupyter environment:
# temporal_plot_lr.show()
# To save the plot to a file:
# temporal_plot_lr.savefig("temporal_trends_left_right.png", dpi=300)
print("\nTemporal trend plot for 'left_right' scale has been generated.")

# --- Inspecting the Robustness Analysis Results ---
print("\n--- Example: Comparing Robustness Check Results ---")

# Compare the cleavage point found in the weighted vs. unweighted scenarios
try:
    weighted_cleavage = project_results['robustness_analysis']['weighted']['cleavage_analysis']
    unweighted_cleavage = project_results['robustness_analysis']['unweighted']['cleavage_analysis']

    print("\nCleavage points from 'weighted' analysis:")
    print(weighted_cleavage[['cleavage_point', 'relative_position']])

    print("\nCleavage points from 'unweighted' analysis:")
    print(unweighted_cleavage[['cleavage_point', 'relative_position']])
except KeyError:
    print("\nRobustness check for one or more scenarios may have failed.")

```
This example provides a complete, end-to-end workflow, demonstrating how a user can prepare their data, configure the analysis, execute the entire pipeline with one command, and access the structured, high-value outputs for interpretation. 

## Output Structure

The `execute_full_research_project` function returns a single, comprehensive dictionary with the following top-level keys:

-   `main_analysis`: Contains the results of the primary pipeline run.
    -   `results`: A dictionary of all key data artifacts and analytical DataFrames (e.g., `polarization_results`, `cleavage_analysis`).
    -   `report_assets`: A dictionary containing the generated `matplotlib` figures and formatted LaTeX/HTML tables.
-   `robustness_analysis`: Contains the results from the different robustness scenarios (e.g., 'weighted' vs. 'unweighted'), allowing for direct comparison.

## Project Structure

```
ginzburg_polarization_model_python_implementation/
│
├── ginzburg_polarization_index_implementation_draft.ipynb  # Main implementation notebook
├── requirements.txt                                      # Python package dependencies
├── LICENSE                                                 # MIT license file
└── README.md                                               # This documentation file
```

## Customization

The pipeline is highly customizable via the master `params` dictionary. Users can easily modify:
-   The lists of `x_stars` to analyze for each scale.
-   The `time_points` for the cleavage finder.
-   The `midpoints` and `centrist_definitions` for analysis and comparison.
-   The parameters for the theoretical simulations.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{ginzburg2025flexible,
  title={A Flexible Measure of Voter Polarization},
  author={Ginzburg, Boris},
  journal={arXiv preprint arXiv:2507.07770},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of the Ginzburg Flexible Polarization Model. 
GitHub repository: https://github.com/chirindaopensource/ginzburg_polarization_model_python_implementation
```

## Acknowledgments

-   Credit to Boris Ginzburg for the novel theoretical framework and the flexible polarization measure.
-   Thanks to the developers of the `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn` libraries, which are the foundational pillars of this analytical pipeline.

--

*This README was generated based on the structure and content of `ginzburg_polarization_index_implementation_draft.ipynb` and follows best practices for research software documentation.*
