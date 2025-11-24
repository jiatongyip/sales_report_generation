
# Sales Report Generation Workflow

This repository implements a **sales report generation workflow** using **LangGraph** and **GeminiClient**. The workflow orchestrates multiple agents to analyze sales data, generate plots, identify key performers, perform statistical analyses, and compile a Word report.

---

## Features

- **Multi-agent workflow**: Each agent focuses on a specific task (sales trends, top performers, key drivers, generic analysis).  
- **Data previewing**: Quickly inspect the dataset.  
- **Flexible plotting**: Bar and line plots with sum/mean aggregation; line plots support grouping.  
- **Statistical analysis**: OLS regression to identify key drivers of sales.  
- **Automated report generation**: Compile all outputs into a structured Word document (`.docx`).  
- **Logging**: Detailed execution logging for debugging and monitoring workflow progress.  

---

## Folder Structure

```
.
├── main.py                 # Entry point for running the workflow
├── src
│   ├── __init__.py         # Workflow setup with nodes, state, and LangGraph orchestration
│   ├── tools.py            # Utility tools: plotting, OLS analysis, top/under performers, DOCX assembly
│   ├── prompts.py          # Agent prompts (e.g., for sales trends, top performers)
│   ├── utils.py            # Utility functions, e.g., load_dataframe, GeminiClient wrapper
│   └── config.py           # Configuration (e.g., input data path)
```

---

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `python-docx`
- `langgraph`
- `langchain-core`

---

## Usage

1. Update the input dataset path in `src/config.py`:

```python
input_path = "data/sales_data.csv"
```

2. Run the main workflow:

```bash
python main.py
```

3. The workflow executes the following steps:

- **Sales Trend Analysis** → generates plots and text summarizing trends.  
- **Top/Under Performer Analysis** → identifies best/worst performers.  
- **Key Driver Analysis** → performs OLS regression to highlight factors driving sales.  
- **Generic Analysis** → optional additional analysis.  
- **Report Compilation** → all outputs are combined into a Word document.  

4. The final report is printed to stdout and saved as a `.docx` file in the configured output path.  

---

## Tools Overview

### `tools.py`

- `preview_data(input_path, head_rows=5)`  
  Preview the first rows of the dataset.  

- `top_performers(input_path, group_by_column, metric_column, n=5)`  
  Return top N performers based on a metric.  

- `under_performers(input_path, group_by_column, metric_column, n=5)`  
  Return bottom N performers.  

- `ols_key_driver_analysis(input_path, target_column, feature_columns)`  
  Perform OLS regression and return coefficients, p-values, and R².  

- `generate_simple_plot(input_path, plot_type, title, x_col, y_col, groupby_col=None, save_path, agg_func="sum")`  
  Generate bar or line plots with sum/mean aggregation; line plots support grouping.  

- `assemble_simple_report_docx(sections_json, report_title='Generated Report', outfile='report.docx')`  
  Compile multiple sections (text + plots) into a Word report.
