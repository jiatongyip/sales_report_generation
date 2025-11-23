# Sales Data Automated Report Generator

This project uses Gemini + Python tools to automatically generate a full sales analysis report (including plots and a DOCX file).

## Features
- LLM-driven analysis: trends, top/under-performers, OLS regression, recommendations
- Tooling for:
  - previewing data
  - ranking performers
  - generating bar/line plots
  - running OLS key-driver analysis
  - assembling DOCX reports
- Full report generation workflow via `main.py`

## How It Works
1. `GeminiClient` runs a prompt instructing the LLM to generate a report.
2. The LLM automatically calls functions from `tools.py`:
   - `top_performers`, `under_performers`
   - `preview_data`
   - `generate_simple_plot`
   - `ols_key_driver_analysis`
   - `assemble_simple_report_docx`
3. The final output is a complete `report.docx` with plots and text.

## Run Instructions
### 1. Install Dependencies
`pip install -r requirements.txt`

### 2. Add your API key
`export GEMINI_API_KEY="your_key"`


### 3. Set your input file path in `src/config.py`
Example:
`input_path = "data/sales.xlsx"`


### 4. Run the project
`python main.py`

A fully generated `report.docx` will appear in your project folder.
