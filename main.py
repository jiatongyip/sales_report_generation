from src import ReportState, run_workflow
from src.config import input_path
import logging
import sys

# ------------------------------- MAIN ------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the logging level (e.g., INFO, DEBUG)
if not logger.handlers:
    # Create a StreamHandler that writes to sys.stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__ == "__main__":
    init_state: ReportState = {"input_data_path": input_path}
    final = run_workflow(init_state)
    print(final.get("final_report_text", "<No report generated>"))

