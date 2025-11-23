import os
import json
from typing import Any, Dict, Callable, List, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime
from src.tools import tools
# ----------------------------- LLM  ------------------------------
from IPython.display import Markdown, display

def print_history(chat):
  for content in chat.get_history():
      display(Markdown("###" + content.role + ":"))
      for part in content.parts:
          if part.text:
              display(Markdown(part.text))
          if part.function_call:
              print("Function call: {", part.function_call, "}")
          if part.function_response:
              print("Function response: {", part.function_response, "}")
      print("-" * 80)

try:
    from google import genai
    from google.genai import types
    import os

    class GeminiClient:
        def __init__(self, model: str = "gemini-2.5-flash"):
            self.model = model
            # The client picks up the API key automatically from the environment variable
            # GEMINI_API_KEY or GOOGLE_API_KEY.
            self.client = genai.Client()

        def generate(self, prompt: str, system_instruction: str, tools: List) -> str:
            chat = self.client.chats.create(model=self.model)

            response = chat.send_message(
                message=prompt,
                config={
                    "system_instruction": system_instruction,
                    "tools": tools,
                    "tool_config" : types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="auto"
                        )
                    )
                }
            )

            print_history(chat)
            return response.text
except ImportError:
    print("The 'google-genai' library is not installed. Please run 'pip install google-genai'.")
    GeminiClient = None
except Exception as e:
    # Catching general exceptions related to API key or network issues
    print(f"An error occurred: {e}")
    GeminiClient = None


# ---------------------------- AGENT SETUP --------------------------
llm = GeminiClient()
system_instruction = "You are a sales data report generation specialist. Access the sales data through the tools, and your task is to write a comprehensive report."
prompt = """
Write a comprehensive report with the following sections:

    (1) Executive Summary
    (2) Analysis 
- Identify and describe sales performance trend over time (by region), attaching line plot.
- Highlight top-performing and underperforming models, with a bar plot.
- Highlight top-performing and underperforming markets, with a bar plot.
- Explore key drivers of sales (e.g., price, region, transmission, fuel type or model type) using OLS regression.
- Include 1–2 additional insights, such as hypothesis 

    (3) Recommendations

Lastly, ensure that you generate the DOCX report.
"""

# ------------------------------- MAIN ------------------------------

def main():
    llm = GeminiClient()
    llm.generate(prompt = prompt, system_instruction=system_instruction, tools = tools)


if __name__ == '__main__':
    main()
