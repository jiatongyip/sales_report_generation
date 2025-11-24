import google.genai as genai
from google.genai import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import pandas as pd
from IPython.display import Markdown, display
import json
from pydantic import BaseModel, Field
import re

class JSONOutput(BaseModel):
    text: str = Field(description="The analysis results.")
    paths: List[str] = Field(description="Plot paths")

class GeminiClient:
    """
    Simple wrapper for Gemini API client.

    Parameters
    ----------
    model : str
        The Gemini model to use. Defaults to "gemini-2.5-flash".

    Notes
    -----
    The API key is automatically read from environment variables
    `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
    """
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client()

    def generate(self, prompt: str, system_instruction: str, tools: List) -> Any:
        """
        Sends a prompt to Gemini with system instructions and tool schema.

        Parameters
        ----------
        prompt : str
            User instruction or content.
        system_instruction : str
            The agent's role, behavior, and constraints.
        tools : list
            List of callable tool functions following Gemini function-calling spec.

        Returns
        -------
        Any
            The model's response—usually structured JSON or text.
        """
        chat = self.client.chats.create(model=self.model)
        response = chat.send_message(
            message=prompt,
            config={
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="auto")
                ),
                # "response_mime_type": "application/json",
                # "response_json_schema": JSONOutput.model_json_schema(),
            },
        )

        if hasattr(response, 'text'):
            model_output_text = response.text
            # Attempt to parse the model's text output as JSON
            parsed_model_output = extract_json(model_output_text)
            return parsed_model_output
        return response

def load_dataframe(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load a DataFrame from a path (Excel/CSV/JSON) or pass-through if already a DataFrame.


    Supported extensions: .xlsx, .xls, .csv, .json. If a path string has no extension, it is
    treated as CSV by default.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv" or ext == "":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension: {ext}")  

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

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def extract_json(text: str) -> dict:
    """
    Extracts JSON from an LLM response that may include:
    - Code fences (```json ... ```)
    - Extra text before or after JSON
    - Slightly malformed JSON
    - Multiple JSON blocks

    Returns a parsed dict or {} on failure.
    """
    if not text:
        return {}

    # 1. Try direct json.loads first
    try:
        return json.loads(text)
    except Exception:
        pass  # fall through


    # 2. Extract content inside ```json ... ``` blocks
    code_fence_matches = re.findall(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in code_fence_matches:
        cleaned = block.strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass


    # 3. Extract ANY { ... } JSON-like substring
    #    This handles text like "Here is the result: { ... } Thanks!"
    curly_matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for block in curly_matches:
        candidate = _cleanup_json(block)
        try:
            return json.loads(candidate)
        except Exception:
            pass


    # 4. Last attempt: heuristic cleaning of the full text
    candidate = _cleanup_json(text)
    try:
        return json.loads(candidate)
    except Exception as e:
        logger.warning(f"extract_json final attempt failed: {e}")

    return {"text": text}

def _cleanup_json(text: str) -> str:
    """
    Best-effort cleanup for malformed JSON.
    Handles:
    - Trailing commas
    - Triple backticks
    - Comments // or #
    """
    cleaned = text

    # remove code fences
    cleaned = cleaned.replace("```json", "").replace("```", "")

    # remove single-line comments
    cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"#.*?$", "", cleaned, flags=re.MULTILINE)

    # remove trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # collapse multiple spaces/newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
