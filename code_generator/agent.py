import google.generativeai as genai
import openai
from typing import Optional, Dict, Any
import logging
import asyncio
from google.api_core.exceptions import InternalServerError as GeminiInternalServerError, GoogleAPIError, DeadlineExceeded as GeminiDeadlineExceeded
from openai import APIError as OpenAIAPIError, APITimeoutError as OpenAITimeoutError, APIConnectionError as OpenAIAPIConnectionError, RateLimitError as OpenAIRateLimitError, BadRequestError
import time
import re
import os

from core.interfaces import CodeGeneratorInterface # BaseAgent is implicitly inherited via CodeGeneratorInterface
from config import settings

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # super().__init__(config) # BaseAgent init is implicitly called by CodeGeneratorInterface if it inherits BaseAgent
        self.config = config or {}
        self.llm_provider = settings.LLM_PROVIDER
        self.model_name: Optional[str] = None
        self.client: Any = None # For OpenAI and compatible providers
        self.max_retries = settings.API_MAX_RETRIES
        self.retry_delay_seconds = settings.API_RETRY_DELAY_SECONDS
        self.openai_compatible_num_ctx: Optional[int] = None

        self.default_temperature = 0.6
        self.default_top_p = 0.9 # For Gemini, OpenAI uses top_p
        self.default_top_k = 40  # Gemini specific

        logger.info(f"Initializing CodeGeneratorAgent with LLM_PROVIDER: '{self.llm_provider}'")

        if self.llm_provider == "gemini":
            if not settings.GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not properly configured in .env for the 'gemini' provider.")
            if not settings.GEMINI_MODEL_NAME:
                 raise ValueError("GEMINI_MODEL_NAME is not configured in .env for the 'gemini' provider.")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model_name = self._clean_model_name(settings.GEMINI_MODEL_NAME)
            self.generation_config_gemini = genai.types.GenerationConfig(
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                top_k=self.default_top_k
            )
            logger.info(f"Gemini provider configured. Model: '{self.model_name}'")
        elif self.llm_provider == "openai":
            if not settings.OPENAI_API_KEY or "YOUR_OPENAI_API_KEY" in settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not properly configured in .env for the 'openai' provider.")
            if not settings.OPENAI_MODEL_NAME:
                raise ValueError("OPENAI_MODEL_NAME is not configured in .env for the 'openai' provider.")
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = self._clean_model_name(settings.OPENAI_MODEL_NAME)
            logger.info(f"OpenAI provider configured. Model: '{self.model_name}'")
        elif self.llm_provider == "openai_compatible":
            if not settings.OPENAI_COMPATIBLE_ENDPOINT_URL:
                raise ValueError("OPENAI_COMPATIBLE_ENDPOINT_URL is not configured in .env for 'openai_compatible' provider.")
            if not settings.OPENAI_COMPATIBLE_MODEL_NAME:
                raise ValueError("OPENAI_COMPATIBLE_MODEL_NAME is not configured in .env for 'openai_compatible' provider.")
            
            self.model_name = self._clean_model_name(settings.OPENAI_COMPATIBLE_MODEL_NAME)
            self.client = openai.AsyncOpenAI(
                base_url=settings.OPENAI_COMPATIBLE_ENDPOINT_URL,
                api_key=settings.OPENAI_COMPATIBLE_API_KEY # Can be "not-needed", "ollama", or an actual key
            )
            if settings.OPENAI_COMPATIBLE_NUM_CTX:
                try:
                    self.openai_compatible_num_ctx = int(settings.OPENAI_COMPATIBLE_NUM_CTX)
                    logger.info(f"OpenAI-compatible context window (num_ctx) set to: {self.openai_compatible_num_ctx}")
                except ValueError:
                    logger.warning(f"Invalid value for OPENAI_COMPATIBLE_NUM_CTX: '{settings.OPENAI_COMPATIBLE_NUM_CTX}'. Must be an integer. Ignoring num_ctx.")
                    self.openai_compatible_num_ctx = None
            
            logger.info(f"OpenAI-compatible provider configured. Endpoint: '{settings.OPENAI_COMPATIBLE_ENDPOINT_URL}', Model: '{self.model_name}', API Key Used: '{settings.OPENAI_COMPATIBLE_API_KEY != 'not-needed' and bool(settings.OPENAI_COMPATIBLE_API_KEY)}'")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{self.llm_provider}'. Choose 'gemini', 'openai', or 'openai_compatible'.")

    def _clean_model_name(self, name: Optional[str]) -> str:
        if not isinstance(name, str):
            return ""
        name_no_comment = name.split('#', 1)[0]
        cleaned_name = name_no_comment.strip()
        if (cleaned_name.startswith('"') and cleaned_name.endswith('"')) or \
           (cleaned_name.startswith("'") and cleaned_name.endswith("'")):
            cleaned_name = cleaned_name[1:-1]
        return cleaned_name.strip()

    async def _generate_with_gemini(self, prompt: str, effective_model_name: str, temperature: Optional[float]) -> str:
        current_generation_config = genai.types.GenerationConfig(
            temperature=temperature if temperature is not None else self.generation_config_gemini.temperature,
            top_p=self.generation_config_gemini.top_p,
            top_k=self.generation_config_gemini.top_k
        )
        if temperature is not None: logger.debug(f"Using custom temperature for Gemini: {temperature}")

        model_to_use = genai.GenerativeModel(
            effective_model_name, 
            generation_config=current_generation_config
        )
        delay = self.retry_delay_seconds
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Gemini API Call Attempt {attempt + 1}/{self.max_retries} to model '{effective_model_name}'.")
                response = await model_to_use.generate_content_async(prompt)
                
                if not response.candidates:
                    logger.warning(f"Gemini API returned no candidates for model '{effective_model_name}'.")
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        logger.error(f"Prompt blocked by Gemini for model '{effective_model_name}'. Reason: {response.prompt_feedback.block_reason}, Feedback: {response.prompt_feedback.safety_ratings}")
                        raise GoogleAPIError(f"Prompt blocked by Gemini API. Reason: {response.prompt_feedback.block_reason}")
                    return "" # No candidates, no block reason, return empty

                generated_text = response.text # More direct way to get text if available
                return generated_text
            except (GeminiInternalServerError, GeminiDeadlineExceeded, GoogleAPIError) as e:
                logger.warning(f"Gemini API error (model '{effective_model_name}', attempt {attempt + 1}): {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60) # Cap delay
                else:
                    logger.error(f"Gemini API call failed after {self.max_retries} retries for model '{effective_model_name}'.")
                    raise
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error during Gemini code generation with model '{effective_model_name}': {e}", exc_info=True)
                raise
        return "" # Should be unreachable if retries fail due to raise

    async def _generate_with_openai_compatible(self, prompt: str, model_to_use: str, temperature: Optional[float]) -> str:
        current_temperature = temperature if temperature is not None else self.default_temperature
        if temperature is not None: logger.debug(f"Using custom temperature for OpenAI/Compatible: {current_temperature}")

        api_payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": current_temperature,
            # "top_p": self.default_top_p, # OpenAI compatible usually uses temperature more
        }
        
        extra_body_payload: Optional[Dict[str, Any]] = None
        if self.llm_provider == "openai_compatible" and self.openai_compatible_num_ctx is not None:
            extra_body_payload = {"options": {"num_ctx": self.openai_compatible_num_ctx}}
            logger.debug(f"Including num_ctx: {self.openai_compatible_num_ctx} in OpenAI-compatible API call via extra_body.")

        delay = self.retry_delay_seconds
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI/Compatible API Call Attempt {attempt + 1}/{self.max_retries} to model '{model_to_use}'. Extra_body: {extra_body_payload is not None}")
                
                if extra_body_payload:
                    response = await self.client.chat.completions.create(**api_payload, extra_body=extra_body_payload)
                else:
                    response = await self.client.chat.completions.create(**api_payload)
                
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    generated_text = response.choices[0].message.content.strip()
                    return generated_text
                else:
                    logger.warning(f"OpenAI/Compatible API (model '{model_to_use}') returned no content in choices.")
                    if response.choices and response.choices[0].finish_reason:
                        logger.warning(f"Finish reason for model '{model_to_use}': {response.choices[0].finish_reason}")
                        if response.choices[0].finish_reason == 'content_filter':
                            raise OpenAIAPIError(f"Content filtered by OpenAI/Compatible API for model '{model_to_use}'.")
                    return "" # No content, return empty
            except BadRequestError as e: 
                logger.error(f"OpenAI/Compatible API BadRequestError (model '{model_to_use}', attempt {attempt + 1}): {e.status_code} - {e.body}")
                error_body_str = str(e.body).lower() if e.body else ""
                # Specific check for num_ctx issues if passed via extra_body
                if "options" in error_body_str and "num_ctx" in error_body_str and ("unexpected" in error_body_str or "unknown field" in error_body_str):
                    logger.error("The server reported an issue with 'options' or 'num_ctx' (e.g., 'unexpected parameter', 'unknown field'). "
                                 "This means your OpenAI-compatible server (e.g., Ollama) might not support setting 'num_ctx' via the 'options' field in the API, or the field name/structure is incorrect for your server version. "
                                 "Refer to your local LLM server's API documentation. For Ollama, context is often set at model creation time (Modelfile) or server run command. This error will not be retried if it's parameter-related.")
                    raise # Do not retry persistent parameter issues
                # General retry for other BadRequestErrors
                if attempt < self.max_retries - 1:
                    logger.warning(f"Retrying BadRequestError in {delay}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    logger.error(f"OpenAI/Compatible API call failed due to BadRequestError after {self.max_retries} retries for model '{model_to_use}'.")
                    raise 
            except (OpenAIAPIError, OpenAITimeoutError, OpenAIAPIConnectionError, OpenAIRateLimitError) as e:
                logger.warning(f"OpenAI/Compatible API error (model '{model_to_use}', attempt {attempt + 1}): {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    logger.error(f"OpenAI/Compatible API call failed after {self.max_retries} retries for model '{model_to_use}'.")
                    raise
            except TypeError as te: 
                logger.error(f"TypeError during OpenAI/Compatible API call for model '{model_to_use}': {te}. This may indicate an issue with parameters.", exc_info=True)
                if "extra_body" in str(te).lower() or ("options" in str(te).lower() and "num_ctx" in str(te).lower()):
                    logger.error("The 'extra_body' or 'options.num_ctx' parameter caused a TypeError. "
                                 "This might mean the OpenAI library version or the server endpoint doesn't support passing it this way for model '{model_to_use}'. "
                                 "Try configuring context size at the server level (e.g., Ollama Modelfile or run command).")
                raise # Do not retry TypeError for parameters
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error during OpenAI/Compatible code generation with model '{model_to_use}': {e}", exc_info=True)
                raise
        return "" # Should be unreachable

    async def generate_code(self, prompt: str, model_name_override: Optional[str] = None, temperature: Optional[float] = None) -> str:
        raw_effective_model_name = model_name_override if model_name_override else self.model_name
        
        if not raw_effective_model_name: # Should have been caught by __init__
            logger.error("CRITICAL: No model name available for code generation.")
            return ""
            
        effective_model_name = self._clean_model_name(raw_effective_model_name)
        if not effective_model_name: 
            logger.error(f"CRITICAL: Model name '{raw_effective_model_name}' became empty after cleaning.")
            return ""

        logger.info(f"Generating code with provider: '{self.llm_provider}', model: '{effective_model_name}'.")
        logger.debug(f"Prompt for model '{effective_model_name}':\n--PROMPT START--\n{prompt[:500]}...\n--PROMPT END--")

        generated_text = ""
        try:
            if self.llm_provider == "gemini":
                generated_text = await self._generate_with_gemini(prompt, effective_model_name, temperature)
            elif self.llm_provider in ["openai", "openai_compatible"]:
                generated_text = await self._generate_with_openai_compatible(prompt, effective_model_name, temperature)
            else: # Should be caught by __init__
                logger.error(f"Unsupported LLM_PROVIDER '{self.llm_provider}' encountered in generate_code.")
                return ""
            
            logger.debug(f"Raw LLM response (model '{effective_model_name}', first 500 chars): {generated_text[:500]}")
            cleaned_code = self._clean_llm_output(generated_text)
            logger.debug(f"Cleaned code (model '{effective_model_name}', first 500 chars): {cleaned_code[:500]}")
            return cleaned_code

        except (GoogleAPIError, OpenAIAPIError, BadRequestError) as e: 
            logger.error(f"API communication failed for model '{effective_model_name}' after retries: {type(e).__name__} - {e}")
            return "" # Return empty string on final API error
        except TypeError as te: # Parameter errors
            logger.error(f"Parameter TypeError during code generation for model '{effective_model_name}': {te}")
            return ""
        except Exception as e: # Other critical errors
            logger.critical(f"Unexpected critical error in generate_code for model '{effective_model_name}': {e}", exc_info=True)
            return ""

    def _clean_llm_output(self, raw_code: str) -> str:
        if not isinstance(raw_code, str): 
            logger.warning(f"LLM output was not a string (type: {type(raw_code)}), cannot clean. Returning empty.")
            return ""

        # Remove <think>...</think> blocks first, case-insensitively
        code_no_think = re.sub(r"<think>.*?</think>", "", raw_code, flags=re.DOTALL | re.IGNORECASE).strip()
        if len(raw_code) != len(code_no_think):
            logger.debug("Removed <think> blocks from LLM output.")
        
        # Common pattern: ```python\ncode\n``` or ```\ncode\n``` or ```code```
        # Regex to capture content within (optional) python markdown block
        # It tries to match the most common "python" block first.
        python_md_match = re.match(r"^\s*```(?:python)?\s*(.*?)\s*```\s*$", code_no_think, re.DOTALL | re.IGNORECASE)
        if python_md_match:
            extracted_code = python_md_match.group(1).strip()
            logger.debug("Extracted code from Python markdown fences.")
            return extracted_code
        
        # If no specific python block, try generic markdown block (e.g. ```text ... ```)
        # This is less common for code but good to have as a fallback.
        generic_md_match = re.match(r"^\s*```(?:[a-zA-Z0-9]*)?\s*(.*?)\s*```\s*$", code_no_think, re.DOTALL)
        if generic_md_match:
            extracted_code = generic_md_match.group(1).strip()
            logger.debug("Extracted code from generic markdown fences.")
            return extracted_code
        
        # If no markdown fences detected, return the (think-block-removed and stripped) code as is.
        # This handles cases where the LLM returns plain code without fences.
        logger.debug("No markdown fences found. Returning code after <think> removal and stripping.")
        return code_no_think # Already stripped

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        Main entry point for the agent to generate code.
        Inherited from BaseAgent -> CodeGeneratorInterface.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Relaying to generate_code with model_override: {model_name}, temp: {temperature}.")
        return await self.generate_code(prompt=prompt, model_name_override=model_name, temperature=temperature)

async def test_generation():
    # --- IMPORTANT ---
    # To test, set your .env file with the appropriate LLM_PROVIDER and API keys/endpoints/model names.
    # Example .env for OpenAI-Compatible (e.g., Ollama with a model like 'llama3' or 'qwen2:7b'):
    #
    # LLM_PROVIDER="openai_compatible"
    # OPENAI_COMPATIBLE_ENDPOINT_URL="http://localhost:11434/v1" # Default for Ollama
    # OPENAI_COMPATIBLE_MODEL_NAME="qwen2:0.5b" # Make sure this model is pulled in Ollama: `ollama pull qwen2:0.5b`
    # OPENAI_COMPATIBLE_API_KEY="ollama" # Or "not-needed"
    # OPENAI_COMPATIBLE_NUM_CTX="2048" # Optional
    #
    # Ensure your local LLM server (e.g., Ollama) is running.
    # `ollama serve`
    # `ollama list` (to see available models)

    # Configure basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Ensure .env is loaded (settings.py should do this, but explicit here for standalone test)
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=True)
            logger.info(f"Test: Loaded .env file from: {dotenv_path}")
        else:
            logger.warning(f"Test: .env file not found at {dotenv_path}. Relying on system env vars.")

        agent = CodeGeneratorAgent() 
        logger.info(f"Test: Initialized CodeGeneratorAgent for provider: '{agent.llm_provider}', model: '{agent.model_name}'")
        if agent.llm_provider == "openai_compatible" and agent.openai_compatible_num_ctx is not None:
            logger.info(f"Test: OpenAI-compatible num_ctx for test: {agent.openai_compatible_num_ctx}")
    except ValueError as e:
        logger.error(f"Test: Failed to initialize CodeGeneratorAgent: {e}")
        logger.error("Test: Please ensure your .env file is correctly configured for the selected LLM_PROVIDER, model name, and any required keys/endpoints.")
        return
    except Exception as e_init:
        logger.error(f"Test: Unexpected error during agent initialization: {e_init}", exc_info=True)
        return

    test_prompt = "Write a very simple Python function that takes two numbers, a and b, and returns their sum. Name the function `add_two_numbers`."
    
    logger.info(f"Test: Sending prompt to {agent.llm_provider} model '{agent.model_name}': '{test_prompt}'")
    generated_code = await agent.generate_code(test_prompt, temperature=0.5) # Use a moderate temperature
    
    print(f"\n--- Generated Code (Provider: {agent.llm_provider}, Model: '{agent.model_name}') ---")
    if generated_code:
        print(generated_code)
    else:
        print("!!! No code was generated. Check logs for errors. !!!")
        logger.warning("Test: Generation returned empty code. Check API key/token, model name, endpoint, server logs (if local), and network connectivity.")
    print("---------------------------------------------------\n")

    # Test with a slightly more complex prompt (e.g., asking for a specific structure)
    test_prompt_diff_like = (
        "Task: Given the Python function below:\n"
        "```python\n"
        "def calculate_area(length, width):\n"
        "    return length * width\n"
        "```\n"
        "Modify this function to also include an optional `height` parameter. If `height` is provided, "
        "the function should return the volume (length * width * height). If `height` is not provided, "
        "it should return the area (length * width) as before. "
        "Provide only the complete modified Python function code for `calculate_area`."
    )
    logger.info(f"Test: Sending diff-like prompt to {agent.llm_provider} model '{agent.model_name}': '{test_prompt_diff_like[:100]}...'")
    generated_diff_code = await agent.generate_code(test_prompt_diff_like, temperature=0.6)
    print(f"--- Generated Diff-Like Code (Provider: {agent.llm_provider}, Model: '{agent.model_name}') ---")
    if generated_diff_code:
        print(generated_diff_code)
    else:
        print("!!! No code was generated for the diff-like prompt. Check logs. !!!")
    print("-------------------------------------------------------------\n")


if __name__ == "__main__":
    # This allows the test_generation function to be run directly using `python code_generator/agent.py`
    asyncio.run(test_generation())