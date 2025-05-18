import google.generativeai as genai
import openai # Added for OpenAI
from typing import Optional, Dict, Any
import logging
import asyncio
from google.api_core.exceptions import InternalServerError as GeminiInternalServerError, GoogleAPIError, DeadlineExceeded as GeminiDeadlineExceeded
from openai import APIError as OpenAIAPIError, APITimeoutError as OpenAITimeoutError, APIConnectionError as OpenAIAPIConnectionError, RateLimitError as OpenAIRateLimitError, BadRequestError # Added for OpenAI errors
import time
import re

from core.interfaces import CodeGeneratorInterface, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_provider = settings.LLM_PROVIDER
        self.model_name: Optional[str] = None
        self.client: Any = None
        self.max_retries = settings.API_MAX_RETRIES
        self.retry_delay_seconds = settings.API_RETRY_DELAY_SECONDS
        self.openai_compatible_num_ctx: Optional[int] = None # For num_ctx parameter

        # Default generation config values (can be overridden per provider)
        self.default_temperature = 0.6
        self.default_top_p = 0.9
        self.default_top_k = 40 # Gemini specific, OpenAI uses top_p

        if self.llm_provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in settings for gemini provider.")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model_name = self._clean_model_name(settings.GEMINI_MODEL_NAME) # Ensure default is cleaned
            self.generation_config_gemini = genai.types.GenerationConfig(
                temperature=self.default_temperature,
                top_p=self.default_top_p,
                top_k=self.default_top_k
            )
            logger.info(f"CodeGeneratorAgent initialized with Gemini provider. Model: {self.model_name}")
        elif self.llm_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in settings for openai provider.")
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model_name = self._clean_model_name(settings.OPENAI_MODEL_NAME) # Ensure default is cleaned
            logger.info(f"CodeGeneratorAgent initialized with OpenAI provider. Model: {self.model_name}")
        elif self.llm_provider == "openai_compatible":
            if not settings.OPENAI_COMPATIBLE_ENDPOINT_URL:
                raise ValueError("OPENAI_COMPATIBLE_ENDPOINT_URL not found for openai_compatible provider.")
            raw_model_name_from_settings = settings.OPENAI_COMPATIBLE_MODEL_NAME
            if not raw_model_name_from_settings:
                raise ValueError("OPENAI_COMPATIBLE_MODEL_NAME not found for openai_compatible provider.")
            # Clean the model name obtained from settings
            self.model_name = self._clean_model_name(raw_model_name_from_settings)
            
            self.client = openai.AsyncOpenAI(
                base_url=settings.OPENAI_COMPATIBLE_ENDPOINT_URL,
                api_key=settings.OPENAI_COMPATIBLE_API_KEY # Often 'not-needed' or a placeholder
            )
            if settings.OPENAI_COMPATIBLE_NUM_CTX:
                try:
                    self.openai_compatible_num_ctx = int(settings.OPENAI_COMPATIBLE_NUM_CTX)
                    logger.info(f"OpenAI-compatible context window (num_ctx) will be set to: {self.openai_compatible_num_ctx} via extra_body options.")
                except ValueError:
                    logger.warning(f"Invalid value for OPENAI_COMPATIBLE_NUM_CTX: '{settings.OPENAI_COMPATIBLE_NUM_CTX}'. Must be an integer. Ignoring.")
            
            logger.info(f"CodeGeneratorAgent initialized with OpenAI-compatible provider. Endpoint: {settings.OPENAI_COMPATIBLE_ENDPOINT_URL}, Cleaned Model: '{self.model_name}' (Raw from settings: '{raw_model_name_from_settings}')")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm_provider}. Choose 'gemini', 'openai', or 'openai_compatible'.")

    def _clean_model_name(self, name: Optional[str]) -> str:
        """Cleans the model name string by stripping whitespace, quotes, and comments."""
        if not isinstance(name, str):
            return ""
        # Remove potential comments if they are part of the string (e.g., "model-name # some comment")
        name_no_comment = name.split('#', 1)[0] # Split only on the first #
        # Strip whitespace
        cleaned_name = name_no_comment.strip()
        # Remove surrounding quotes if they are part of the string itself
        if (cleaned_name.startswith('"') and cleaned_name.endswith('"')) or \
           (cleaned_name.startswith("'") and cleaned_name.endswith("'")):
            cleaned_name = cleaned_name[1:-1]
        return cleaned_name.strip() # Strip again after quote removal


    async def _generate_with_gemini(self, prompt: str, effective_model_name: str, temperature: Optional[float]) -> str:
        current_generation_config = genai.types.GenerationConfig(
            temperature=temperature if temperature is not None else self.generation_config_gemini.temperature,
            top_p=self.generation_config_gemini.top_p,
            top_k=self.generation_config_gemini.top_k
        )
        if temperature is not None:
            logger.debug(f"Using temperature override for Gemini: {temperature}")

        model_to_use = genai.GenerativeModel(
            effective_model_name, 
            generation_config=current_generation_config
        )
        delay = self.retry_delay_seconds
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Gemini API Call Attempt {attempt + 1} of {self.max_retries} to '{effective_model_name}'.")
                response = await model_to_use.generate_content_async(prompt)
                
                if not response.candidates:
                    logger.warning("Gemini API returned no candidates.")
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        logger.error(f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason}")
                        logger.error(f"Prompt feedback details: {response.prompt_feedback.safety_ratings}")
                        raise GoogleAPIError(f"Prompt blocked by Gemini API. Reason: {response.prompt_feedback.block_reason}")
                    return ""

                generated_text = response.candidates[0].content.parts[0].text
                return generated_text
            except (GeminiInternalServerError, GeminiDeadlineExceeded, GoogleAPIError) as e:
                logger.warning(f"Gemini API error on attempt {attempt + 1}: {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Gemini API call failed after {self.max_retries} retries for model '{effective_model_name}'.")
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during Gemini code generation with '{effective_model_name}': {e}", exc_info=True)
                raise
        return "" # Should be unreachable if retries fail due to raise

    async def _generate_with_openai_compatible(self, prompt: str, model_to_use: str, temperature: Optional[float]) -> str:
        current_temperature = temperature if temperature is not None else self.default_temperature
        if temperature is not None:
            logger.debug(f"Using temperature override for OpenAI/Compatible: {current_temperature}")

        logger.debug(f"Attempting OpenAI/Compatible API call with model: '{model_to_use}' (type: {type(model_to_use)}), temp: {current_temperature}")

        api_payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": current_temperature,
        }
        
        extra_body_payload: Optional[Dict[str, Any]] = None
        if self.llm_provider == "openai_compatible" and self.openai_compatible_num_ctx is not None:
            # ### PR REVIEW: Check added for self.llm_provider ensures num_ctx is only applied for compatible, not standard OpenAI.
            if extra_body_payload is None:
                extra_body_payload = {}
            if "options" not in extra_body_payload: # Common pattern for Ollama
                extra_body_payload["options"] = {}
            extra_body_payload["options"]["num_ctx"] = self.openai_compatible_num_ctx
            logger.debug(f"Including num_ctx: {self.openai_compatible_num_ctx} in OpenAI-compatible API call via extra_body.")

        delay = self.retry_delay_seconds
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI/Compatible API Call Attempt {attempt + 1} of {self.max_retries} to model '{model_to_use}'. Payload keys: {list(api_payload.keys())}, Extra_body: {extra_body_payload is not None}")
                
                if extra_body_payload:
                    response = await self.client.chat.completions.create(**api_payload, extra_body=extra_body_payload)
                else:
                    response = await self.client.chat.completions.create(**api_payload)
                
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    generated_text = response.choices[0].message.content.strip()
                    return generated_text
                else:
                    logger.warning("OpenAI/Compatible API returned no content in choices.")
                    if response.choices and response.choices[0].finish_reason:
                        logger.warning(f"Finish reason: {response.choices[0].finish_reason}")
                        if response.choices[0].finish_reason == 'content_filter':
                            raise OpenAIAPIError("Content filtered by OpenAI/Compatible API.") # Re-raise as APIError to allow retry if appropriate
                    return ""
            except BadRequestError as e: 
                logger.error(f"OpenAI/Compatible API BadRequestError on attempt {attempt + 1} for model '{model_to_use}': {e.status_code} - {e.body}")
                error_body_str = str(e.body).lower() if e.body else ""
                if "unexpected" in error_body_str and "num_ctx" in error_body_str:
                    logger.error("The server reported 'num_ctx' as an unexpected parameter even with extra_body. "
                                "Please ensure your OpenAI-compatible server (e.g., Ollama) supports setting 'num_ctx' via the 'options' field in the API request, "
                                "or configure context size at the server/model level directly. This error will not be retried if it seems parameter-related.")
                    raise # Do not retry if it's a persistent parameter issue with num_ctx
                if attempt < self.max_retries -1:
                    logger.warning(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *=2
                else:
                    logger.error(f"OpenAI/Compatible API call failed due to BadRequestError after {self.max_retries} retries for model '{model_to_use}'.")
                    raise 
            except (OpenAIAPIError, OpenAITimeoutError, OpenAIAPIConnectionError, OpenAIRateLimitError) as e:
                logger.warning(f"OpenAI/Compatible API error on attempt {attempt + 1} for model '{model_to_use}': {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"OpenAI/Compatible API call failed after {self.max_retries} retries for model '{model_to_use}'.")
                    raise
            except TypeError as te: 
                logger.error(f"TypeError during OpenAI/Compatible API call for model '{model_to_use}': {te}. This usually means an unsupported parameter was passed.", exc_info=True)
                if "extra_body" in str(te).lower() or "num_ctx" in str(te).lower() : # Check if TypeError is related to extra_body or num_ctx
                    logger.error("The 'extra_body' or 'num_ctx' parameter caused a TypeError. This might indicate the OpenAI library version or the server endpoint does not support passing it this way. "
                                "Try configuring context size at the server level (e.g., Ollama Modelfile or run command).")
                # Do not retry on TypeError for parameter issues as it's likely persistent
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during OpenAI/Compatible code generation with model '{model_to_use}': {e}", exc_info=True)
                raise # Re-raise to be caught by the calling function
        return "" # Should be unreachable

    async def generate_code(self, prompt: str, model_name_override: Optional[str] = None, temperature: Optional[float] = None) -> str:
        raw_effective_model_name = model_name_override if model_name_override else self.model_name
        
        if not raw_effective_model_name:
            logger.error("No model name configured or provided for code generation.")
            return ""
            
        effective_model_name = self._clean_model_name(raw_effective_model_name)
        if not effective_model_name: 
            logger.error(f"Model name '{raw_effective_model_name}' became empty after cleaning.")
            return ""

        logger.info(f"Attempting to generate code using provider: {self.llm_provider}, effective model: '{effective_model_name}' (Raw: '{raw_effective_model_name}')")
        logger.debug(f"Received prompt for code generation:\n--PROMPT START--\n{prompt}\n--PROMPT END--")

        generated_text = ""
        try:
            if self.llm_provider == "gemini":
                generated_text = await self._generate_with_gemini(prompt, effective_model_name, temperature)
            elif self.llm_provider == "openai" or self.llm_provider == "openai_compatible":
                generated_text = await self._generate_with_openai_compatible(prompt, effective_model_name, temperature)
            else:
                logger.error(f"generate_code called with unsupported LLM_PROVIDER: {self.llm_provider}")
                return ""
            
            logger.debug(f"Raw response from {self.llm_provider} API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
            cleaned_code = self._clean_llm_output(generated_text)
            logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
            return cleaned_code

        except (GoogleAPIError, OpenAIAPIError, BadRequestError) as e: 
            logger.error(f"Final API error after retries for {self.llm_provider} model '{effective_model_name}': {type(e).__name__} - {e}")
            return "" 
        except TypeError as te: 
            logger.error(f"Parameter TypeError during code generation for {self.llm_provider} model '{effective_model_name}': {te}")
            return ""
        except Exception as e:
            logger.error(f"An critical unexpected error occurred during code generation orchestration with {self.llm_provider} model '{effective_model_name}': {e}", exc_info=True)
            return ""

    def _clean_llm_output(self, raw_code: str) -> str:
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        if not isinstance(raw_code, str): 
            logger.warning(f"Raw code is not a string (type: {type(raw_code)}), returning empty.")
            return ""

        cleaned_code = re.sub(r"<think>.*?</think>", "", raw_code, flags=re.DOTALL | re.IGNORECASE).strip() # Added IGNORECASE for <think>
        if len(raw_code) != len(cleaned_code): # Check if something was actually removed
            logger.debug(f"Removed <think> blocks. Length after think removal: {len(cleaned_code)}")
        else:
            logger.debug("No <think> blocks found to remove.")
        
        stripped_code = cleaned_code.strip() 
        
        # Common pattern: ```python\ncode\n``` or ```python code```
        # Regex to capture content within python markdown block
        python_md_match = re.match(r"^\s*```python\s*(.*?)\s*```\s*$", stripped_code, re.DOTALL | re.IGNORECASE)
        if python_md_match:
            extracted_code = python_md_match.group(1).strip()
            logger.debug("Cleaned Python specific markdown fences using regex.")
            return extracted_code
        
        # Regex to match general markdown code blocks, possibly with language hints or none
        # Matches ```<optional_lang>\n ... code ... \n``` or ```\n ... code ... \n``` or ```code```
        # It's non-greedy (.*?)
        generic_md_match = re.match(r"^\s*```(?:[a-zA-Z0-9]*)?\s*(.*?)\s*```\s*$", stripped_code, re.DOTALL)
        if generic_md_match:
            extracted_code = generic_md_match.group(1).strip()
            logger.debug(f"Cleaned generic markdown fences using regex. Extracted: '{extracted_code[:100]}...'")
            return extracted_code
        
        logger.debug("No markdown fences found or standard cleaning applied. Returning as is (after think removal and stripping).")
        return stripped_code


    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> str:
        logger.debug(f"CodeGeneratorAgent.execute called. Relaying to generate_code.")
        return await self.generate_code(prompt=prompt, model_name_override=model_name, temperature=temperature)

async def test_generation(): # ### PR REVIEW: This is an async function
    # --- IMPORTANT ---
    # To test, set your .env file with the appropriate LLM_PROVIDER and API keys/endpoints.
    # Ensure the model name in .env is exactly as your provider expects it, e.g.:
    #
    # For Gemini:
    # LLM_PROVIDER="gemini"
    # GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    # GEMINI_MODEL_NAME="gemini-1.5-flash-latest"
    #
    # For OpenAI:
    # LLM_PROVIDER="openai"
    # OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    # OPENAI_MODEL_NAME="gpt-3.5-turbo"
    #
    # For OpenAI-Compatible (e.g., Ollama with Llama3):
    # LLM_PROVIDER="openai_compatible"
    # OPENAI_COMPATIBLE_ENDPOINT_URL="http://localhost:11434/v1"
    # OPENAI_COMPATIBLE_MODEL_NAME="llama3" # Or "qwen2:latest", "codellama:7b-instruct" etc.
    # OPENAI_COMPATIBLE_API_KEY="ollama" # Or "not-needed"
    # OPENAI_COMPATIBLE_NUM_CTX="2048" # Optional: set context window size

    try:
        agent = CodeGeneratorAgent() 
        logger.info(f"Testing with provider: {agent.llm_provider}, effective model for test: '{agent.model_name}'")
        if agent.llm_provider == "openai_compatible" and agent.openai_compatible_num_ctx is not None:
            logger.info(f"OpenAI-compatible num_ctx for test: {agent.openai_compatible_num_ctx}")
    except ValueError as e:
        logger.error(f"Failed to initialize CodeGeneratorAgent: {e}")
        logger.error("Please ensure your .env file is correctly configured for the selected LLM_PROVIDER and model name.")
        return

    test_prompt = "Write a Python function that takes two numbers and returns their sum. Name the function `add_numbers`."
    
    generated_code = await agent.generate_code(test_prompt, temperature=0.6)
    print(f"--- Generated Code (Provider: {agent.llm_provider}, Model: '{agent.model_name}') ---")
    print(generated_code)
    print("----------------------")

    if not generated_code:
        logger.warning("Test generation returned empty code. Check API key, model name, endpoint, and server logs if applicable.")
        return

    test_prompt_diff = ("""
Given the function below:
```python
def add(a, b):
    return a + b
```
Suggest a modification to handle an optional third argument c which defaults to 0 and is added to the sum. Provide only the complete modified function code.
""")
    generated_diff_code = await agent.generate_code(test_prompt_diff, temperature=0.5)
    print(f"--- Generated Diff Code (Provider: {agent.llm_provider}) ---")
    print(generated_diff_code)
    print("----------------------")
    asyncio.run(test_generation())