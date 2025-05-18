# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True) # override=True ensures .env values take precedence over system env vars

# --- LLM Provider Configuration ---
# Choose your LLM provider: "gemini", "openai", or "openai_compatible"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# --- Gemini Configuration (if LLM_PROVIDER is "gemini") ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

# --- OpenAI Configuration (if LLM_PROVIDER is "openai") ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# --- OpenAI-Compatible Endpoint Configuration (if LLM_PROVIDER is "openai_compatible") ---
OPENAI_COMPATIBLE_ENDPOINT_URL = os.getenv("OPENAI_COMPATIBLE_ENDPOINT_URL") # e.g., "http://localhost:11434/v1" for Ollama
OPENAI_COMPATIBLE_MODEL_NAME = os.getenv("OPENAI_COMPATIBLE_MODEL_NAME") # e.g., "llama3", "codellama:7b-instruct"
# OPENAI_COMPATIBLE_API_KEY is often optional for local endpoints, but can be set if required.
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed")
# Context window size (num_ctx) for OpenAI-compatible models, if supported by the server (e.g., Ollama).
OPENAI_COMPATIBLE_NUM_CTX = os.getenv("OPENAI_COMPATIBLE_NUM_CTX")


# LLM Model Configuration (Legacy - specific Gemini models, can be deprecated or used as fallbacks if needed)
GEMINI_PRO_MODEL_NAME_LEGACY = "gemini-1.5-pro-latest" ### PR REVIEW: Consider removing "LEGACY" if these are just alternative model choices.
GEMINI_FLASH_MODEL_NAME_LEGACY = "gemini-1.5-flash-latest"


# Evolutionary Parameters (examples)
POPULATION_SIZE = 10
GENERATIONS = 10
ELITISM_COUNT = 2
MUTATION_RATE = 0.7 # ### PR REVIEW: Note: MUTATION_RATE and CROSSOVER_RATE are defined but not directly used by LLM mutation strategy. They might be for future traditional EA operators.
CROSSOVER_RATE = 0.2

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = 800

# Database settings
DATABASE_TYPE = "in_memory" # Options: "in_memory", "json_file", "postgresql", "mongodb" (latter require more setup)
DATABASE_PATH = "program_database.json" # Used if DATABASE_TYPE is "json_file"

# Logging Parameters
LOG_LEVEL = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE = "alpha_evolve.log" # Set to None to disable file logging

# API Retry Parameters
API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10 # Initial delay, typically doubles with each retry

# Placeholder for RL Fine-Tuning
RL_TRAINING_INTERVAL_GENERATIONS = 50 # How often to trigger RL fine-tuning (if enabled)
RL_MODEL_PATH = "rl_finetuner_model.pth" # Path to save/load RL model

# Monitoring
MONITORING_DASHBOARD_URL = "http://localhost:8080" # Example URL for a monitoring dashboard

# --- Helper function to get a specific setting ---
def get_setting(key, default=None):
    """
    Retrieves a setting value from this module's global scope.
    """
    return globals().get(key, default)

# Example of how to get a model, perhaps with fallback logic
def get_active_llm_model_name() -> str:
    """
    Gets the configured model name based on the active LLM_PROVIDER.
    Returns a default string if the provider or model is not properly configured.
    """
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL_NAME if GEMINI_MODEL_NAME else "gemini_model_not_set"
    elif LLM_PROVIDER == "openai":
        return OPENAI_MODEL_NAME if OPENAI_MODEL_NAME else "openai_model_not_set"
    elif LLM_PROVIDER == "openai_compatible":
        return OPENAI_COMPATIBLE_MODEL_NAME if OPENAI_COMPATIBLE_MODEL_NAME else "openai_compatible_model_not_set"
    return "default_model_not_configured"

# --- Validate essential configurations based on LLM_PROVIDER ---
# ### PR REVIEW: Using print for these early warnings is okay as logging might not be configured yet.
# ### However, for a library, it might be better to raise specific ConfigurationError exceptions
# ### or let the agents raise ValueErrors when they can't find their required settings.
# ### For an application, print warnings are acceptable.
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    print("Warning: LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. Functionality will be limited.")
elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    print("Warning: LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. Functionality will be limited.")
elif LLM_PROVIDER == "openai_compatible":
    if not OPENAI_COMPATIBLE_ENDPOINT_URL:
        print("Warning: LLM_PROVIDER is 'openai_compatible' but OPENAI_COMPATIBLE_ENDPOINT_URL is not set. Functionality will be limited.")
    if not OPENAI_COMPATIBLE_MODEL_NAME:
        print("Warning: LLM_PROVIDER is 'openai_compatible' but OPENAI_COMPATIBLE_MODEL_NAME is not set. Functionality will be limited.")
    if OPENAI_COMPATIBLE_NUM_CTX:
        try:
            num_ctx_val = int(OPENAI_COMPATIBLE_NUM_CTX)
            if num_ctx_val <= 0:
                print(f"Warning: OPENAI_COMPATIBLE_NUM_CTX ('{OPENAI_COMPATIBLE_NUM_CTX}') must be a positive integer. It might be ignored or cause errors.")
        except ValueError:
            print(f"Warning: OPENAI_COMPATIBLE_NUM_CTX ('{OPENAI_COMPATIBLE_NUM_CTX}') is not a valid integer. It might be ignored or cause errors.")
elif LLM_PROVIDER not in ["gemini", "openai", "openai_compatible"]:
    # Defaulting behavior is usually handled by agents; this is just a startup warning.
    print(f"Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Expected 'gemini', 'openai', or 'openai_compatible'. Application may not function correctly.")