import os
from dotenv import load_dotenv

# Load environment variables from .env file
# override=True ensures .env values take precedence over system env vars if both exist.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- LLM Provider Configuration ---
# Choose your LLM provider: "gemini", "openai", or "openai_compatible"
# Defaults to "gemini" if not set in .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# --- Gemini Configuration (if LLM_PROVIDER is "gemini") ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest").strip()
# Fallback Gemini models (less commonly used, but available if specific logic needs them)
GEMINI_MODEL_NAME_PRO_FALLBACK = os.getenv("GEMINI_MODEL_NAME_PRO_FALLBACK", "gemini-1.5-pro-latest").strip()
GEMINI_MODEL_NAME_FLASH_FALLBACK = os.getenv("GEMINI_MODEL_NAME_FLASH_FALLBACK", "gemini-1.5-flash-latest").strip()


# --- OpenAI Configuration (if LLM_PROVIDER is "openai") ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o").strip()

# --- OpenAI-Compatible Endpoint Configuration (if LLM_PROVIDER is "openai_compatible") ---
OPENAI_COMPATIBLE_ENDPOINT_URL = os.getenv("OPENAI_COMPATIBLE_ENDPOINT_URL")
OPENAI_COMPATIBLE_MODEL_NAME = os.getenv("OPENAI_COMPATIBLE_MODEL_NAME")
# OPENAI_COMPATIBLE_API_KEY is often optional for local endpoints (e.g., "ollama", "not-needed").
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed").strip()
# Context window size (num_ctx) for OpenAI-compatible models, if supported by the server (e.g., Ollama).
OPENAI_COMPATIBLE_NUM_CTX = os.getenv("OPENAI_COMPATIBLE_NUM_CTX")
if OPENAI_COMPATIBLE_NUM_CTX:
    OPENAI_COMPATIBLE_NUM_CTX = OPENAI_COMPATIBLE_NUM_CTX.strip()


# Evolutionary Parameters
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", 10))
GENERATIONS = int(os.getenv("GENERATIONS", 10))
ELITISM_COUNT = int(os.getenv("ELITISM_COUNT", 2)) # Number of best individuals to carry over directly
# MUTATION_RATE and CROSSOVER_RATE are placeholders for traditional EA operators,
# not directly used by the current LLM-driven mutation strategy.
MUTATION_RATE = float(os.getenv("MUTATION_RATE", 0.7))
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", 0.2))

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_TIMEOUT_SECONDS", 60)) # Increased default

# Database settings
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "in_memory").strip().lower()
DATABASE_PATH = os.getenv("DATABASE_PATH", "program_database.json").strip() # Used if DATABASE_TYPE is "json_file"

# Logging Parameters
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
LOG_FILE = os.getenv("LOG_FILE", "alpha_evolve.log").strip() # Set to "None" or "" to disable file logging

# API Retry Parameters
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", 5))
API_RETRY_DELAY_SECONDS = int(os.getenv("API_RETRY_DELAY_SECONDS", 10)) # Initial delay

# Placeholder for RL Fine-Tuning (Not actively used yet)
RL_TRAINING_INTERVAL_GENERATIONS = int(os.getenv("RL_TRAINING_INTERVAL_GENERATIONS", 50))
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "rl_finetuner_model.pth").strip()

# Monitoring (Not actively used yet)
MONITORING_DASHBOARD_URL = os.getenv("MONITORING_DASHBOARD_URL", "http://localhost:8080").strip()


# --- Helper function to get a specific setting (less common to use directly) ---
def get_setting(key, default=None):
    """
    Retrieves a setting value from this module's global scope.
    """
    return globals().get(key, default)

# --- Function to get the active LLM model name based on provider ---
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
    return "unknown_llm_provider_default_model"

# --- Validate essential configurations based on LLM_PROVIDER ---
# These warnings are printed at startup if configurations seem missing.
# Agents themselves will raise ValueErrors if critical settings are absent during initialization.
_startup_warnings = []
if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY: # Basic check
        _startup_warnings.append("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY seems to be missing or a placeholder.")
    if not GEMINI_MODEL_NAME:
        _startup_warnings.append("LLM_PROVIDER is 'gemini' but GEMINI_MODEL_NAME is not set.")
elif LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY" in OPENAI_API_KEY: # Basic check
        _startup_warnings.append("LLM_PROVIDER is 'openai' but OPENAI_API_KEY seems to be missing or a placeholder.")
    if not OPENAI_MODEL_NAME:
        _startup_warnings.append("LLM_PROVIDER is 'openai' but OPENAI_MODEL_NAME is not set.")
elif LLM_PROVIDER == "openai_compatible":
    if not OPENAI_COMPATIBLE_ENDPOINT_URL:
        _startup_warnings.append("LLM_PROVIDER is 'openai_compatible' but OPENAI_COMPATIBLE_ENDPOINT_URL is not set.")
    if not OPENAI_COMPATIBLE_MODEL_NAME:
        _startup_warnings.append("LLM_PROVIDER is 'openai_compatible' but OPENAI_COMPATIBLE_MODEL_NAME is not set.")
    if OPENAI_COMPATIBLE_NUM_CTX:
        try:
            num_ctx_val = int(OPENAI_COMPATIBLE_NUM_CTX)
            if num_ctx_val <= 0:
                _startup_warnings.append(f"OPENAI_COMPATIBLE_NUM_CTX ('{OPENAI_COMPATIBLE_NUM_CTX}') must be a positive integer.")
        except ValueError:
            _startup_warnings.append(f"OPENAI_COMPATIBLE_NUM_CTX ('{OPENAI_COMPATIBLE_NUM_CTX}') is not a valid integer.")
elif not LLM_PROVIDER:
    _startup_warnings.append("LLM_PROVIDER is not set in the environment. Please configure it in your .env file.")
else: # LLM_PROVIDER is set but not to one of the known values
    _startup_warnings.append(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Expected 'gemini', 'openai', or 'openai_compatible'.")

if _startup_warnings:
    print("--- Configuration Warnings ---", flush=True)
    for warning in _startup_warnings:
        print(f"Warning: {warning}", flush=True)
    print("----------------------------", flush=True)

# Ensure API_KEY variables are not accidentally logged if they are placeholders
_log_safe_api_key_openai = OPENAI_API_KEY if OPENAI_API_KEY and "YOUR_OPENAI_API_KEY" not in OPENAI_API_KEY else "OpenAI API Key (placeholder or not set)"
_log_safe_api_key_gemini = GEMINI_API_KEY if GEMINI_API_KEY and "YOUR_GEMINI_API_KEY" not in GEMINI_API_KEY else "Gemini API Key (placeholder or not set)"
# For OpenAI compatible, the key is often "ollama" or "not-needed", so it's less sensitive but good practice.
_log_safe_api_key_compatible = OPENAI_COMPATIBLE_API_KEY if OPENAI_COMPATIBLE_API_KEY and "YOUR_" not in OPENAI_COMPATIBLE_API_KEY.upper() else "OpenAI Compatible API Key (placeholder, not needed, or set)"

# print(f"Settings Loaded: LLM_PROVIDER='{LLM_PROVIDER}', ActiveModel='{get_active_llm_model_name()}'", flush=True)
# Avoid printing actual keys, even "not-needed", during initial load.
# The CodeGeneratorAgent will log its specific configuration on init.