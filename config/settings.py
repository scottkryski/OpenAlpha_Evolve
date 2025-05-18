# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# --- LLM Provider Configuration ---
# Choose your LLM provider: "gemini", "openai", or "openai_compatible"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# --- Gemini Configuration (if LLM_PROVIDER is "gemini") ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # "gemini-pro" is also an option

# --- OpenAI Configuration (if LLM_PROVIDER is "openai") ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") # Or "gpt-4", "gpt-4-turbo-preview", etc.

# --- OpenAI-Compatible Endpoint Configuration (if LLM_PROVIDER is "openai_compatible") ---
OPENAI_COMPATIBLE_ENDPOINT_URL = os.getenv("OPENAI_COMPATIBLE_ENDPOINT_URL") # e.g., "http://localhost:11434/v1" for Ollama
OPENAI_COMPATIBLE_MODEL_NAME = os.getenv("OPENAI_COMPATIBLE_MODEL_NAME") # e.g., "llama3", "codellama"
# OPENAI_COMPATIBLE_API_KEY is often optional for local endpoints, but can be set if required.
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "not-needed")
# Context window size (num_ctx) for OpenAI-compatible models, if supported by the server.
OPENAI_COMPATIBLE_NUM_CTX = os.getenv("OPENAI_COMPATIBLE_NUM_CTX")


# LLM Model Configuration (Legacy - specific Gemini models, can be deprecated or used as fallbacks if needed)
GEMINI_PRO_MODEL_NAME_LEGACY = "gemini-1.5-pro-latest"
GEMINI_FLASH_MODEL_NAME_LEGACY = "gemini-1.5-flash-latest"


# Evolutionary Parameters (examples)
POPULATION_SIZE = 10
GENERATIONS = 10
ELITISM_COUNT = 2
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.2

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = 800

# Database settings
DATABASE_TYPE = "in_memory"
DATABASE_PATH = "program_database.json"

# Logging Parameters
LOG_LEVEL = "INFO"
LOG_FILE = "alpha_evolve.log"

# API Retry Parameters
API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10

# Placeholder for RL Fine-Tuning
RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

# Monitoring
MONITORING_DASHBOARD_URL = "http://localhost:8080"

# --- Helper function to get a specific setting ---
def get_setting(key, default=None):
    return globals().get(key, default)

# Example of how to get a model, perhaps with fallback logic
def get_active_llm_model_name():
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL_NAME
    elif LLM_PROVIDER == "openai":
        return OPENAI_MODEL_NAME
    elif LLM_PROVIDER == "openai_compatible":
        return OPENAI_COMPATIBLE_MODEL_NAME
    return "default_model_not_configured"

# --- Validate essential configurations based on LLM_PROVIDER ---
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
            int(OPENAI_COMPATIBLE_NUM_CTX)
        except ValueError:
            print(f"Warning: OPENAI_COMPATIBLE_NUM_CTX ('{OPENAI_COMPATIBLE_NUM_CTX}') is not a valid integer. It might be ignored or cause errors.")
elif LLM_PROVIDER not in ["gemini", "openai", "openai_compatible"]:
    print(f"Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Expected 'gemini', 'openai', or 'openai_compatible'. Defaulting to 'gemini' behavior might occur if not handled by agents.")