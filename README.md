# OpenAlpha_Evolve: Regenerating Autonomous Algorithmic Discovery 🚀 
### PR REVIEW: Corrected typo "OpenAplha_Evolve" to "OpenAlpha_Evolve"

![openalpha_evolve_workflow](https://github.com/user-attachments/assets/9d4709ad-0072-44ae-bbb5-7eea1c5fa08c)
### PR REVIEW: Ensure this image URL is stable or embed the image directly in the repository (e.g., in an `assets` folder) and use a relative link.

OpenAlpha_Evolve is an open-source Python framework inspired by the groundbreaking research on autonomous coding agents like DeepMind's AlphaDev (often conceptualized as AlphaEvolve-like systems). It's a **regeneration** of the core idea: an intelligent system that iteratively writes, tests, and improves code using Large Language Models (LLMs) like Google's Gemini, OpenAI's GPT models, or local OpenAI-API compatible models (e.g., via Ollama), guided by the principles of evolution.

Our mission is to provide an accessible, understandable, and extensible platform for researchers, developers, and enthusiasts to explore the fascinating intersection of AI, code generation, and automated problem-solving.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  The Vision: AI-Driven Algorithmic Innovation

Imagine an agent that can:

*   Understand a complex problem description.
*   Generate initial algorithmic solutions.
*   Rigorously test its own code.
*   Learn from failures and successes.
*   Evolve increasingly sophisticated and efficient algorithms over time.

OpenAlpha_Evolve is a step towards this vision. It's not just about generating code; it's about creating a system that *discovers* and *refines* solutions autonomously.

---

##  How It Works: The Evolutionary Cycle

OpenAlpha_Evolve employs a modular, agent-based architecture to orchestrate an evolutionary process:

1.  **Task Definition**: You, the user, define the algorithmic "quest" – the problem to be solved, including examples of inputs and expected outputs, and allowed library imports.
2.  **Prompt Engineering (`PromptDesignerAgent`)**: This agent crafts intelligent prompts for the LLM. It designs:
    *   *Initial Prompts*: To generate the first set of candidate solutions.
    *   *Mutation Prompts*: To introduce variations and improvements to existing solutions.
    *   *Bug-Fix Prompts*: To guide the LLM in correcting errors from previous attempts.
3.  **Code Generation (`CodeGeneratorAgent`)**: Powered by an LLM (supporting Gemini, OpenAI, and OpenAI-compatible local models), this agent takes the prompts and generates Python code.
4.  **Evaluation (`EvaluatorAgent`)**: The generated code is put to the test!
    *   *Syntax Check*: Is the code valid Python?
    *   *Execution*: The code is run in a sandboxed environment against the input/output examples defined in the task.
    *   *Fitness Scoring*: Programs are scored based on correctness, efficiency (runtime), and other potential metrics.
5.  **Database (`DatabaseAgent`)**: All programs (code, fitness, generation, lineage) are stored, creating a record of the evolutionary history. (Currently In-Memory).
6.  **Selection (`SelectionControllerAgent`)**: The "survival of the fittest" principle in action. This agent selects:
    *   *Parents*: Promising programs from the current generation to produce offspring.
    *   *Survivors*: The best programs from both the current population and new offspring to advance to the next generation.
7.  **Iteration**: This cycle repeats for a defined number of generations, with each new generation aiming to produce better solutions than the last.
8.  **Orchestration (`TaskManagerAgent`)**: The maestro of the operation, coordinating all other agents and managing the overall evolutionary loop.

---

##  Key Features

*   **Flexible LLM Backends**:
    *   Google Gemini API.
    *   OpenAI API (GPT-3.5, GPT-4, etc.).
    *   OpenAI-Compatible Endpoints: Supports local LLMs served via tools like Ollama, LiteLLM, vLLM, etc., using the standard OpenAI API format.
*   **Evolutionary Algorithm Core**: Implements iterative improvement through selection, mutation (via prompting), and survival.
*   **Modular Agent Architecture**: Easily extend or replace individual components.
*   **Automated Program Evaluation**: Syntax checking and functional testing against user-provided examples.
*   **Configuration Management**: Easily tweak parameters like population size, number of generations, and LLM settings via a `.env` file and `config/settings.py`.
*   **Detailed Logging**: Comprehensive logs provide insights into each step.
*   **Open Source & Extensible**: Built with Python, designed for experimentation.

---

##  Project Structure\
OpenAlpha_Evolve/
├── code_generator/ # Agent for LLM-based code generation
├── database_agent/ # Agent for storing/retrieving programs
├── evaluator_agent/ # Agent for evaluating generated code
├── prompt_designer/ # Agent for designing LLM prompts
├── selection_controller/ # Agent for evolutionary selection
├── task_manager/ # Agent orchestrating the main loop
├── rl_finetuner/ # (Placeholder) For RL-based prompt/model tuning
├── monitoring_agent/ # (Placeholder) For monitoring metrics
├── config/ # Configuration files (settings.py)
├── core/ # Core interfaces, data models (Program, TaskDefinition)
├── utils/ # Utility functions (currently minimal)
├── tests/ # Unit and integration tests (to be expanded)
├── scripts/ # Helper scripts (currently minimal)
├── main.py # Main entry point to run the system
├── requirements.txt # Project dependencies
├── .env.example # Example for environment variables (copy to .env)
└── README.md # This file!

## 🏁 Getting Started

1.  **Prerequisites**:
    *   Python 3.10+
    *   `pip` for package management
    *   `git` for cloning

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shyamsaktawat/OpenAlpha_Evolve.git # ### PR REVIEW: Or the original repo URL if this is a fork
    cd OpenAlpha_Evolve 
    ```

3.  **Set Up a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Environment Variables**:
    *   Copy `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` and configure it for your chosen LLM provider. **Only configure one provider section.**

        **For Google Gemini:**
        ```dotenv
        LLM_PROVIDER="gemini"
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        GEMINI_MODEL_NAME="gemini-1.5-flash-latest" # Or other compatible Gemini models
        ```
        *Obtain your API key from Google AI Studio.*

        **For OpenAI API:**
        ```dotenv
        LLM_PROVIDER="openai"
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # sk-....
        OPENAI_MODEL_NAME="gpt-3.5-turbo" # Or "gpt-4", "gpt-4-turbo", etc.
        ```

        **For OpenAI-Compatible Local LLM (e.g., Ollama):**
        ```dotenv
        LLM_PROVIDER="openai_compatible"
        # Example for Ollama running locally:
        OPENAI_COMPATIBLE_ENDPOINT_URL="http://localhost:11434/v1" 
        OPENAI_COMPATIBLE_MODEL_NAME="llama3" # The model name as served by Ollama (e.g., "llama3", "codellama:7b-instruct")
        OPENAI_COMPATIBLE_API_KEY="ollama" # Often "not-needed", "ollama", or any string if not required by your endpoint
        OPENAI_COMPATIBLE_NUM_CTX="4096" # Optional: context window size if your server supports it (e.g., for Ollama)
        ```
        *Ensure your local LLM server (like Ollama) is running and the model is downloaded/available.*

6.  **Run OpenAlpha_Evolve!**
    The `main.py` file is configured with an example task (Dijkstra's algorithm). To run it:
    ```bash
    python main.py 
    ```
    Watch the logs to see the evolutionary process unfold! The default logging level is INFO. For more details, you can change `LOG_LEVEL` in `config/settings.py` to `DEBUG`.

---

## 💡 Defining Your Own Algorithmic Quests!

Want to challenge OpenAlpha_Evolve with a new problem?

1.  **Open `main.py`** (or your own script that uses `TaskManagerAgent`).
2.  **Create or Modify a `TaskDefinition` object**:
    *   `id`: A unique string identifier for your task.
    *   `description`: A clear, detailed natural language description of the problem. This is crucial for the LLM.
    *   `function_name_to_evolve`: The name of the Python function the agent should create/evolve.
    *   `input_output_examples`: A list of dictionaries. Each dictionary must have:
        *   `"input"`: The input(s) for the function. This can be a single value, a list (for multiple positional arguments), or a dictionary (for named arguments).
        *   `"output"`: The corresponding expected output.
        *   Example: `{"input": {"numbers": [1, 2, 3]}, "output": 6}` or `{"input": [[1,2], 3], "output": 6}`
    *   `allowed_imports`: A list of Python standard library modules/submodules the generated code is allowed to import (e.g., `["heapq", "math", "sys", "collections.defaultdict"]`).
    *   `evaluation_criteria` (Optional): A string or dictionary describing how success is measured (e.g., "Prioritize correctness, then minimize runtime.").
    *   `initial_code_prompt` (Optional): A specific initial prompt for the LLM if the default isn't suitable.
    *   `hints` (Conceptual - for future enhancement): You might extend `TaskDefinition` to include structured hints for different stages (initial prompt, error fixing) to make `PromptDesignerAgent` more adaptable.

3.  **Instantiate `TaskManagerAgent` with your `TaskDefinition`**.
4.  **Run the agent's `execute()` method.**

The quality of your `description` and `input_output_examples` significantly impacts the agent's success!

---

##  The Horizon: Future Evolution

OpenAlpha_Evolve is a living project. Here are some directions we're excited to explore:

*   **Advanced Evaluation Sandboxing**: More robust and secure sandboxing (e.g., using Docker, E2B, or Firejail) for code execution.
*   **Sophisticated Fitness Metrics**: Beyond correctness/runtime, including complexity, style, resource usage.
*   **Reinforcement Learning for Prompt Strategy**: Implementing `RLFineTunerAgent` to optimize prompt engineering.
*   **Enhanced Monitoring & Visualization**: Tools to visualize evolution, track fitness, via `MonitoringAgent`.
*   **Wider LLM Support & Fine-tuning**: Easier integrations and support for fine-tuning LLMs on successful programs.
*   **Self-Correction & Reflection**: Deeper analysis of failures to refine problem-solving.
*   **Diverse Task Domains**: Applying to more problem types.
*   **Task-Specific Hinting System**: A more robust way to inject task-specific advice into prompts.
*   **Community-Driven Task Library**: A collection of challenging tasks.

---

##  Join the Evolution: Contributing

This is an open invitation to collaborate!

*   **Report Bugs**: Find an issue? Let us know by opening a GitHub Issue.
*   **Suggest Features**: Have an idea? Open a GitHub Issue with a feature request.
*   **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch for your feature or bugfix.
    *   Write clean, well-documented code.
    *   Add tests for your changes if possible.
    *   Ensure your changes don't break existing functionality.
    *   Submit a pull request with a clear description of your changes!

Let's evolve this agent together!

---

##  License

This project is licensed under the **MIT License**. (You'll need to create a `LICENSE` file with the MIT license text if one doesn't exist).

---

##  Homage

OpenAlpha_Evolve is proudly inspired by the pioneering work of Google DeepMind (e.g., AlphaDev) and other related research in LLM-driven code generation and automated discovery. This project aims to make the core concepts more accessible for broader experimentation and learning.

---

*Disclaimer: This is an experimental project. Generated code may not always be optimal, correct, or secure. Always review and test code thoroughly, especially before using it in production environments.*