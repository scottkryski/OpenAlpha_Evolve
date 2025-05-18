// README.md
# OpenAlpha_Evolve: Regenerating Autonomous Algorithmic Discovery 🚀

![openalpha_evolve_workflow](https://github.com/user-attachments/assets/9d4709ad-0072-44ae-bbb5-7eea1c5fa08c)
*Note: The image URL above might be temporary. If contributing, consider embedding images in the repo.*

OpenAlpha_Evolve is an open-source Python framework inspired by the groundbreaking research on autonomous coding agents like DeepMind's AlphaDev. It's a **regeneration** of the core idea: an intelligent system that iteratively writes, tests, and improves code using Large Language Models (LLMs), guided by the principles of evolution. This version supports:
*   Google Gemini API
*   OpenAI API (GPT-3.5, GPT-4, etc.)
*   OpenAI-Compatible Endpoints for Local LLMs (e.g., via Ollama, LiteLLM, vLLM)

Our mission is to provide an accessible, understandable, and extensible platform for researchers, developers, and enthusiasts to explore the fascinating intersection of AI, code generation, and automated problem-solving.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ The Vision: AI-Driven Algorithmic Innovation

Imagine an agent that can:

*   Understand a complex problem description.
*   Generate initial algorithmic solutions using powerful LLMs.
*   Rigorously test its own code.
*   Learn from failures and successes.
*   Evolve increasingly sophisticated and efficient algorithms over time.

OpenAlpha_Evolve is a step towards this vision. It's not just about generating code; it's about creating a system that *discovers* and *refines* solutions autonomously.

---

## 🧠 How It Works: The Evolutionary Cycle

OpenAlpha_Evolve employs a modular, agent-based architecture:

1.  **Task Definition**: You define the algorithmic "quest" – the problem, input/output examples, and allowed library imports.
2.  **Prompt Engineering (`PromptDesignerAgent`)**: Crafts intelligent prompts for the LLM (initial, mutation, bug-fix).
3.  **Code Generation (`CodeGeneratorAgent`)**: Powered by your chosen LLM (Gemini, OpenAI, or OpenAI-compatible local model), this agent generates Python code.
4.  **Evaluation (`EvaluatorAgent`)**: Tests the generated code for syntax and functional correctness against task examples in an isolated environment. Scores fitness (correctness, runtime).
5.  **Database (`DatabaseAgent`)**: Stores programs, fitness, and evolutionary history (currently In-Memory).
6.  **Selection (`SelectionControllerAgent`)**: Applies "survival of the fittest" to select parents for the next generation and survivors.
7.  **Iteration**: Repeats the cycle, aiming for better solutions each generation.
8.  **Orchestration (`TaskManagerAgent`)**: Coordinates all agents and manages the evolutionary loop.

---

## 🚀 Key Features

*   **Flexible LLM Backends**:
    *   **Google Gemini API**: Leverages models like `gemini-1.5-flash` or `gemini-1.5-pro`.
    *   **OpenAI API**: Supports models like `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, etc.
    *   **OpenAI-Compatible Endpoints**: Enables use of local LLMs served via tools like Ollama, LiteLLM, Jan.ai, vLLM, etc., that expose an OpenAI-compatible API.
*   **Evolutionary Algorithm Core**: Implements iterative improvement through selection, LLM-driven mutation/bug-fixing, and survival.
*   **Modular Agent Architecture**: Easily extend or replace components (e.g., use a different LLM evaluation strategy, or database).
*   **Automated Program Evaluation**: Syntax checking and functional testing against user-provided examples with timeout mechanisms.
*   **Configuration Management**: Easily tweak parameters (population size, generations, LLM models, API settings) via `.env` and `config/settings.py`.
*   **Detailed Logging**: Comprehensive logs for insights into the evolutionary process.
*   **Open Source & Extensible**: Built with Python, designed for experimentation and community contributions.

---

## 📂 Project Structure
OpenAlpha_Evolve/
├── agents/ # Core intelligent agents (subdirectories for each)
│ ├── code_generator/ # Agent for LLM-based code generation
│ ├── database_agent/ # Agent for storing/retrieving programs
│ ├── evaluator_agent/ # Agent for evaluating generated code
│ ├── prompt_designer/ # Agent for designing LLM prompts
│ ├── selection_controller/ # Agent for evolutionary selection
│ └── task_manager/ # Agent orchestrating the main loop
├── config/ # Configuration files (settings.py)
├── core/ # Core interfaces, data models (Program, TaskDefinition)
├── utils/ # Utility functions (currently minimal)
├── tests/ # Unit and integration tests (to be expanded)
├── scripts/ # Helper scripts (currently minimal)
├── main.py # Main entry point to run the system from CLI
├── app.py # Gradio Web UI entry point
├── requirements.txt # Project dependencies
├── .env.example # Example for environment variables (copy to .env)
├── .gitignore # Specifies intentionally untracked files
├── LICENSE.md # Project's license information (MIT License)
└── README.md # This file!


---

## 🏁 Getting Started

1.  **Prerequisites**:
    *   Python 3.10+
    *   `pip` for package management
    *   `git` for cloning

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shyamsaktawat/OpenAlpha_Evolve.git # Or your forked repository URL
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

5.  **Set Up Environment Variables (Crucial for LLM Access)**:
    *   Copy `.env.example` to a new file named `.env` in the project root:
        ```bash
        cp .env.example .env
        ```
    *   **Edit the `.env` file** and configure it for **one** LLM provider of your choice:

        **Option 1: For Google Gemini API**
        ```dotenv
        LLM_PROVIDER="gemini"
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        GEMINI_MODEL_NAME="gemini-1.5-flash-latest" # Or "gemini-1.5-pro-latest"
        ```
        *Obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).*

        **Option 2: For OpenAI API**
        ```dotenv
        LLM_PROVIDER="openai"
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # e.g., sk-....
        OPENAI_MODEL_NAME="gpt-4o" # Or "gpt-3.5-turbo", "gpt-4-turbo", etc.
        ```

        **Option 3: For OpenAI-Compatible Local LLM (e.g., using Ollama)**
        ```dotenv
        LLM_PROVIDER="openai_compatible"

        # Example for Ollama running locally:
        # 1. Ensure Ollama is installed and running (ollama serve)
        # 2. Pull a model: ollama pull llama3
        OPENAI_COMPATIBLE_ENDPOINT_URL="http://localhost:11434/v1"
        OPENAI_COMPATIBLE_MODEL_NAME="llama3" # The model name as served by Ollama (e.g., "llama3", "codellama:7b-instruct", "qwen2:7b")
        OPENAI_COMPATIBLE_API_KEY="ollama"    # Often "ollama", "not-needed", or any string if not required by your endpoint.
        OPENAI_COMPATIBLE_NUM_CTX="4096"      # Optional: context window size if your server/model supports it (e.g., for Ollama's num_ctx).
                                             # Remove or leave blank if not needed/supported.
        ```
        *Ensure your local LLM server (like Ollama) is running and the specified model is downloaded/available.*

6.  **Review Configuration (Optional)**:
    *   Open `config/settings.py`. Here you can adjust evolutionary parameters like `POPULATION_SIZE`, `GENERATIONS`, logging levels, etc. Defaults are generally reasonable for a start.

7.  **Run OpenAlpha_Evolve (CLI)**:
    The `main.py` file is configured with an example task (Dijkstra's algorithm). To run it:
    ```bash
    python main.py
    ```
    Watch the logs in your terminal to see the evolutionary process! Log files are also saved to `alpha_evolve.log` by default.

8.  **Run OpenAlpha_Evolve (Web UI with Gradio)**:
    You can also interact with the system and define tasks through a web UI.
    ```bash
    python app.py
    ```
    Gradio will display a local URL (e.g., `http://127.0.0.1:7860`). Open this in your browser.

---

## 💡 Defining Your Own Algorithmic Quests!

To challenge OpenAlpha_Evolve with a new problem:

1.  **Modify `main.py` (for CLI) or use the Gradio UI (`app.py`)**.
2.  **Define/Update the `TaskDefinition` object with**:
    *   `id`: A unique string identifier (e.g., "sort_list_task").
    *   `description`: A clear, detailed natural language description of the problem. Crucial for the LLM.
    *   `function_name_to_evolve`: The Python function name the agent should create/evolve (e.g., "custom_sort").
    *   `input_output_examples`: A list of dictionaries, each with `"input"` and `"output"`.
        *   Inputs can be single values, lists (for positional args), or dicts (for named args).
        *   Use `float('inf')` or `float('-inf')` directly in Python code for examples (JSON in Gradio needs "Infinity").
    *   `allowed_imports`: List of Python standard libraries allowed (e.g., `["heapq", "math"]`).
    *   (Optional) `evaluation_criteria`: String describing success measures.
    *   (Optional) `initial_code_prompt`: Override default initial prompt.
3.  **Run the agent** as described in step 7 or 8 above.

The quality of your `description` and `input_output_examples` significantly impacts success!

---

## 🔮 The Horizon: Future Evolution

OpenAlpha_Evolve is a living project. Future directions include:

*   **Advanced Evaluation Sandboxing**: More robust, secure sandboxing (e.g., Docker, E2B, Firejail).
*   **Sophisticated Fitness Metrics**: Code complexity, style, resource usage.
*   **Reinforcement Learning for Prompt Strategy**: Dynamic optimization of prompt engineering.
*   **Enhanced Monitoring & Visualization**: Tools to visualize evolution and agent behavior.
*   **Broader LLM Support & Fine-tuning**: Easier integrations and support for fine-tuning LLMs.
*   **Self-Correction & Reflection**: Deeper analysis of failures.
*   **Community-Driven Task Library**: A collection of interesting tasks.
*   **Crossover Implementation**: Adding genetic crossover alongside LLM-driven mutation.

---

## 🤝 Join the Evolution: Contributing

This is an open invitation to collaborate!

*   **Report Bugs**: Find an issue? Create an issue on GitHub!
*   **Suggest Features**: Have an idea? Open an issue to discuss it!
*   **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch (`git checkout -b feature/your-feature-name`).
    *   Write clean, well-documented code. Add tests if applicable.
    *   Ensure changes don't break existing functionality (test other LLM providers if you change shared code).
    *   Submit a pull request with a clear description of your changes!

Let's evolve this agent together!

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE.md` file for details.

---

## 🙏 Homage

OpenAlpha_Evolve is proudly inspired by the pioneering work of Google DeepMind (e.g., AlphaDev) and other related research in LLM-driven code generation and automated discovery. This project aims to make the core concepts more accessible for broader experimentation and learning.

---

*Disclaimer: This is an experimental project. Generated code may not always be optimal, correct, or secure. Always review and test code thoroughly, especially before using it in production environments.*