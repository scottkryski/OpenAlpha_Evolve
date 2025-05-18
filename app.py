"""
Gradio web interface for OpenAlpha_Evolve.
"""
import gradio as gr
import asyncio
import json
import os
import sys
import time 
import logging
from typing import Dict, Any, Optional


project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"Loaded .env file from: {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. Using system environment variables if set.")

from core.interfaces import TaskDefinition, Program 
from task_manager.agent import TaskManagerAgent
from config import settings

class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_capture = []
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_capture.append(msg)
        except Exception:
            self.handleError(record)
    
    def get_logs(self):
        return "\n".join(self.log_capture)
    
    def clear(self):
        self.log_capture = []

string_handler = StringIOHandler()
string_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else logging.INFO)
root_logger.addHandler(string_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

desired_agent_log_level_str = os.getenv("GRADIO_AGENT_LOG_LEVEL", "INFO").upper()
agent_log_level = getattr(logging, desired_agent_log_level_str, logging.INFO)

for module_name in ['task_manager.agent', 'code_generator.agent', 'evaluator_agent.agent', 'database_agent.agent', 
              'selection_controller.agent', 'prompt_designer.agent']:
    logging.getLogger(module_name).setLevel(agent_log_level)

API_KEY_WARNING = ""
if not settings.LLM_PROVIDER:
    API_KEY_WARNING = "⚠️ LLM_PROVIDER is not set in .env! Please configure it."
elif settings.LLM_PROVIDER == "gemini":
    if not settings.GEMINI_API_KEY or "YOUR_ACTUAL_GEMINI_API_KEY" in settings.GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in settings.GEMINI_API_KEY:
        API_KEY_WARNING = "⚠️ Gemini API key (GEMINI_API_KEY) not properly set for 'gemini' provider in .env."
elif settings.LLM_PROVIDER == "openai":
    if not settings.OPENAI_API_KEY or "YOUR_OPENAI_API_KEY" in settings.OPENAI_API_KEY:
        API_KEY_WARNING = "⚠️ OpenAI API key (OPENAI_API_KEY) not properly set for 'openai' provider in .env."
elif settings.LLM_PROVIDER == "openai_compatible":
    if not settings.OPENAI_COMPATIBLE_ENDPOINT_URL:
        API_KEY_WARNING = "⚠️ OpenAI Compatible Endpoint URL (OPENAI_COMPATIBLE_ENDPOINT_URL) not set for 'openai_compatible' provider in .env."
    elif not settings.OPENAI_COMPATIBLE_MODEL_NAME:
         API_KEY_WARNING = "⚠️ OpenAI Compatible Model Name (OPENAI_COMPATIBLE_MODEL_NAME) not set for 'openai_compatible' provider in .env."
else:
    API_KEY_WARNING = f"⚠️ Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' configured in .env. Expected 'gemini', 'openai', or 'openai_compatible'."

current_best_program: Optional[Program] = None

async def run_evolution(
    task_id, 
    description, 
    function_name, 
    examples_json, 
    allowed_imports_text,
    population_size, 
    generations,
    progress=gr.Progress(track_tqdm=True) 
):
    string_handler.clear()
    global current_best_program
    current_best_program = None
    results_output = ""

    original_pop_size = settings.POPULATION_SIZE
    original_generations = settings.GENERATIONS

    try:
        logger.info(f"Starting Gradio 'run_evolution' for Task ID: {task_id}")
        logger.info(f"LLM_PROVIDER from settings: {settings.LLM_PROVIDER}")
        if API_KEY_WARNING and ("not set" in API_KEY_WARNING or "Unknown LLM_PROVIDER" in API_KEY_WARNING) :
             results_output = f"Configuration Error: {API_KEY_WARNING}\nPlease check your .env file."
             logger.error(results_output)
             return results_output, string_handler.get_logs()

        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list):
                results_output = "Error: Input/Output Examples must be a JSON list of objects."
                logger.error(results_output)
                return results_output, string_handler.get_logs()
            for i, example_item in enumerate(examples):
                if not isinstance(example_item, dict) or "input" not in example_item or "output" not in example_item:
                    results_output = f"Error in example {i+1}: Each example must be an object with 'input' and 'output' keys."
                    logger.error(results_output)
                    return results_output, string_handler.get_logs()
        except json.JSONDecodeError as e:
            results_output = f"Error: Input/Output Examples must be valid JSON. Details: {e}"
            logger.error(results_output, exc_info=True)
            return results_output, string_handler.get_logs()
        
        allowed_imports = [imp.strip() for imp in allowed_imports_text.split(",") if imp.strip()]
        
        settings.POPULATION_SIZE = int(population_size)
        settings.GENERATIONS = int(generations)
        
        logger.info(f"Runtime Settings: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")

        task = TaskDefinition(
            id=task_id,
            description=description,
            function_name_to_evolve=function_name,
            input_output_examples=examples,
            allowed_imports=allowed_imports
        )
        
        progress(0.01, desc="Initializing Task Manager...") # Start with a small progress value

        class GradioProgressListener(logging.Handler):
            def __init__(self, gr_progress_obj, max_gens_total):
                super().__init__()
                self.gr_progress = gr_progress_obj
                self.max_gens = max_gens_total
                self.current_progress_value = 0.01 # Match initial
                self.current_description = "Initializing..."
                self.last_logged_gen = 0 # Track the generation number from logs

            def update_progress(self, new_desc: str, progress_increment: Optional[float] = None):
                self.current_description = new_desc
                if progress_increment is not None:
                    self.current_progress_value += progress_increment
                    self.current_progress_value = min(self.current_progress_value, 1.0) # Cap at 1.0
                
                # Ensure progress value is passed if track_tqdm is False or for manual updates.
                # If track_tqdm is True, the visual bar is mainly driven by tqdm iterations.
                # Calling progress() with a value might override or complement track_tqdm.
                try:
                    self.gr_progress(self.current_progress_value, desc=self.current_description)
                except Exception as e:
                    logger.warning(f"GradioProgressListener: Error updating progress UI: {e}")


            def emit(self, record):
                msg = record.getMessage()
                new_desc_for_ui = self.current_description # Default to current
                gen_progress_increment = None

                if "--- Generation" in msg and "/" in msg:
                    try:
                        gen_part = msg.split("--- Generation")[1].strip().split("/")[0]
                        current_gen_from_log = int(gen_part)
                        if current_gen_from_log > self.last_logged_gen:
                            self.last_logged_gen = current_gen_from_log
                            # Calculate an increment if max_gens is known
                            if self.max_gens > 0:
                                gen_progress_increment = (1.0 - self.current_progress_value) / (self.max_gens - current_gen_from_log + 1) if (self.max_gens - current_gen_from_log + 1) > 0 else 0.05

                        new_desc_for_ui = f"Running Generation {current_gen_from_log}/{self.max_gens}"
                    except Exception: pass
                elif "Initializing population" in msg:
                    new_desc_for_ui = f"Gen {self.last_logged_gen}: Initializing population"
                elif "Evaluating population" in msg:
                    new_desc_for_ui = f"Gen {self.last_logged_gen}: Evaluating population"
                elif "Selected" in msg and "parents" in msg:
                     new_desc_for_ui = f"Gen {self.last_logged_gen}: Selecting parents"
                elif "Generated" in msg and "offspring" in msg:
                     new_desc_for_ui = f"Gen {self.last_logged_gen}: Generating offspring"
                elif "Evolutionary cycle completed." in msg:
                    new_desc_for_ui = "Finishing up..."
                    # Don't set gen_progress_increment here, let the final progress(1.0, ...) handle it
                
                if new_desc_for_ui != self.current_description or gen_progress_increment is not None:
                    self.update_progress(new_desc_for_ui, gen_progress_increment)


        gr_progress_listener = GradioProgressListener(progress, settings.GENERATIONS)
        gr_progress_listener.setLevel(logging.INFO)
        task_manager_logger = logging.getLogger('task_manager.agent')
        task_manager_logger.addHandler(gr_progress_listener)
        
        task_manager = TaskManagerAgent(task_definition=task)
        logger.info("TaskManagerAgent initialized. Starting evolution...")
        
        best_program_result = await task_manager.execute()
        progress(1.0, desc="Evolution Completed!")

        if best_program_result:
            current_best_program = best_program_result
            program = current_best_program
            results_output = f"✅ Evolution completed successfully! Best solution found:\n\n"
            results_output += f"### Solution Details (ID: {program.id})\n"
            fitness_str = json.dumps(program.fitness_scores, indent=2)
            results_output += f"- **Fitness Scores:**\n```json\n{fitness_str}\n```\n"
            results_output += f"- **Discovered in Generation:** {program.generation}\n\n"
            # Ensure function_name_to_evolve exists on program object or use the input one
            func_name_display = program.function_name_to_evolve if hasattr(program, 'function_name_to_evolve') and program.function_name_to_evolve else function_name
            results_output += f"**Generated Code (`{func_name_display}`):**\n"
            results_output += "```python\n" + program.code.strip() + "\n```\n\n"
            if program.errors:
                error_list_str = "\n".join([f"  - {e}" for e in program.errors])
                results_output += f"**Note:** This program had the following messages/errors during its last evaluation:\n{error_list_str}\n"
            logger.info(f"Evolution completed. Best program ID: {program.id}. Fitness: {program.fitness_scores}")
        else:
            results_output = "❌ Evolution completed, but no suitable solution was found."
            logger.info(results_output)
        
        return results_output, string_handler.get_logs()

    except Exception as e:
        import traceback
        logger.error("Unhandled error during Gradio 'run_evolution'", exc_info=True)
        results_output = f"💥 An unexpected error occurred: {str(e)}\n\nTrace:\n{traceback.format_exc()}"
        return results_output, string_handler.get_logs()
    finally:
        settings.POPULATION_SIZE = original_pop_size
        settings.GENERATIONS = original_generations
        if 'task_manager_logger' in locals() and 'gr_progress_listener' in locals() and gr_progress_listener in task_manager_logger.handlers:
            task_manager_logger.removeHandler(gr_progress_listener)


FIB_EXAMPLES = '''[
    {"input": [0], "output": 0},
    {"input": [1], "output": 1},
    {"input": [5], "output": 5},
    {"input": [10], "output": 55}
]'''

def set_fib_example():
    return (
        "fibonacci_example_001",
        "Write a Python function named `fibonacci` that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1. The input will be a list containing a single integer argument `n` passed to the function.",
        "fibonacci",
        FIB_EXAMPLES,
        ""
    )

DEFAULT_DIJKSTRA_EXAMPLES = """
[
    {
        "input": {
            "graph": {
                "0": {"1": 4, "7": 8},
                "1": {"0": 4, "2": 8, "7": 11},
                "2": {"1": 8, "3": 7, "8": 2, "5": 4},
                "3": {"2": 7, "4": 9, "5": 14},
                "4": {"3": 9, "5": 10},
                "5": {"2": 4, "3": 14, "4": 10, "6": 2},
                "6": {"5": 2, "7": 1, "8": 6},
                "7": {"0": 8, "1": 11, "6": 1, "8": 7},
                "8": {"2": 2, "6": 6, "7": 7}
            },
            "source_node": "0"
        },
        "output": {"0": 0, "1": 4, "2": 12, "3": 19, "4": 21, "5": 11, "6": 9, "7": 8, "8": 14}
    },
    {
        "input": {"graph": {"A": {"B": 1, "C": 4}, "B": {"A":1, "C":2, "D":5}, "C":{"A":4, "B":2, "D":1}, "D":{"B":5, "C":1}}, "source_node": "A"},
        "output": {"A":0, "B":1, "C":3, "D":4}
    }
]
"""

def set_dijkstra_example():
    return (
        "dijkstra_example_001",
        "Implement Dijkstra's algorithm for shortest paths in a weighted graph. The function should take `graph` (an adjacency list dictionary, e.g., {'node_id': {'neighbor_id': weight}}) and `source_node` (the starting node ID) as input. It must return a dictionary mapping all reachable node IDs (including the source) to their shortest distance from the source. Use float('inf') for unreachable nodes. Ensure all nodes present in the graph structure (as keys or neighbor values) are considered for initialization and are present as keys in the output dictionary.",
        "dijkstra",
        DEFAULT_DIJKSTRA_EXAMPLES,
        "heapq, math, sys"
    )

with gr.Blocks(title="OpenAlpha_Evolve", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 OpenAlpha_Evolve: AI-Driven Algorithm Evolution")
    
    if API_KEY_WARNING:
        gr.Warning(API_KEY_WARNING)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📝 Task Definition")
            
            task_id_input = gr.Textbox(label="Task ID", value="fibonacci_001")
            description_input = gr.Textbox(label="Task Description", lines=5)
            function_name_input = gr.Textbox(label="Function Name to Evolve")
            
            examples_json_input = gr.Code(
                label="Input/Output Examples (JSON format)", 
                language="json", 
                lines=10
            )
            
            allowed_imports_input = gr.Textbox(
                label="Allowed Imports (comma-separated)", 
                placeholder="e.g., math, heapq"
            )
            
            gr.Markdown("### Evolutionary Parameters")
            with gr.Row():
                population_size_input = gr.Slider(label="Population Size", minimum=2, maximum=20, value=settings.POPULATION_SIZE, step=1)
                generations_input = gr.Slider(label="Generations", minimum=1, maximum=20, value=settings.GENERATIONS, step=1)
            
            with gr.Accordion("Load Example Task", open=False):
                with gr.Row():
                    fib_btn = gr.Button("🔢 Fibonacci")
                    dijkstra_btn = gr.Button("🗺️ Dijkstra")
            
            run_btn = gr.Button("🚀 Run Evolution", variant="primary", scale=2)
        
        with gr.Column(scale=3):
            gr.Markdown("## 📊 Results & Logs")
            with gr.Tabs():
                with gr.TabItem("🏆 Best Solution"):
                    results_markdown_output = gr.Markdown("Evolution results will appear here...")
                with gr.TabItem("📜 Console Logs"):
                    logs_output = gr.Textbox(label="Log Output (from current run)", lines=25, autoscroll=True, interactive=False, max_lines=1000)
   
    (default_task_id, default_desc, default_func_name, default_examples, default_imports) = set_fib_example()
    task_id_input.value = default_task_id
    description_input.value = default_desc
    function_name_input.value = default_func_name
    examples_json_input.value = default_examples
    allowed_imports_input.value = default_imports

    fib_btn.click(
        set_fib_example, 
        outputs=[task_id_input, description_input, function_name_input, examples_json_input, allowed_imports_input]
    )
    dijkstra_btn.click(
        set_dijkstra_example,
        outputs=[task_id_input, description_input, function_name_input, examples_json_input, allowed_imports_input]
    )
    
    run_btn.click(
        run_evolution, 
        inputs=[
            task_id_input, description_input, function_name_input, 
            examples_json_input, allowed_imports_input,
            population_size_input, generations_input
        ], 
        outputs=[results_markdown_output, logs_output]
    )

if __name__ == "__main__":
    logger.info(f"Starting Gradio app. Project root: {project_root}")
    logger.info(f"Attempting to load .env from: {dotenv_path}")
    logger.info(f"Initial LLM_PROVIDER from settings: {settings.LLM_PROVIDER}")
    if API_KEY_WARNING:
        logger.warning(f"Gradio App API Key/Config Warning: {API_KEY_WARNING}")
    else:
        logger.info("API Key/Config check passed for the selected LLM provider.")

    demo.launch(share=False, debug=True)