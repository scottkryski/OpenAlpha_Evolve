import asyncio
import logging
import sys # Required for sys.maxsize in task definition, math for float('inf')
import math # ### PR REVIEW: Added math import for float('inf')

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
# from config import settings # Not strictly needed here if TaskManagerAgent handles it

# Configure logging
logging.basicConfig(
    level=logging.INFO, # ### PR REVIEW: Consider using settings.LOG_LEVEL here
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], # ### PR REVIEW: Consider adding FileHandler using settings.LOG_FILE
)
logger = logging.getLogger(__name__)

async def run_alpha_evolve_pro():
    """
    Initializes and runs the AlphaEvolve Pro task manager with a specific task.
    """
    logger.info("Starting AlphaEvolve Pro...")

    # 1. Define the algorithmic task
    # Task: Implement Dijkstra's algorithm for shortest paths in a weighted graph using adjacency list.
    dijkstra_task = TaskDefinition(
        id="dijkstra_shortest_path_adj_list", # ### PR REVIEW: More descriptive ID
        description=(
            "Implement Dijkstra's algorithm to find the shortest paths from a source node "
            "to all other nodes in a weighted graph. The graph is represented as an adjacency list "
            "where keys are node IDs (integers or strings convertible to integers) and values are dictionaries of "
            "neighbor_node_id: weight. The function should take the graph and the source_node "
            "as input and return a dictionary of node_id: shortest_distance_from_source. "
            "Use float('inf') for unreachable nodes. Ensure all nodes mentioned in the graph "
            "(both as keys and as neighbors) are considered for distance initialization."
        ), # ### PR REVIEW: Corrected typo Dijkstra\\'s -> Dijkstra's
        function_name_to_evolve="dijkstra",
        input_output_examples=[
            {
                "input": {"graph": { # String keys for nodes as they might appear in JSON
                    "0": {"1": 4, "7": 8},
                    "1": {"0": 4, "2": 8, "7": 11},
                    "2": {"1": 8, "3": 7, "8": 2, "5": 4},
                    "3": {"2": 7, "4": 9, "5": 14},
                    "4": {"3": 9, "5": 10},
                    "5": {"2": 4, "3": 14, "4": 10, "6": 2},
                    "6": {"5": 2, "7": 1, "8": 6},
                    "7": {"0": 8, "1": 11, "6": 1, "8": 7}, # Added 8 as neighbor of 7 for completeness from diagram
                    "8": {"2": 2, "6": 6, "7": 7}
                }, "source_node": 0}, # Source node typically passed as int if function expects int
                "output": {0: 0, 1: 4, 2: 12, 3: 19, 4: 21, 5: 11, 6: 9, 7: 8, 8: 14} # Output keys are ints
            },
            {
                "input": {"graph": {
                    "0": {"1": 10},
                    "1": {"0": 10},
                    "2": {"3": 5},
                    "3": {"2": 5}
                }, "source_node": 0},
                "output": {0: 0, 1: 10, 2: math.inf, 3: math.inf} # Using math.inf
            },
            {
                "input": {"graph": {"0": {}}, "source_node": 0},
                "output": {0: 0}
            },
            {
                "input": {"graph": {
                    "0": {"1": 1},
                    "1": {"0": 1, "2": 2}, # Assuming bidirectional for 0-1
                    "2": {"1": 2, "3": 3}  # Assuming bidirectional for 1-2
                }, "source_node": 0},
                "output": {0: 0, 1: 1, 2: 3, 3: 6} 
            },
            {
                "input": {"graph": {
                    "0": {"1": 2, "2": 5},
                    "1": {"2": 1, "3": 6}, # Cycle 0-1-2-0
                    "2": {"0": 5, "1": 1, "3": 2}, 
                    "3": {} # Node 3 has no outgoing edges
                }, "source_node": 0},
                "output": {0: 0, 1: 2, 2: 3, 3: 5} 
            }
        ],
        evaluation_criteria=( # ### PR REVIEW: String is fine, or make it a dict for structured criteria.
            "Correctness: Must pass all test cases. Output for unreachable nodes should be float('inf'). "
            "Efficiency: Aim for efficient Dijkstra implementation (e.g., using a min-priority queue like heapq). "
            "Standard library imports like 'heapq', 'sys', 'math' are allowed. "
            "Do not use external libraries not available in a standard Python environment."
        ),
        allowed_imports=["heapq", "sys", "math"], # What the generated code can import
        # initial_code_prompt is optional. If None, PromptDesignerAgent will use its default logic.
        # ### PR REVIEW: Specific hints for this task could go here or be part of a more structured TaskDefinition.
        # ### E.g., task_definition.hints = {"initialization": "Remember to initialize distances for ALL nodes...",
        # ###                                "key_error": "A KeyError often means..."}
    )

    # 2. Initialize the Task Manager Agent
    task_manager = TaskManagerAgent(task_definition=dijkstra_task)

    # 3. Run the evolutionary process
    try:
        best_program = await task_manager.execute()
        if best_program:
            logger.info(f"AlphaEvolve Pro finished. Overall best program found for task '{dijkstra_task.id}':")
            logger.info(f"Program ID: {best_program.id}")
            correctness_score = best_program.fitness_scores.get('correctness_score', 0.0) 
            runtime_ms = best_program.fitness_scores.get('runtime_ms', 'N/A')
            logger.info(f"Fitness: Correctness={correctness_score*100:.2f}%, Runtime={runtime_ms}ms")
            logger.info(f"Generation: {best_program.generation}")
            logger.info("Code:\n" + best_program.code)
            if best_program.errors:
                logger.warning(f"Best program had errors during its last evaluation: {best_program.errors}")
        else:
            logger.info(f"AlphaEvolve Pro finished. No successful program was evolved for task '{dijkstra_task.id}'.")
    except Exception as e:
        logger.error(f"An error occurred during the evolutionary process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_alpha_evolve_pro())