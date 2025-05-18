from typing import Optional, Dict, Any
import logging
import json
import re
from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)

class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        logger.info(f"PromptDesignerAgent initialized for task: {self.task_definition.id}")

    def design_initial_prompt(self) -> str:
        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
        
        graph_initialization_advice = (
            "A common requirement in graph algorithms like Dijkstra's is to initialize distances for all nodes. "
            "Ensure your solution correctly identifies *all unique nodes* present in the input `graph` for this initialization. "
            "This includes nodes that might only appear as destinations in edges (values in the adjacency sub-dictionaries) "
            "and not as primary keys in the `graph`'s top-level adjacency list (i.e., nodes with no outgoing edges). "
            "You will need to iterate through the graph structure (both keys and all neighbor values) to collect all unique node identifiers first, "
            "before setting up your distances dictionary (e.g., `distances = {node: float('inf') for node in all_graph_nodes}`). "
            "Also, remember that when processing a `current_node`, if it has no outgoing edges (i.e., it's not a key in the `graph` dictionary, or `graph[current_node]` is empty), "
            "you should handle this gracefully (e.g., by checking `if current_node in graph:` before trying `graph[current_node].items()`)."
        )

        prompt = (
            f"Task: {self.task_definition.description}\n\n"
            f"You are to implement a Python function named `{self.task_definition.function_name_to_evolve}`.\n"
            f"The function signature, based on the problem, is expected to be: `{self.task_definition.function_name_to_evolve}(graph, source_node)` "
            f"where `graph` is an adjacency list (dictionary of dictionaries) and `source_node` is the starting node.\n"
            f"The function must RETURN a dictionary mapping nodes to their shortest distance from the source. Use `float('inf')` for unreachable nodes.\n\n"
            f"Input examples showing the structure of arguments passed to `{self.task_definition.function_name_to_evolve}` and their expected RETURN values:\n"
            f"{json.dumps(self.task_definition.input_output_examples, indent=2)}\n\n"
            f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"Allowed standard library imports for your solution: {self.task_definition.allowed_imports}. Do not use other external libraries.\n\n"
            f"VERY IMPORTANT - Graph Processing Advice: {graph_initialization_advice}\n\n"
            f"IMPORTANT: Provide ONLY the complete Python code for the function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result, it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _get_key_error_advice(self, error_message: str, execution_output: Optional[str], program_code: str) -> str:
        specific_advice = ""
        key_error_match_primary = re.search(r"KeyError:\s*['\"]?(\w+)['\"]?", error_message)
        key_error_match_secondary = None
        if execution_output:
            key_error_match_secondary = re.search(r"KeyError:\s*['\"]?(\w+)['\"]?", execution_output)
        
        key_error_match = key_error_match_primary or key_error_match_secondary
        
        if key_error_match and "graph" in self.task_definition.description.lower() and \
        ("dijkstra" in self.task_definition.description.lower() or "path" in self.task_definition.description.lower()):
            
            missing_key_raw = key_error_match.group(1)
            missing_key_cleaned = missing_key_raw.strip("'\"")

            # Check if the error is related to accessing graph[current_node].items()
            if f"graph[{missing_key_cleaned}]" in error_message or (execution_output and f"graph[{missing_key_cleaned}]" in execution_output) or \
            "graph[current_node].items()" in program_code or "graph[current_node]" in program_code: # Check code structure too
                specific_advice = (
                    f"\nThe error `KeyError: {missing_key_raw}` likely occurred when trying to access `graph[{missing_key_cleaned}]` (e.g., `graph[current_node].items()`). "
                    f"This usually happens if `current_node` (which is '{missing_key_cleaned}' in this case) does not have any outgoing edges defined in the `graph` dictionary "
                    f"(i.e., it's not a key in `graph`, or `graph['{missing_key_cleaned}']` does not exist).\n"
                    f"To fix this: Before iterating through neighbors like `for neighbor, weight in graph[current_node].items():`, "
                    f"you MUST add a check: `if current_node in graph:`. Only proceed to access `graph[current_node].items()` if this condition is true. "
                    f"If `current_node` is not in `graph`, it means it's a terminal node with no outgoing edges to process, so you can skip iterating its neighbors.\n"
                    f"This check is crucial for nodes that are destinations but not sources of further edges.\n"
                )
            else: # General advice for KeyError regarding node initialization
                specific_advice = (
                    f"\nThe error `KeyError: {missing_key_raw}` often indicates that a node (potentially node '{missing_key_cleaned}') "
                    f"was accessed (e.g., as a neighbor or when trying to look up its distance) but was not properly initialized "
                    f"in a data structure like the 'distances' dictionary.\n"
                    f"To fix this: Before starting the main algorithm loop (e.g., Dijkstra's), you MUST identify ALL unique nodes present in the input `graph`. "
                    f"This includes nodes that are primary keys in the adjacency list AND nodes that *only* appear as destinations. "
                    f"Create a set of all such nodes, and then initialize your distances dictionary (e.g., `distances = {{node: float('inf') for node in all_nodes_in_graph}}`) for every node in this complete set, "
                    f"setting the source node's distance to 0.\n"
                )
        return specific_advice

    def design_mutation_prompt(self, program: Program, evaluation_feedback: dict | None = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Parent program code:\n{program.code}")
        
        feedback_prompt_segment = ""
        specific_key_error_advice = ""

        if evaluation_feedback:
            logger.debug(f"Evaluation feedback received for parent:\n{evaluation_feedback}")
            fitness_scores = evaluation_feedback.get("fitness_scores", {})
            if not isinstance(fitness_scores, dict): fitness_scores = {}

            correctness = fitness_scores.get("correctness_score", 0) * 100
            runtime = fitness_scores.get("runtime_ms", "N/A")
            errors = evaluation_feedback.get("errors", []) 

            feedback_prompt_segment = f"The previous version of this code had a correctness score of {correctness:.2f}% and a runtime of {runtime} ms.\n"
            if errors:
                errors_str = "; ".join(errors)
                feedback_prompt_segment += f"It produced the following errors/issues during evaluation: {errors_str}\n"
                if correctness < 100:
                    specific_key_error_advice = self._get_key_error_advice(errors_str, None, program.code)
            
            if correctness < 100 and not errors:
                feedback_prompt_segment += "It did not achieve 100% correctness but did not produce explicit execution errors. Review logic for test case failures based on the task's input/output examples.\n"
        else:
            feedback_prompt_segment = "The previous version of this code was evaluated, but detailed feedback is not available. Attempt a general improvement based on the task requirements.\n"

        prompt = (
            f"Task: {self.task_definition.description}\n\n"
            f"You are to improve a Python function named `{self.task_definition.function_name_to_evolve}`.\n"
            f"The function signature is expected to be: `{self.task_definition.function_name_to_evolve}(graph, source_node)`.\n"
            f"The function must RETURN a dictionary mapping nodes to their shortest distance. Use `float('inf')` for unreachable nodes.\n\n"
            f"Allowed standard library imports: {self.task_definition.allowed_imports}. Do not use other external libraries.\n\n"
            f"Current Code:\n```python\n{program.code}\n```\n\n"
            f"Evaluation Feedback on Current Code:\n{feedback_prompt_segment}\n"
            f"{specific_key_error_advice}" 
            f"Instruction: Based on the task, the current code, and the evaluation feedback, provide an improved version of the function `{self.task_definition.function_name_to_evolve}`. "
            f"Focus on improving correctness to pass all test cases (refer to task description for examples and the specific advice above if a KeyError was mentioned) and then efficiency. "
            f"If a KeyError related to accessing `graph[current_node]` occurred, ensure you check `if current_node in graph:` before trying to iterate its neighbors. "
            f"Also ensure all nodes are correctly initialized in the `distances` dictionary.\n\n"
            f"IMPORTANT: Provide ONLY the complete Python code for the improved function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result, it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed mutation prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: str | None = None) -> str:
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Buggy program code:\n{program.code}")
        logger.debug(f"Primary Error message from evaluation: {error_message}")
        if execution_output: 
            logger.debug(f"Context for bug fix (e.g. full list of errors): {execution_output}")

        output_segment = f"Additional Context (e.g., full list of errors or relevant prior outputs):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the primary error.\n"
        
        specific_key_error_advice = self._get_key_error_advice(error_message, execution_output, program.code)

        prompt = (
            f"Task: {self.task_definition.description}\n\n"
            f"You are to fix a Python function named `{self.task_definition.function_name_to_evolve}`.\n"
            f"The function signature is expected to be: `{self.task_definition.function_name_to_evolve}(graph, source_node)`.\n"
            f"The function must RETURN a dictionary mapping nodes to their shortest distance. Use `float('inf')` for unreachable nodes.\n\n"
            f"Allowed standard library imports: {self.task_definition.allowed_imports}. Do not use other external libraries.\n\n"
            f"Buggy Code:\n```python\n{program.code}\n```\n\n"
            f"Primary Error Encountered During Evaluation: {error_message}\n"
            f"{output_segment}"
            f"{specific_key_error_advice}" 
            f"Instruction: The above code produced an error or failed test cases. Please analyze the code, the error, any provided context, and the specific advice (if any) to identify and fix the bug(s). "
            f"If the error was a `KeyError` related to accessing `graph[current_node]` (e.g. for its items), "
            f"the fix is usually to add a condition `if current_node in graph:` before trying to access its neighbors. "
            f"Also, ensure all nodes present in the graph (keys and neighbor values) are initialized in your `distances` structure.\n\n"
            f"IMPORTANT: Provide ONLY the complete Python code for the fixed function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result, it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed bug-fix prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        # This agent's primary role is fulfilled through specific design methods (design_initial_prompt, etc.)
        # rather than a general execute. However, to satisfy the interface, we provide a default.
        logger.warning(f"PromptDesignerAgent.execute() called. Task: {self.task_definition.id}. Args: {args}, Kwargs: {kwargs}. "
                       f"This agent is typically used via its specific design methods. Returning a generic message.")
        # Depending on the expected behavior for a generic execute, you might:
        # 1. Raise NotImplementedError if it should never be called directly.
        # 2. Return a default prompt (e.g., initial prompt) if that makes sense.
        # 3. Perform a specific action if `args` or `kwargs` indicate it.
        # For now, let's make it clear it's not the primary use.
        if 'action' in kwargs:
            action = kwargs.get('action')
            if action == 'design_initial_prompt':
                return self.design_initial_prompt()
            # Add more actions if needed, or raise error for unhandled ones.
            else:
                 raise NotImplementedError(f"PromptDesignerAgent.execute() does not support action: {action}. Call specific design methods.")
        raise NotImplementedError("PromptDesignerAgent.execute() called without a specific 'action'. Please use specific design methods like design_initial_prompt(), design_mutation_prompt(), etc.")

# Example Usage:
if __name__ == '__main__':
    import json # Add this import for json.dumps
    logging.basicConfig(level=logging.DEBUG)

    sample_task_def = TaskDefinition(
        id="task_001_designer_test",
        description="Create a Python function `sum_list(numbers)` that returns the sum of a list of integers. Handle empty lists by returning 0.",
        function_name_to_evolve="sum_list",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6}, 
            {"input": [], "output": 0}
        ],
        allowed_imports=["math"],
        evaluation_criteria="Must be correct and efficient."
    )
    designer = PromptDesignerAgent(task_definition=sample_task_def)

    initial_prompt = designer.design_initial_prompt()
    print("--- Initial Prompt ---")
    print(initial_prompt)

    sample_program = Program(
        id="prog_001",
        code="def sum_list(numbers):\n  # Buggy implementation\n  return sum(numbers) if numbers else 'Error'",
        fitness_scores={"correctness_score": 0.0, "runtime_ms": 10.0},
        generation=1,
        errors=["TypeError: unsupported operand type(s) for +: 'int' and 'str' on empty list with 'Error' return"]
    )
    mutation_prompt = designer.design_mutation_prompt(sample_program, evaluation_feedback={"errors": sample_program.errors, "fitness_scores": sample_program.fitness_scores})
    print("\n--- Mutation Prompt ---")
    print(mutation_prompt)

    bug_fix_prompt = designer.design_bug_fix_prompt(sample_program, error_message="TypeError", execution_output="Fails when list is empty")
    print("\n--- Bug-Fix Prompt ---")
    print(bug_fix_prompt)

    # Example of calling execute (though not its primary use)
    # This will now raise NotImplementedError unless specific action is passed
    # try:
    #     print("\n--- Testing Generic Execute (will raise error) ---")
    #     await designer.execute()
    # except NotImplementedError as e:
    #     print(f"Caught expected error: {e}")

    # try:
    #     print("\n--- Testing Execute with Action ---")
    #     initial_via_execute = await designer.execute(action="design_initial_prompt")
    #     print(f"Initial prompt via execute: {initial_via_execute[:100]}...") # Print first 100 chars
    # except NotImplementedError as e:
    #      print(f"Error during execute with action: {e}")