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
        
        # ### PR REVIEW: This advice is very specific to graph problems (Dijkstra).
        # ### For a generic PromptDesignerAgent, this kind of detailed, task-specific advice
        # ### should ideally come from the TaskDefinition (e.g., task.hints["initial_prompt"])
        # ### or be constructed based on task properties (e.g., if task_type is "graph_algorithm").
        # ### Hardcoding it here limits reusability for non-graph tasks.
        # ### Consider how to make this more general for the PR.
        # ### For now, I'll leave it but add this comment.
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
        # Use initial_code_prompt from task_definition if provided, otherwise default
        initial_instruction = self.task_definition.initial_code_prompt or \
            f"Provide an initial Python solution for the following problem. You are to implement a Python function named `{self.task_definition.function_name_to_evolve}`."


        prompt = (
            f"Task Description: {self.task_definition.description}\n\n"
            f"{initial_instruction}\n"
            f"The function signature, based on the problem, is expected to be: `{self.task_definition.function_name_to_evolve}({self._get_argument_list_string()})` " # Dynamically generate arg list if possible
            f"where `graph` is an adjacency list (dictionary of dictionaries representing node_id -> {{neighbor_id: weight}}) and `source_node` is the starting node ID.\n"
            f"The function must RETURN a dictionary mapping node IDs to their shortest distance from the source. Use `float('inf')` for unreachable nodes.\n\n"
            f"Input/Output Examples (showing structure of arguments and expected return values):\n"
            f"{json.dumps(self.task_definition.input_output_examples, indent=2)}\n\n"
            f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"Allowed standard library imports for your solution: {self.task_definition.allowed_imports if self.task_definition.allowed_imports else 'None specified, assume common math/sys if needed'}. Do not use other external libraries.\n\n"
        )
        # Conditionally add graph-specific advice if it seems like a graph task. This is a simple heuristic.
        if "graph" in self.task_definition.description.lower() and "dijkstra" in self.task_definition.description.lower():
             prompt += f"VERY IMPORTANT - Graph Processing Advice: {graph_initialization_advice}\n\n"
        
        # Add task-specific hints if available from TaskDefinition (ideal future improvement)
        # if hasattr(self.task_definition, 'hints') and self.task_definition.hints.get('initial_prompt'):
        #    prompt += f"Specific Hints for this task:\n{self.task_definition.hints['initial_prompt']}\n\n"

        prompt += (
            f"IMPORTANT: Provide ONLY the complete Python code for the function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result; it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _get_argument_list_string(self) -> str:
        """Helper to create a string representation of function arguments for prompts."""
        if self.task_definition.input_output_examples:
            first_input_example = self.task_definition.input_output_examples[0]
            if "input" in first_input_example: # Check if "input" key exists
                first_input = first_input_example["input"]
                if isinstance(first_input, dict):
                    return ", ".join(first_input.keys())
                elif isinstance(first_input, list): # Assuming args are positional for a list
                    return ", ".join([f"arg{i+1}" for i in range(len(first_input))])
                # Handle single non-dict/list argument if necessary, e.g. if input is just a number or string
                elif first_input is not None : # Not a dict, not a list, but not None
                    return "arg1" 
            # If "input" key is missing or its value is None, fall through to default
        return "..." # Default if no examples or input key is missing/None


    def _get_key_error_advice(self, error_message: str, execution_output: Optional[str], program_code: str) -> str:
        # ### PR REVIEW: This advice is also very specific to graph KeyErrors.
        # ### For a generic agent, this should be generalized or made data-driven from TaskDefinition.
        specific_advice = ""
        key_error_match_primary = re.search(r"KeyError:\s*['\"]?([^'\"]+)['\"]?", error_message) # More general key capture
        key_error_match_secondary = None
        if execution_output:
            key_error_match_secondary = re.search(r"KeyError:\s*['\"]?([^'\"]+)['\"]?", execution_output)
        
        key_error_match = key_error_match_primary or key_error_match_secondary
        
        # Check if it's likely a graph task to provide specific graph advice
        is_graph_task = "graph" in self.task_definition.description.lower() and \
                        ("dijkstra" in self.task_definition.description.lower() or \
                         "path" in self.task_definition.description.lower() or \
                         "node" in self.task_definition.description.lower())

        if key_error_match and is_graph_task:
            missing_key_raw = key_error_match.group(1)
            # missing_key_cleaned = missing_key_raw.strip("'\"") # Already captured without quotes

            current_node_access_pattern = r"graph\[\s*(\w+|current_node)\s*\]" 
            if re.search(current_node_access_pattern, program_code) and \
               (f"graph[{missing_key_raw}]" in error_message or \
                (execution_output and f"graph[{missing_key_raw}]" in execution_output) or \
                ".items()" in error_message or (execution_output and ".items()" in execution_output)):
                specific_advice = (
                    f"\nThe error `KeyError: {missing_key_raw}` likely occurred when trying to access `graph[{missing_key_raw}]` (e.g., `graph[current_node].items()` if `current_node` was `{missing_key_raw}`). "
                    f"This usually happens if node '{missing_key_raw}' does not have any outgoing edges defined in the `graph` dictionary "
                    f"(i.e., it's not a key in `graph`, or `graph['{missing_key_raw}']` does not exist or is not a dictionary).\n"
                    f"To fix this: Before iterating through neighbors like `for neighbor, weight in graph[current_node].items():`, "
                    f"you MUST add a check: `if current_node in graph and isinstance(graph[current_node], dict):`. "
                    f"Only proceed to access `graph[current_node].items()` if this condition is true. "
                    f"If `current_node` is not in `graph` or `graph[current_node]` is not a dictionary, it means it's a terminal node or has malformed edge data, so you can skip iterating its neighbors.\n"
                )
            else: 
                specific_advice = (
                    f"\nThe error `KeyError: {missing_key_raw}` often indicates that a node (potentially node '{missing_key_raw}') "
                    f"was accessed (e.g., as a neighbor or when trying to look up its distance) but was not properly initialized "
                    f"in a data structure like the 'distances' dictionary.\n"
                    f"To fix this: Before starting the main algorithm loop (e.g., Dijkstra's), you MUST identify ALL unique nodes present in the input `graph`. "
                    f"This includes nodes that are primary keys in the adjacency list AND nodes that *only* appear as destinations. "
                    f"Create a set of all such nodes, and then initialize your distances dictionary (e.g., `distances = {{node: float('inf') for node in all_nodes_in_graph}}`) for every node in this complete set, "
                    f"setting the source node's distance to 0.\n"
                )
        elif key_error_match: 
             missing_key_raw = key_error_match.group(1) # Ensure missing_key_raw is defined here too
             specific_advice = (
                f"\nThe error `KeyError: {missing_key_raw}` suggests an attempt to access a dictionary key that does not exist. "
                f"Review the code to ensure that the key '{missing_key_raw}' is expected to be in the dictionary at that point, "
                f"or add checks (e.g., `if '{missing_key_raw}' in my_dict:`) or use `.get()` with a default value.\n"
            )

        return specific_advice

    def design_mutation_prompt(self, program: Program, evaluation_feedback: Optional[Dict] = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Parent program code:\n{program.code}")
        
        feedback_prompt_segment = ""
        specific_error_advice = "" 

        if evaluation_feedback:
            logger.debug(f"Evaluation feedback received for parent:\n{json.dumps(evaluation_feedback, indent=2)}")
            fitness_scores = evaluation_feedback.get("fitness_scores", {})
            if not isinstance(fitness_scores, dict): fitness_scores = {} 

            correctness = fitness_scores.get("correctness_score", 0.0) * 100 
            runtime = fitness_scores.get("runtime_ms", "N/A")
            errors = evaluation_feedback.get("errors", []) 
            if not isinstance(errors, list): errors = [] 

            feedback_prompt_segment = f"The previous version of this code had a correctness score of {correctness:.2f}% and a runtime of {runtime} ms.\n"
            if errors:
                errors_str = "; ".join([str(e) for e in errors]) 
                feedback_prompt_segment += f"It produced the following errors/issues during evaluation: {errors_str}\n"
                if correctness < 100 or any("error" in str(e).lower() for e in errors): 
                    specific_error_advice = self._get_key_error_advice(errors_str, errors_str, program.code)
            
            if correctness < 100 and not errors and not specific_error_advice: 
                feedback_prompt_segment += "It did not achieve 100% correctness but did not produce explicit execution errors. Review logic for test case failures based on the task's input/output examples. Focus on edge cases or complex scenarios described in the task.\n"
        else:
            feedback_prompt_segment = "The previous version of this code was evaluated, but detailed feedback is not available. Attempt a general improvement based on the task requirements.\n"

        prompt = (
            f"Task Description: {self.task_definition.description}\n\n"
            f"You are to improve a Python function named `{self.task_definition.function_name_to_evolve}`.\n"
            f"The function signature is expected to be: `{self.task_definition.function_name_to_evolve}({self._get_argument_list_string()})`.\n"
            f"The function must RETURN the result as specified in the task (e.g., a dictionary for Dijkstra).\n\n"
            f"Allowed standard library imports: {self.task_definition.allowed_imports if self.task_definition.allowed_imports else 'None specified'}.\n\n"
            f"Current Code (to be improved):\n```python\n{program.code}\n```\n\n"
            f"Evaluation Feedback on Current Code:\n{feedback_prompt_segment}\n"
            f"{specific_error_advice if specific_error_advice else ''}\n" 
            f"Instruction: Based on the task, the current code, and the evaluation feedback, provide an improved version of the function `{self.task_definition.function_name_to_evolve}`. "
            f"Focus on improving correctness to pass all test cases (refer to task description for examples and the specific advice above if an error was mentioned) and then efficiency. "
            f"\n\nIMPORTANT: Provide ONLY the complete Python code for the improved function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result, it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed mutation prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: Optional[str] = None) -> str:
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Buggy program code:\n{program.code}")
        logger.debug(f"Primary Error message from evaluation: {error_message}")
        if execution_output: 
            logger.debug(f"Context for bug fix (e.g. full list of errors): {execution_output}")

        output_segment = f"Additional Context (e.g., full list of errors or relevant prior outputs):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the primary error.\n"
        
        specific_error_advice = self._get_key_error_advice(error_message, execution_output, program.code)

        prompt = (
            f"Task Description: {self.task_definition.description}\n\n"
            f"You are to fix a Python function named `{self.task_definition.function_name_to_evolve}`.\n"
            f"The function signature is expected to be: `{self.task_definition.function_name_to_evolve}({self._get_argument_list_string()})`.\n"
            f"The function must RETURN the result as specified.\n\n"
            f"Allowed standard library imports: {self.task_definition.allowed_imports if self.task_definition.allowed_imports else 'None specified'}.\n\n"
            f"Buggy Code:\n```python\n{program.code}\n```\n\n"
            f"Primary Error Encountered During Evaluation: {error_message}\n"
            f"{output_segment}"
            f"{specific_error_advice if specific_error_advice else ''}\n" 
            f"Instruction: The above code produced an error or failed test cases. Please analyze the code, the error, any provided context, and the specific advice (if any) to identify and fix the bug(s). "
            f"\n\nIMPORTANT: Provide ONLY the complete Python code for the fixed function `{self.task_definition.function_name_to_evolve}`. "
            f"The function should be self-contained or use only the allowed imports. "
            f"Do NOT include any surrounding text, explanations, example usage, or markdown code fences (like ```python). "
            f"The function must RETURN the result, it should NOT use `print()` for its main output."
        )
        logger.debug(f"Designed bug-fix prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        logger.debug(f"PromptDesignerAgent.execute() called. Task: {self.task_definition.id}. Args: {args}, Kwargs: {kwargs}.")
        
        action = kwargs.get('action')
        if action == 'design_initial_prompt':
            return self.design_initial_prompt()
        elif action == 'design_mutation_prompt':
            program = kwargs.get('program')
            evaluation_feedback = kwargs.get('evaluation_feedback')
            if not isinstance(program, Program):
                raise ValueError("Missing or invalid 'program' for design_mutation_prompt action.")
            return self.design_mutation_prompt(program, evaluation_feedback)
        elif action == 'design_bug_fix_prompt':
            program = kwargs.get('program')
            error_message = kwargs.get('error_message')
            execution_output = kwargs.get('execution_output')
            if not isinstance(program, Program) or not isinstance(error_message, str):
                raise ValueError("Missing or invalid 'program' or 'error_message' for design_bug_fix_prompt action.")
            return self.design_bug_fix_prompt(program, error_message, execution_output)
        else:
            raise NotImplementedError(
                f"PromptDesignerAgent.execute() does not support action: '{action}'. "
                "Call specific design methods directly or provide a valid action."
            )

async def main_test():
    logging.basicConfig(level=logging.DEBUG)

    sample_task_def = TaskDefinition(
        id="task_001_designer_test",
        description="Create a Python function `sum_list(numbers)` that returns the sum of a list of integers. Handle empty lists by returning 0.",
        function_name_to_evolve="sum_list",
        input_output_examples=[
            {"input": {"numbers": [1, 2, 3]}, "output": 6}, 
            {"input": {"numbers": []}, "output": 0}
        ],
        allowed_imports=["math"], # Example
        evaluation_criteria="Must be correct and efficient."
    )
    designer = PromptDesignerAgent(task_definition=sample_task_def)

    initial_prompt = designer.design_initial_prompt()
    print("--- Initial Prompt ---")
    print(initial_prompt)

    sample_program = Program(
        id="prog_001",
        code="def sum_list(numbers):\n  # Buggy implementation\n  s = 0\n  for x in numbers:\n    s += x\n  return s if numbers else 'Error'", 
        fitness_scores={"correctness_score": 0.5, "runtime_ms": 10.0}, 
        generation=1,
        errors=["Test case 2 (Input: {'numbers': []}): Failed. Expected '0', Got ''Error''"]
    )
    
    mutation_feedback = {"errors": sample_program.errors, "fitness_scores": sample_program.fitness_scores}
    mutation_prompt = designer.design_mutation_prompt(sample_program, evaluation_feedback=mutation_feedback)
    print("\n--- Mutation Prompt ---")
    print(mutation_prompt)

    bug_fix_prompt = designer.design_bug_fix_prompt(sample_program, error_message=sample_program.errors[0], execution_output="Fails when list is empty and returns string 'Error' instead of 0.")
    print("\n--- Bug-Fix Prompt ---")
    print(bug_fix_prompt)

    try:
        print("\n--- Testing Execute with Action (Initial Prompt) ---")
        initial_via_execute = await designer.execute(action="design_initial_prompt") 
        print(f"Initial prompt via execute: {initial_via_execute[:150]}...") 
    except NotImplementedError as e:
         print(f"Error during execute with action: {e}")
    except ValueError as e:
         print(f"ValueError during execute: {e}")


if __name__ == '__main__':
    import asyncio
    # ### PR REVIEW: Moved asyncio.run() into the __main__ block.
    asyncio.run(main_test())