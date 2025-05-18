import time
import logging
import traceback
import subprocess
import tempfile
import os
import ast
import math
import json
import asyncio
import sys
from typing import Optional, Dict, Any, Tuple, Union

from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        self.timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        logger.info(f"EvaluatorAgent initialized for task: {self.task_definition.id} with timeout: {self.timeout_seconds}s")

    def _check_syntax(self, code: str) -> Union[str, None]:
        logger.debug("Performing syntax check.")
        try:
            ast.parse(code)
            logger.debug("Syntax check successful.")
            return None
        except SyntaxError as e:
            logger.warning(f"Syntax check failed: {e}")
            # Provide more detailed error if possible (e.g., e.text, e.offset)
            return f"SyntaxError: {e.msg} on line {e.lineno} (offset {e.offset}). Problematic line: '{e.text.strip() if e.text else ''}'"
        except Exception as e: # Catch other potential errors during parsing
            logger.warning(f"Unexpected error during syntax check: {e}", exc_info=True)
            return f"Unexpected SyntaxCheckError: {str(e)}"

    async def _execute_code_safely(
        self, 
        code_to_run: str, 
        function_name_from_task: str,
        timeout_seconds: int
    ) -> Dict[str, Any]:
        logger.debug(f"Preparing to execute code for function '{function_name_from_task}' with task examples. Timeout: {timeout_seconds}s")
        tmp_file_path = None
        try:
            # ### PR REVIEW: Using NamedTemporaryFile is good. Ensure encoding is robust.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as fp:
                fp.write("# -*- coding: utf-8 -*-\n") # Good practice for explicit encoding
                fp.write("import sys\nimport json\nimport math\n") # Common imports

                # Add allowed imports from task definition
                if self.task_definition.allowed_imports:
                    for imp_statement in self.task_definition.allowed_imports:
                        # Basic validation for import statement safety could be added here if needed
                        # For now, assuming imp_statement is a valid Python import string (e.g., "heapq", "collections.defaultdict")
                        fp.write(f"import {imp_statement}\n") 
                
                fp.write("\n# --- Helper function to convert graph keys to int (if applicable for the task) ---\n")
                # ### PR REVIEW: This helper is specific to graph-like tasks where JSON string keys need to be integers.
                # ### For a truly generic evaluator, this might be moved to task-specific pre-processing
                # ### or made configurable via TaskDefinition. For now, it's a pragmatic solution.
                fp.write(
                    "def _convert_graph_keys(graph_obj):\n"
                    "    if not isinstance(graph_obj, dict):\n"
                    "        return graph_obj\n"
                    "    new_graph = {}\n"
                    "    for k, v_obj in graph_obj.items():\n"
                    "        try:\n"
                    "            int_k = int(k) # Attempt conversion only if key is string and looks like int\n"
                    "        except (ValueError, TypeError):\n" 
                    "            int_k = k \n"
                    "        if isinstance(v_obj, dict):\n" # Assuming graph is dict of dicts for weights
                    "            new_v_inner_dict = {}\n"
                    "            for nk, nv in v_obj.items():\n"
                    "                try:\n"
                    "                    int_nk = int(nk)\n"
                    "                except (ValueError, TypeError):\n"
                    "                    int_nk = nk\n"
                    "                new_v_inner_dict[int_nk] = nv\n"
                    "            new_graph[int_k] = new_v_inner_dict\n"
                    "        else:\n" # This case might apply if graph values are not dicts (e.g. list of tuples)
                    "            new_graph[int_k] = v_obj \n"
                    "    return new_graph\n\n"
                )

                fp.write("\n# --- Generated Code Start ---\n")
                fp.write(code_to_run)
                fp.write("\n# --- Generated Code End ---\n\n")
                
                if self.task_definition.input_output_examples and function_name_from_task:
                    # Handle Infinity and NaN in JSON test cases robustly
                    test_cases_json_str = json.dumps(self.task_definition.input_output_examples)
                    # Using replacements that are less likely to clash with user strings
                    test_cases_py_str = test_cases_json_str.replace("Infinity", "float('inf')") # Standard JSON for Infinity
                    test_cases_py_str = test_cases_py_str.replace("-Infinity", "float('-inf')") # Standard JSON for -Infinity
                    test_cases_py_str = test_cases_py_str.replace("NaN", "float('nan')") # Standard JSON for NaN
                    
                    fp.write(f"test_cases_from_task = {test_cases_py_str}\n")
                    fp.write(f"evaluation_results = []\n")
                    fp.write(f"for case_idx, case_data in enumerate(test_cases_from_task):\n")
                    fp.write(f"    input_args = case_data.get('input') # 'input' should be a dict of args or a list\n")
                    fp.write(f"    actual_output = None\n")
                    fp.write(f"    error_occurred = None\n")
                    fp.write(f"    try:\n")
                    # ### PR REVIEW: The argument passing logic below is specific to the Dijkstra example (graph, source_node).
                    # ### For a generic evaluator, this needs to be more flexible.
                    # ### Suggestion: TaskDefinition could specify how to unpack 'input' into function arguments.
                    # ### E.g., input_args could be a list for *args or a dict for **kwargs.
                    # ### For now, it implicitly assumes a 'graph' and 'source_node' in input_args if _convert_graph_keys is used.
                    fp.write(f"        # --- Argument preparation specific to current task structure ---\n")
                    fp.write(f"        func_args = []\n")
                    fp.write(f"        func_kwargs = {{}}\n")
                    fp.write(f"        if isinstance(input_args, dict):\n")
                    fp.write(f"            processed_input_args = {{}}\n")
                    fp.write(f"            for arg_name, arg_val in input_args.items():\n")
                    fp.write(f"                if arg_name == 'graph': # Special handling for 'graph'\n") # Task-specific name
                    fp.write(f"                    processed_input_args[arg_name] = _convert_graph_keys(arg_val)\n")
                    fp.write(f"                else:\n")
                    fp.write(f"                    processed_input_args[arg_name] = arg_val\n")
                    fp.write(f"            func_kwargs = processed_input_args\n")
                    fp.write(f"            actual_output = {function_name_from_task}(**func_kwargs)\n")
                    fp.write(f"        elif isinstance(input_args, list):\n") # If input is a list of args
                    fp.write(f"            # Potentially apply _convert_graph_keys if a graph is passed positionally\n")
                    fp.write(f"            # This part would need more robust logic for generic positional args\n")
                    fp.write(f"            func_args = input_args\n")
                    fp.write(f"            actual_output = {function_name_from_task}(*func_args)\n")
                    fp.write(f"        else:\n") # Single argument case
                    fp.write(f"            actual_output = {function_name_from_task}(input_args)\n")
                    fp.write(f"    except Exception as e_exec:\n")
                    fp.write(f"        actual_output = None\n")
                    fp.write(f"        error_type_name = type(e_exec).__name__\n")
                    fp.write(f"        error_message_raw = str(e_exec)\n")
                    # Sanitize for JSON: escape backslashes first, then quotes, then control characters like newline
                    fp.write(f"        sanitized_error_message = error_message_raw.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\\n', ' ').replace('\\r', '')\n")
                    fp.write(f"        error_occurred = f'EXECUTION_ERROR_CASE_{{case_idx}}: {{error_type_name}}: {{sanitized_error_message}}'\n") 
                    fp.write(f"    evaluation_results.append({{'output': actual_output, 'error': error_occurred}})\n")
                    fp.write(f"print(json.dumps(evaluation_results, allow_nan=True))\n") # allow_nan for float('nan')
                else:
                    # Code for tasks without I/O examples or no function name (e.g., script-like tasks)
                    # This part would need different handling, perhaps capturing stdout.
                    # For now, assuming function-based tasks with I/O examples.
                    fp.write("print(json.dumps({'message': 'No I/O examples or function to evaluate in harness.'}))\n")
                tmp_file_path = fp.name
            
            start_time = time.perf_counter()
            # Using -X utf8 enforces UTF-8 mode for the Python subprocess, good for consistency.
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "utf8", tmp_file_path, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds) # Use member variable
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            # Log stdout/stderr only if they contain data
            if stdout_str: logger.debug(f"Raw stdout from script execution (first 1000 chars): {stdout_str[:1000]}")
            if stderr_str: logger.debug(f"Raw stderr from script execution (first 1000 chars): {stderr_str[:1000]}")


            if process.returncode != 0:
                error_msg = f"Execution failed with return code {process.returncode}. Stderr: {stderr_str[:500]}" # Limit stderr length
                logger.warning(error_msg)
                return {"error": error_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            
            if self.task_definition.input_output_examples and function_name_from_task:
                try:
                    # Attempt to load JSON output from the harness
                    harness_outputs = json.loads(stdout_str) 
                    return {"harness_outputs": harness_outputs, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
                except json.JSONDecodeError as je:
                    err_msg = f"JSONDecodeError: Output from harness was not valid JSON. Error: {je}. Stdout (first 500 chars): {stdout_str[:500]}"
                    logger.warning(err_msg)
                    return {"error": err_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            else: 
                # If no I/O examples, return raw stdout (potentially after JSON parsing if it was a message)
                return {"output_str": stdout_str, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}

        except asyncio.TimeoutError:
            logger.warning(f"Code execution timed out after {self.timeout_seconds} seconds.")
            # Ensure process is terminated if timed out during communicate()
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=1.0) # Brief wait for termination
                except asyncio.TimeoutError:
                    process.kill() # Force kill if terminate doesn't work quickly
                except ProcessLookupError:
                    pass # Process already exited
                except Exception as e_term:
                    logger.error(f"Error during process termination after timeout: {e_term}")
            return {"error": f"TimeoutError: Execution exceeded {self.timeout_seconds} seconds.", "execution_time_ms": self.timeout_seconds * 1000}
        except Exception as e:
            logger.error(f"Error during _execute_code_safely: {e}\n{traceback.format_exc()}")
            return {"error": f"FrameworkExecutionError: An unexpected error occurred in the evaluation framework: {str(e)}", "execution_time_ms": 0.0}
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e_rm:
                    logger.error(f"Error deleting temporary file {tmp_file_path}: {e_rm}")

    async def evaluate_program(self, program: Program) -> Program:
        logger.info(f"Evaluating program: {program.id} for task: {self.task_definition.id}")
        program.status = "evaluating"
        program.errors = [] # Reset errors for this evaluation run
        program.fitness_scores = {"correctness_score": 0.0, "runtime_ms": float('inf')} # Reset scores

        syntax_error = self._check_syntax(program.code)
        if syntax_error:
            program.errors.append(syntax_error)
            program.fitness_scores["correctness_score"] = 0.0 # Explicitly 0 for syntax error
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id} failed syntax check: {syntax_error}")
            return program

        if not self.task_definition.input_output_examples or not self.task_definition.function_name_to_evolve:
            program.errors.append("No input/output examples or function_name_to_evolve provided in task definition for evaluation.")
            program.fitness_scores["correctness_score"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id} cannot be evaluated: Missing I/O examples or function name.")
            return program

        exec_result = await self._execute_code_safely(
            program.code,
            self.task_definition.function_name_to_evolve, 
            self.timeout_seconds
        )
        program.fitness_scores["runtime_ms"] = exec_result.get("execution_time_ms", float('inf'))

        if "error" in exec_result:
            program.errors.append(exec_result["error"])
            raw_stderr = exec_result.get("raw_stderr")
            # Add raw_stderr only if it's substantial and not already part of the main error.
            if raw_stderr and raw_stderr not in exec_result["error"] and len(raw_stderr) > 10 : 
                program.errors.append(f"Raw Stderr (first 500 chars): {raw_stderr[:500]}")
            program.fitness_scores["correctness_score"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id} failed execution/harness: {exec_result['error']}")
            return program

        harness_outputs = exec_result.get("harness_outputs")
        if not isinstance(harness_outputs, list) or len(harness_outputs) != len(self.task_definition.input_output_examples):
            err_msg = (f"Harness output format mismatch. Expected list of {len(self.task_definition.input_output_examples)} items. "
                       f"Got type {type(harness_outputs)} with length {len(harness_outputs) if isinstance(harness_outputs, list) else 'N/A'}. "
                       f"Output (first 200 chars): {str(harness_outputs)[:200]}")
            program.errors.append(err_msg)
            raw_stdout = exec_result.get("raw_stdout")
            if raw_stdout: program.errors.append(f"Raw Stdout (first 500 chars): {raw_stdout[:500]}")
            program.fitness_scores["correctness_score"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id}: {err_msg}")
            return program
        
        correct_count = 0
        total_tests = len(self.task_definition.input_output_examples)

        for i, expected_example in enumerate(self.task_definition.input_output_examples):
            if i >= len(harness_outputs): # Should be caught by length check above, but defensive
                program.errors.append(f"Test case {i+1}: Missing result from harness output.")
                continue

            actual_run_result = harness_outputs[i] 
            
            if not isinstance(actual_run_result, dict) or 'output' not in actual_run_result:
                program.errors.append(f"Test case {i+1}: Invalid result format from harness: {str(actual_run_result)[:100]}")
                continue
            
            if actual_run_result.get("error"):
                input_repr = str(expected_example.get('input', 'N/A'))
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Execution error in harness: {actual_run_result['error']}"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
                continue 

            # ### PR REVIEW: Key conversion for expected output.
            # ### Similar to input, this could be generalized.
            converted_expected_output = self._convert_output_keys_if_needed(expected_example['output'])

            if self._compare_outputs(actual_run_result['output'], converted_expected_output):
                correct_count += 1
            else:
                actual_str = self._safe_output_str(actual_run_result['output'])
                expected_str = self._safe_output_str(converted_expected_output)
                
                input_repr = str(expected_example.get('input', 'N/A'))
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Failed. Expected '{expected_str[:100]}', Got '{actual_str[:100]}'"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
        
        program.fitness_scores["correctness_score"] = (correct_count / total_tests) if total_tests > 0 else 0.0
        
        if not program.errors and program.fitness_scores["correctness_score"] == 1.0 :
            logger.info(f"Program {program.id} evaluated successfully. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms")
        elif not program.errors and program.fitness_scores["correctness_score"] < 1.0: # Some tests failed logically
            logger.info(f"Program {program.id} evaluated. Some tests failed. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms.")
            # Errors list would contain mismatch details
        elif program.errors: 
            logger.warning(f"Program {program.id} evaluated with errors during execution or harness. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms. Errors: {'; '.join(program.errors[:3])}...") # Log first few errors

        program.status = "evaluated"
        return program

    async def execute(self, program: Program) -> Program: 
        return await self.evaluate_program(program)

    def _safe_output_str(self, data: Any) -> str:
        """Helper to convert output to string, handling NaN and large objects."""
        if isinstance(data, float) and math.isnan(data):
            return "float('nan')"
        s = str(data)
        return s

    def _convert_output_keys_if_needed(self, output_data: Any) -> Any:
        """
        Recursively converts string keys that are digits to integers in dictionaries.
        This is useful if the expected output (e.g., from JSON) has stringified integer keys.
        """
        if isinstance(output_data, dict):
            new_dict = {}
            for k, v in output_data.items():
                new_key = k
                if isinstance(k, str) and k.isdigit():
                    try:
                        new_key = int(k)
                    except ValueError: # Should not happen if k.isdigit() is true
                        pass 
                new_dict[new_key] = self._convert_output_keys_if_needed(v) # Recurse for values
            return new_dict
        elif isinstance(output_data, list):
            return [self._convert_output_keys_if_needed(item) for item in output_data]
        return output_data


    def _compare_outputs(self, actual: Any, expected: Any, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        """
        Compares actual and expected outputs, handling floats, NaNs, Infs, dicts, and lists.
        """
        if type(actual) != type(expected) and not (isinstance(actual, (int, float)) and isinstance(expected, (int, float))):
            # Allow comparison between int and float if values are equivalent
            if isinstance(actual, dict) and isinstance(expected, dict): # Both are dicts, proceed
                pass
            elif isinstance(actual, list) and isinstance(expected, list): # Both are lists, proceed
                pass
            else:
                logger.debug(f"Type mismatch: Actual type {type(actual)}, Expected type {type(expected)}")
                return False

        if isinstance(expected, dict):
            if not isinstance(actual, dict): return False # Already caught by type check mostly
            # ### PR REVIEW: The original code converted `actual` here.
            # ### It might be better to assume `actual` is already in its final Python form from the student code.
            # ### `expected` is what comes from JSON and might need key conversion.
            # actual_converted = self._convert_output_keys_if_needed(actual) # Potentially remove this line if actual shouldn't be converted.
            # For now, keeping it assuming student code might also produce string keys if they mimic JSON.

            if set(actual.keys()) != set(expected.keys()):
                logger.debug(f"Output key mismatch: Actual keys: {set(actual.keys())}, Expected keys: {set(expected.keys())}")
                return False
            for key in expected:
                if not self._compare_outputs(actual[key], expected[key], rel_tol, abs_tol): 
                    # logger.debug(f"Output value mismatch for key '{key}': Actual: {actual[key]}, Expected: {expected[key]}") # Can be verbose
                    return False
            return True
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                logger.debug(f"List length mismatch: Actual len {len(actual) if isinstance(actual, list) else 'N/A'}, Expected len {len(expected)}")
                return False
            for i in range(len(expected)):
                if not self._compare_outputs(actual[i], expected[i], rel_tol, abs_tol):
                    # logger.debug(f"List element mismatch at index {i}: Actual: {actual[i]}, Expected: {expected[i]}") # Can be verbose
                    return False
            return True
        elif isinstance(actual, float) and isinstance(expected, float):
            if math.isnan(actual) and math.isnan(expected): return True
            if math.isinf(actual) and math.isinf(expected) and (actual > 0) == (expected > 0): return True
            return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)): # Handles int vs float comparison
             return math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
        else: 
            return actual == expected