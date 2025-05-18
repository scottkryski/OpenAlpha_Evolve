# Evaluator Agent 
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
            return f"SyntaxError: {e.msg} on line {e.lineno} (offset {e.offset}). Problematic line: '{e.text.strip() if e.text else ''}'"
        except Exception as e:
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
        process = None # Initialize process to None
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as fp:
                fp.write("# -*- coding: utf-8 -*-\n")
                fp.write("import sys\nimport json\nimport math\n")

                if self.task_definition.allowed_imports:
                    for imp_statement in self.task_definition.allowed_imports:
                        fp.write(f"import {imp_statement}\n") 
                
                fp.write("\n# --- Helper function to convert graph keys to int (if applicable for the task) ---\n")
                fp.write(
                    "def _convert_graph_keys(graph_obj):\n"
                    "    if not isinstance(graph_obj, dict):\n"
                    "        return graph_obj\n"
                    "    new_graph = {}\n"
                    "    for k, v_obj in graph_obj.items():\n"
                    "        try:\n"
                    "            int_k = int(k)\n"
                    "        except (ValueError, TypeError):\n" 
                    "            int_k = k \n"
                    "        if isinstance(v_obj, dict):\n"
                    "            new_v_inner_dict = {}\n"
                    "            for nk, nv in v_obj.items():\n"
                    "                try:\n"
                    "                    int_nk = int(nk)\n"
                    "                except (ValueError, TypeError):\n"
                    "                    int_nk = nk\n"
                    "                new_v_inner_dict[int_nk] = nv\n"
                    "            new_graph[int_k] = new_v_inner_dict\n"
                    "        else:\n"
                    "            new_graph[int_k] = v_obj \n"
                    "    return new_graph\n\n"
                )

                fp.write("\n# --- Generated Code Start ---\n")
                fp.write(code_to_run)
                fp.write("\n# --- Generated Code End ---\n\n")
                
                if self.task_definition.input_output_examples and function_name_from_task:
                    test_cases_json_str = json.dumps(self.task_definition.input_output_examples)
                    test_cases_py_str = test_cases_json_str.replace("Infinity", "float('inf')")
                    test_cases_py_str = test_cases_py_str.replace("-Infinity", "float('-inf')")
                    test_cases_py_str = test_cases_py_str.replace("NaN", "float('nan')") 
                    
                    fp.write(f"test_cases_from_task = {test_cases_py_str}\n")
                    fp.write(f"evaluation_results = []\n")
                    fp.write(f"for case_idx, case_data in enumerate(test_cases_from_task):\n")
                    fp.write(f"    input_args = case_data.get('input')\n")
                    fp.write(f"    actual_output = None\n")
                    fp.write(f"    error_occurred = None\n")
                    fp.write(f"    try:\n")
                    fp.write(f"        func_args = []\n")
                    fp.write(f"        func_kwargs = {{}}\n")
                    fp.write(f"        if isinstance(input_args, dict):\n")
                    fp.write(f"            processed_input_args = {{}}\n")
                    fp.write(f"            for arg_name, arg_val in input_args.items():\n")
                    fp.write(f"                if arg_name == 'graph':\n") 
                    fp.write(f"                    processed_input_args[arg_name] = _convert_graph_keys(arg_val)\n")
                    fp.write(f"                else:\n")
                    fp.write(f"                    processed_input_args[arg_name] = arg_val\n")
                    fp.write(f"            func_kwargs = processed_input_args\n")
                    fp.write(f"            actual_output = {function_name_from_task}(**func_kwargs)\n")
                    fp.write(f"        elif isinstance(input_args, list):\n")
                    fp.write(f"            # This part might need more robust logic for generic positional args if graphs are passed positionally\n")
                    fp.write(f"            # For now, assuming if input_args is a list, it's for non-graph args or graph is handled by name\n")
                    fp.write(f"            func_args = input_args # Or a processed version if _convert_graph_keys needs to apply positionally\n")
                    fp.write(f"            actual_output = {function_name_from_task}(*func_args)\n")
                    fp.write(f"        else:\n") 
                    fp.write(f"            # Single argument case, potentially needs graph conversion if it's the graph itself\n")
                    fp.write(f"            # This path is less likely for the Dijkstra example. For safety, one might check type.\n")
                    fp.write(f"            single_arg_processed = _convert_graph_keys(input_args) if isinstance(input_args, dict) else input_args\n")
                    fp.write(f"            actual_output = {function_name_from_task}(single_arg_processed)\n")
                    fp.write(f"    except Exception as e_exec:\n")
                    fp.write(f"        actual_output = None\n")
                    fp.write(f"        error_type_name = type(e_exec).__name__\n")
                    fp.write(f"        error_message_raw = str(e_exec)\n")
                    fp.write(f"        sanitized_error_message = error_message_raw.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\\n', ' ').replace('\\r', '')\n")
                    fp.write(f"        error_occurred = f'EXECUTION_ERROR_CASE_{{case_idx}}: {{error_type_name}}: {{sanitized_error_message}}'\n") 
                    fp.write(f"    evaluation_results.append({{'output': actual_output, 'error': error_occurred}})\n")
                    fp.write(f"print(json.dumps(evaluation_results, allow_nan=True))\n")
                else:
                    fp.write("print(json.dumps({'message': 'No I/O examples or function to evaluate in harness.'}))\n")
                tmp_file_path = fp.name
            
            start_time = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "utf8", tmp_file_path, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if stdout_str: logger.debug(f"Raw stdout from script execution (first 1000 chars): {stdout_str[:1000]}")
            if stderr_str: logger.debug(f"Raw stderr from script execution (first 1000 chars): {stderr_str[:1000]}")

            if process.returncode != 0:
                error_msg = f"Execution failed with return code {process.returncode}. Stderr: {stderr_str[:500]}"
                logger.warning(error_msg)
                return {"error": error_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            
            if self.task_definition.input_output_examples and function_name_from_task:
                try:
                    harness_outputs = json.loads(stdout_str) 
                    return {"harness_outputs": harness_outputs, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
                except json.JSONDecodeError as je:
                    err_msg = f"JSONDecodeError: Output from harness was not valid JSON. Error: {je}. Stdout (first 500 chars): {stdout_str[:500]}"
                    logger.warning(err_msg)
                    return {"error": err_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            else: 
                return {"output_str": stdout_str, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}

        except asyncio.TimeoutError:
            logger.warning(f"Code execution timed out after {self.timeout_seconds} seconds.")
            if process and process.returncode is None: # Ensure process is cleaned up
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=1.0) 
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process.pid} did not terminate gracefully after timeout, killing.")
                    process.kill()
                except ProcessLookupError: # Process already exited
                    logger.debug(f"Process {process.pid} already exited by the time of timeout cleanup.")
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
        program.errors = [] 
        program.fitness_scores = {"correctness_score": 0.0, "runtime_ms": float('inf')}

        syntax_error = self._check_syntax(program.code)
        if syntax_error:
            program.errors.append(syntax_error)
            program.fitness_scores["correctness_score"] = 0.0
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
            if i >= len(harness_outputs):
                program.errors.append(f"Test case {i+1}: Missing result from harness output.")
                continue

            actual_run_result = harness_outputs[i] 
            
            if not isinstance(actual_run_result, dict) or 'output' not in actual_run_result:
                program.errors.append(f"Test case {i+1}: Invalid result format from harness: {str(actual_run_result)[:100]}")
                continue
            
            actual_output_from_harness = actual_run_result['output'] # This will have string keys if it's a dict from JSON

            if actual_run_result.get("error"):
                input_repr = str(expected_example.get('input', 'N/A'))
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Execution error in harness: {actual_run_result['error']}"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
                continue 

            # `expected_example['output']` has int keys from TaskDefinition in main.py
            # `actual_output_from_harness` has string keys if it's a dict, due to json.loads()
            # The _compare_outputs function will handle converting actual_output_from_harness's keys.
            if self._compare_outputs(actual_output_from_harness, expected_example['output']):
                correct_count += 1
            else:
                # For logging, convert both to string representations that are consistent
                actual_str_repr = self._safe_output_str(self._convert_output_keys_if_needed(actual_output_from_harness))
                expected_str_repr = self._safe_output_str(expected_example['output']) # expected is already in Python types
                
                input_repr = str(expected_example.get('input', 'N/A'))
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Failed. Expected '{expected_str_repr[:100]}', Got '{actual_str_repr[:100]}'"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
        
        program.fitness_scores["correctness_score"] = (correct_count / total_tests) if total_tests > 0 else 0.0
        
        if not program.errors and program.fitness_scores["correctness_score"] == 1.0 :
            logger.info(f"Program {program.id} evaluated successfully. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms")
        elif not program.errors and program.fitness_scores["correctness_score"] < 1.0:
            logger.info(f"Program {program.id} evaluated. Some tests failed. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms.")
        elif program.errors: 
            logger.warning(f"Program {program.id} evaluated with errors. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms. Errors: {'; '.join(program.errors[:3])}...")

        program.status = "evaluated"
        return program

    async def execute(self, program: Program) -> Program: 
        return await self.evaluate_program(program)

    def _safe_output_str(self, data: Any) -> str:
        if isinstance(data, float):
            if math.isnan(data): return "float('nan')"
            if math.isinf(data): return "float('inf')" if data > 0 else "float('-inf')"
        # For dictionaries, ensure a somewhat consistent string representation for logging
        if isinstance(data, dict):
            try:
                # Sort keys for more consistent logging, converting keys to str for sorting
                return str(dict(sorted(data.items(), key=lambda item: str(item[0]))))
            except TypeError: # Fallback if keys are not sortable
                return str(data) 
        return str(data)


    def _convert_output_keys_if_needed(self, output_data: Any) -> Any:
        if isinstance(output_data, dict):
            new_dict = {}
            for k, v in output_data.items():
                new_key = k
                if isinstance(k, str) and k.isdigit():
                    try:
                        new_key = int(k)
                    except ValueError: 
                        pass 
                new_dict[new_key] = self._convert_output_keys_if_needed(v)
            return new_dict
        elif isinstance(output_data, list):
            return [self._convert_output_keys_if_needed(item) for item in output_data]
        return output_data


    def _compare_outputs(self, actual: Any, expected: Any, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        # `expected` comes directly from TaskDefinition (e.g., with int keys for dicts).
        # `actual` comes from JSON parsing of student code output (so dict keys will be strings if originally numbers).

        # Convert `actual` dict keys if necessary before any comparison
        if isinstance(actual, dict) and isinstance(expected, dict):
            actual_to_compare = self._convert_output_keys_if_needed(actual)
        else:
            actual_to_compare = actual

        # Type check after potential conversion for dicts
        if type(actual_to_compare) != type(expected):
            # Allow int/float comparison if values are numerically close
            if isinstance(actual_to_compare, (int, float)) and isinstance(expected, (int, float)):
                pass # Handled by float comparison logic later
            else:
                logger.debug(f"Type mismatch after potential conversion: Actual type {type(actual_to_compare)}, Expected type {type(expected)}. Actual: '{str(actual_to_compare)[:50]}', Expected: '{str(expected)[:50]}'")
                return False

        if isinstance(expected, dict):
            # `actual_to_compare` should also be a dict here due to earlier checks/conversions
            if not isinstance(actual_to_compare, dict): # Should be redundant but safe
                logger.debug(f"Logic error: expected dict, actual_to_compare is {type(actual_to_compare)}")
                return False

            if set(actual_to_compare.keys()) != set(expected.keys()):
                logger.debug(f"Output key mismatch: Actual keys (post-conversion): {set(actual_to_compare.keys())}, Expected keys: {set(expected.keys())}")
                return False
            
            for key_from_expected in expected:
                if not self._compare_outputs(actual_to_compare[key_from_expected], expected[key_from_expected], rel_tol, abs_tol): 
                    # Detailed logging moved to recursive calls to avoid repetition here
                    return False
            return True
        elif isinstance(expected, list):
            if not isinstance(actual_to_compare, list) or len(actual_to_compare) != len(expected):
                logger.debug(f"List length mismatch: Actual len {len(actual_to_compare) if isinstance(actual_to_compare, list) else 'N/A'}, Expected len {len(expected)}")
                return False
            for i in range(len(expected)):
                if not self._compare_outputs(actual_to_compare[i], expected[i], rel_tol, abs_tol):
                    return False
            return True
        elif isinstance(actual_to_compare, float) and isinstance(expected, float):
            if math.isnan(actual_to_compare) and math.isnan(expected): return True
            if math.isinf(actual_to_compare) and math.isinf(expected) and (actual_to_compare > 0) == (expected > 0): return True
            return math.isclose(actual_to_compare, expected, rel_tol=rel_tol, abs_tol=abs_tol)
        elif isinstance(actual_to_compare, (int, float)) and isinstance(expected, (int, float)):
             return math.isclose(float(actual_to_compare), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
        else: 
            are_equal = actual_to_compare == expected
            if not are_equal:
                logger.debug(f"Basic equality failed: Actual: '{str(actual_to_compare)[:50]}', Expected: '{str(expected)[:50]}'")
            return are_equal