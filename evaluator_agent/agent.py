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
            return f"SyntaxError: {e.msg} on line {e.lineno}"
        except Exception as e:
            logger.warning(f"Unexpected error during syntax check: {e}")
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
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as fp:
                fp.write("# -*- coding: utf-8 -*-\n")
                fp.write("import sys\nimport json\nimport math\n")

                if self.task_definition.allowed_imports:
                    for imp in self.task_definition.allowed_imports:
                        if imp and imp.isidentifier(): # Basic check for valid import name
                            fp.write(f"import {imp}\n")
                        elif imp and '.' in imp and all(part.isidentifier() for part in imp.split('.')): # for 'module.submodule'
                            fp.write(f"import {imp}\n")
                
                fp.write("\n# --- Helper function to convert graph keys to int ---\n")
                fp.write(
                    "def _convert_graph_keys(graph_obj):\n"
                    "    if not isinstance(graph_obj, dict):\n"
                    "        return graph_obj\n"
                    "    new_graph = {}\n"
                    "    for k, v_obj in graph_obj.items():\n"
                    "        try:\n"
                    "            int_k = int(k)\n"
                    "        except (ValueError, TypeError):\n" # TypeError if k is not string-like
                    "            int_k = k # Keep original if not convertible to int\n"
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
                    "            new_graph[int_k] = v_obj # Should not happen for graph structure's inner part\n"
                    "    return new_graph\n\n"
                )

                fp.write("\n# --- Generated Code Start ---\n")
                fp.write(code_to_run)
                fp.write("\n# --- Generated Code End ---\n\n")
                
                if self.task_definition.input_output_examples:
                    test_cases_json_str = json.dumps(self.task_definition.input_output_examples)
                    test_cases_py_str = test_cases_json_str.replace(": Infinity", ": float('inf')")
                    test_cases_py_str = test_cases_py_str.replace(":Infinity", ": float('inf')") 
                    test_cases_py_str = test_cases_py_str.replace(": -Infinity", ": float('-inf')")
                    test_cases_py_str = test_cases_py_str.replace(":-Infinity", ": float('-inf')") 
                    test_cases_py_str = test_cases_py_str.replace(": NaN", ": float('nan')")
                    test_cases_py_str = test_cases_py_str.replace(":NaN", ": float('nan')") 

                    fp.write(f"test_cases_from_task = {test_cases_py_str}\n")
                    fp.write(f"evaluation_results = []\n")
                    fp.write(f"for case_idx, case_data in enumerate(test_cases_from_task):\n")
                    fp.write(f"    input_dict = case_data.get('input')\n")
                    fp.write(f"    actual_output = None\n")
                    fp.write(f"    error_occurred = None\n")
                    fp.write(f"    try:\n")
                    fp.write(f"        str_keyed_graph_arg = input_dict.get('graph')\n")
                    fp.write(f"        graph_arg = _convert_graph_keys(str_keyed_graph_arg)\n")
                    fp.write(f"        source_node_arg = input_dict.get('source_node')\n")
                    fp.write(f"        actual_output = {function_name_from_task}(graph_arg, source_node_arg)\n")
                    fp.write(f"    except Exception as e_exec:\n")
                    fp.write(f"        actual_output = None\n")
                    # Corrected and safer string sanitization for error messages
                    fp.write(f"        error_type_name = type(e_exec).__name__\n")
                    fp.write(f"        error_message_raw = str(e_exec)\n")
                    fp.write(f"        sanitized_error_message = error_message_raw.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\\n', ' ')\n") # Escape backslashes, then quotes, then newlines
                    fp.write(f"        error_occurred = f'EXECUTION_ERROR_CASE_{{case_idx}}: {{error_type_name}}: {{sanitized_error_message}}'\n") 
                    fp.write(f"    evaluation_results.append({{'output': actual_output, 'error': error_occurred}})\n")
                    fp.write(f"print(json.dumps(evaluation_results, allow_nan=True))\n") 
                tmp_file_path = fp.name
            
            start_time = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "utf8", tmp_file_path, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if stdout_str:
                logger.debug(f"Raw stdout from script execution: {stdout_str[:1000]}")
            if stderr_str:
                logger.debug(f"Raw stderr from script execution: {stderr_str[:1000]}")


            if process.returncode != 0:
                error_msg = f"Execution failed with return code {process.returncode}. Stderr: {stderr_str[:500]}"
                logger.warning(error_msg)
                return {"error": error_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            
            if self.task_definition.input_output_examples:
                try:
                    harness_outputs = json.loads(stdout_str) 
                    return {"harness_outputs": harness_outputs, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
                except json.JSONDecodeError as je:
                    err_msg = f"JSONDecodeError: Output from harness was not valid JSON. Error: {je}. Stdout was: {stdout_str[:500]}"
                    logger.warning(err_msg)
                    return {"error": err_msg, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}
            else: 
                return {"output_str": stdout_str, "execution_time_ms": execution_time_ms, "raw_stdout": stdout_str, "raw_stderr": stderr_str}

        except asyncio.TimeoutError:
            logger.warning(f"Code execution timed out after {self.timeout_seconds} seconds.")
            return {"error": f"TimeoutError: Execution exceeded {self.timeout_seconds} seconds.", "execution_time_ms": self.timeout_seconds * 1000}
        except Exception as e:
            logger.error(f"Error during _execute_code_safely: {e}\n{traceback.format_exc()}")
            return {"error": f"FrameworkExecutionError: {e}", "execution_time_ms": 0.0}
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

        if not self.task_definition.input_output_examples:
            program.errors.append("No input/output examples provided in task definition for evaluation.")
            program.fitness_scores["correctness_score"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id} cannot be evaluated: No input/output examples.")
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
            if raw_stderr and raw_stderr not in exec_result["error"]: 
                program.errors.append(f"Raw Stderr: {raw_stderr[:500]}")
            program.fitness_scores["correctness_score"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Program {program.id} failed execution/harness: {exec_result['error']}")
            return program

        harness_outputs = exec_result.get("harness_outputs")
        if not isinstance(harness_outputs, list) or len(harness_outputs) != len(self.task_definition.input_output_examples):
            err_msg = f"Harness output format mismatch. Expected list of {len(self.task_definition.input_output_examples)} items. Got: {str(harness_outputs)[:200]}"
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
            actual_run_result = harness_outputs[i] 
            
            if actual_run_result.get("error"):
                input_repr = str(expected_example['input'])
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Execution error in harness: {actual_run_result['error']}"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
                continue 

            converted_expected_output = self._convert_output_keys_if_needed(expected_example['output'])

            if self._compare_outputs(actual_run_result['output'], converted_expected_output):
                correct_count += 1
            else:
                actual_str = "float('nan')" if isinstance(actual_run_result['output'], float) and math.isnan(actual_run_result['output']) else str(actual_run_result['output'])
                expected_str = "float('nan')" if isinstance(converted_expected_output, float) and math.isnan(converted_expected_output) else str(converted_expected_output)
                
                input_repr = str(expected_example['input'])
                input_log_repr = input_repr[:50] + ('...' if len(input_repr) > 50 else '')
                err_msg = f"Test case {i+1} (Input: {input_log_repr}): Failed. Expected '{expected_str[:100]}', Got '{actual_str[:100]}'"
                program.errors.append(err_msg)
                logger.debug(f"Program {program.id}: {err_msg}")
        
        program.fitness_scores["correctness_score"] = (correct_count / total_tests) if total_tests > 0 else 0.0
        
        if not program.errors and program.fitness_scores["correctness_score"] == 1.0 :
            logger.info(f"Program {program.id} evaluated successfully. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms")
        elif not program.errors and program.fitness_scores["correctness_score"] < 1.0:
            logger.info(f"Program {program.id} evaluated. Some tests failed. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms.")
            if any("Failed. Expected" in e for e in program.errors):
                logger.info(f"Program {program.id} specific test failures: {[e for e in program.errors if 'Failed. Expected' in e]}")
        elif program.errors: 
            logger.warning(f"Program {program.id} evaluated with errors. Correctness: {program.fitness_scores['correctness_score']:.2f}, Runtime: {program.fitness_scores['runtime_ms']:.2f}ms. Errors: {program.errors}")

        program.status = "evaluated"
        return program

    async def execute(self, program: Program) -> Program: 
        return await self.evaluate_program(program)

    def _convert_output_keys_if_needed(self, output_data: Any) -> Any:
        if isinstance(output_data, dict):
            all_keys_are_str_int = True
            if not output_data: 
                return output_data
                
            for k in output_data.keys():
                if not (isinstance(k, str) and k.isdigit()): # Ensure k is a string before k.isdigit()
                    all_keys_are_str_int = False
                    break
            
            if all_keys_are_str_int:
                new_output = {}
                for k, v in output_data.items():
                    new_output[int(k)] = v 
                return new_output
        return output_data


    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            actual_converted = self._convert_output_keys_if_needed(actual)
            
            if set(actual_converted.keys()) != set(expected.keys()):
                logger.debug(f"Output key mismatch: Actual keys: {set(actual_converted.keys())}, Expected keys: {set(expected.keys())}")
                return False
            
            for key in expected:
                if not self._compare_outputs(actual_converted[key], expected[key]): 
                    logger.debug(f"Output value mismatch for key '{key}': Actual: {actual_converted[key]}, Expected: {expected[key]}")
                    return False
            return True
        elif isinstance(actual, float) and isinstance(expected, float):
            if math.isnan(actual) and math.isnan(expected):
                return True
            if math.isinf(actual) and math.isinf(expected) and (actual > 0) == (expected > 0):
                return True
            return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=0.0)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                return False
            for i in range(len(expected)):
                if not self._compare_outputs(actual[i], expected[i]):
                    return False
            return True
        else: 
            return actual == expected
