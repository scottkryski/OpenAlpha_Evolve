import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from config import settings

# Import concrete agent implementations
from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent
from evaluator_agent.agent import EvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent # Using InMemory for now
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition # Store the task definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        # Pass task_definition to EvaluatorAgent
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        self.database: DatabaseAgentInterface = InMemoryDatabaseAgent() # Can be swapped with other DB agents
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()

        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        self.num_parents_to_select = settings.POPULATION_SIZE // 2 # Example: select half the population size as parents
        if self.num_parents_to_select == 0 and self.population_size > 0: # Ensure at least one parent if pop exists
             self.num_parents_to_select = 1


    async def initialize_population(self) -> List[Program]:
        logger.info(f"Initializing population for task: {self.task_definition.id}")
        initial_population = []
        generation_tasks = []

        for i in range(self.population_size):
            program_id = f"{self.task_definition.id}_gen0_prog{i}"
            # Create task for code generation
            generation_tasks.append(self._generate_initial_program(program_id, 0))
        
        # Gather all generated programs
        generated_program_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        for result in generated_program_results:
            if isinstance(result, Exception):
                logger.error(f"Error during initial program generation: {result}", exc_info=result)
                # Optionally create a placeholder failed program or skip
            elif result:
                initial_population.append(result)
                await self.database.save_program(result) # Save to DB
            
        logger.info(f"Initialized population with {len(initial_population)} programs after generation.")
        return initial_population

    async def _generate_initial_program(self, program_id: str, generation: int) -> Optional[Program]:
        logger.debug(f"Generating initial program with id {program_id}")
        initial_prompt = self.prompt_designer.design_initial_prompt()
        # Temperature can be higher for initial generation for diversity
        generated_code = await self.code_generator.generate_code(initial_prompt, temperature=0.8) 
        
        if not generated_code or generated_code.strip().startswith("# Error:") or "Error:" in generated_code:
            logger.warning(f"Failed to generate valid initial code for {program_id}. LLM Output: {generated_code}")
            # Create a program with error status or return None
            return Program(
                id=program_id,
                code="# Generation failed\n" + str(generated_code),
                generation=generation,
                status="failed_generation",
                errors=[f"LLM failed to produce valid initial code: {generated_code}"]
            )

        return Program(
            id=program_id,
            code=generated_code,
            generation=generation,
            status="unevaluated"
        )

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs = []
        
        # Filter out programs that don't need evaluation (e.g., failed generation)
        programs_to_evaluate = [prog for prog in population if prog.status == "unevaluated"]
        logger.info(f"Found {len(programs_to_evaluate)} programs needing evaluation.")

        evaluation_tasks = [self.evaluator.evaluate_program(prog) for prog in programs_to_evaluate]
        
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        processed_program_ids = set()

        for i, result in enumerate(results):
            # original_program = programs_to_evaluate[i] # This assumes order is maintained and result corresponds to prog
            # It's safer to rely on the 'result' being the Program object itself
            
            if isinstance(result, Program): # EvaluatorAgent returns the Program object
                evaluated_program = result
                evaluated_programs.append(evaluated_program)
                await self.database.save_program(evaluated_program) # Update DB with evaluation results
                processed_program_ids.add(evaluated_program.id)
            elif isinstance(result, Exception):
                # This case should ideally be handled within evaluate_program to update program status/errors
                # If an exception bubbles up here, it means evaluate_program itself failed unexpectedly
                original_program_for_error = programs_to_evaluate[i] # Fallback to get ID
                logger.error(f"Critical error during evaluation task for program {original_program_for_error.id}: {result}", exc_info=result)
                original_program_for_error.status = "failed_evaluation_framework"
                original_program_for_error.errors.append(f"Framework error: {str(result)}")
                evaluated_programs.append(original_program_for_error) # Add it back with error status
                await self.database.save_program(original_program_for_error)
                processed_program_ids.add(original_program_for_error.id)
            else: # Should not happen
                logger.error(f"Unexpected result type from evaluator: {type(result)} for program {programs_to_evaluate[i].id}")


        # Add back programs that were not evaluated (e.g. failed_generation)
        for prog in population:
            if prog.id not in processed_program_ids:
                evaluated_programs.append(prog)
            
        logger.info(f"Finished evaluating population. {len(evaluated_programs)} programs processed/accounted for.")
        return evaluated_programs


    async def manage_evolutionary_cycle(self) -> Optional[Program]: # Return single best program or None
        logger.info(f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}...")
        current_population = await self.initialize_population()
        
        if not current_population:
            logger.warning("Initialization yielded no programs. Ending evolution.")
            return None
            
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            # Filter out programs that are not suitable for parenting (e.g., completely failed)
            viable_parents_pool = [p for p in current_population if p.status == "evaluated"]
            if not viable_parents_pool:
                logger.warning(f"Generation {gen}: No viable (evaluated) programs in current population to select parents from. Ending evolution early.")
                break

            # 1. Selection
            parents = self.selection_controller.select_parents(viable_parents_pool, self.num_parents_to_select)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected from viable pool. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")

            # 2. Crossover (simplified: not implemented, LLM mutation is primary)
            # 3. Mutation (generating offspring)
            offspring_population: List[Program] = []
            
            # Determine how many offspring to generate. Aim to fill up to population_size.
            # If elitism is strong, fewer new offspring might be needed.
            # For simplicity, let's aim to generate enough offspring to consider replacing non-elites.
            num_offspring_to_generate = self.population_size - settings.ELITISM_COUNT 
            if num_offspring_to_generate <= 0: num_offspring_to_generate = 1 # Ensure at least one attempt if pop_size is small
            
            # Distribute offspring generation among selected parents
            offspring_tasks = []
            for i in range(num_offspring_to_generate):
                parent_for_offspring = parents[i % len(parents)] # Cycle through parents
                child_id = f"{self.task_definition.id}_gen{gen}_child{i}"
                offspring_tasks.append(self.generate_offspring(parent_for_offspring, gen, child_id))
            
            generated_offspring_results = await asyncio.gather(*offspring_tasks, return_exceptions=True)

            for result in generated_offspring_results:
                if isinstance(result, Exception):
                    logger.error(f"Error generating offspring: {result}", exc_info=result)
                elif result: # result is a Program object or None
                    offspring_population.append(result)
                    await self.database.save_program(result) # Save to DB
            
            logger.info(f"Generation {gen}: Generated {len(offspring_population)} offspring.")
            if not offspring_population:
                logger.warning(f"Generation {gen}: No offspring successfully generated. Population may stagnate.")
                # If no offspring, the next population will be selected only from current_population

            # 4. Evaluation of Offspring
            if offspring_population:
                offspring_population = await self.evaluate_population(offspring_population)
            else: # ensure offspring_population is an empty list if no offspring were generated.
                offspring_population = []


            # 5. Survivor Selection
            # Combine current population with newly evaluated offspring for survivor selection
            current_population = self.selection_controller.select_survivors(current_population, offspring_population, self.population_size)
            logger.info(f"Generation {gen}: New population size after survival: {len(current_population)}.")

            if not current_population:
                logger.warning(f"Generation {gen}: Population is empty after survivor selection. Ending evolution.")
                break

            # Log best program of this generation
            # Sort by correctness (desc), then by runtime (asc, so -runtime desc)
            best_program_this_gen_list = sorted(
                [p for p in current_population if p.status == "evaluated"], # Only consider evaluated for "best"
                key=lambda p: (
                    p.fitness_scores.get("correctness_score", -1.0), 
                    -p.fitness_scores.get("runtime_ms", float('inf'))
                ), 
                reverse=True
            )
            if best_program_this_gen_list:
                 logger.info(f"Generation {gen}: Best program in pop: ID={best_program_this_gen_list[0].id}, Fitness={best_program_this_gen_list[0].fitness_scores}")
                 # Check for termination condition (e.g., perfect score)
                 if best_program_this_gen_list[0].fitness_scores.get("correctness_score", 0.0) == 1.0:
                     logger.info(f"Perfect score achieved in generation {gen}! Ending evolution.")
                     # Await self.database.get_best_programs returns a list
                     final_best_list = await self.database.get_best_programs(
                         task_id=self.task_definition.id, # Pass task_id here
                         limit=1, 
                         objective="correctness_score", 
                         sort_order="desc"
                     )
                     return final_best_list[0] if final_best_list else None
            else:
                logger.warning(f"Generation {gen}: No evaluated programs in current population to determine best.")


        logger.info("Evolutionary cycle completed.")
        # Get overall best from the database
        final_best_list = await self.database.get_best_programs(
            task_id=self.task_definition.id, # Pass task_id here as well
            limit=1, 
            objective="correctness_score", 
            sort_order="desc"
        )
        if final_best_list:
            best_program = final_best_list[0]
            logger.info(f"Overall Best Program: {best_program.id}, Code:\n{best_program.code}\nFitness: {best_program.fitness_scores}")
            return best_program
        else:
            logger.info("No best program found at the end of evolution.")
            return None
    
    async def generate_offspring(self, parent: Program, generation_num: int, child_id:str) -> Optional[Program]:
        logger.debug(f"Generating offspring from parent {parent.id} for generation {generation_num} (child_id: {child_id})")
        
        mutation_prompt = ""
        # Try to fix bugs if parent has errors and low correctness (e.g. correctness 0)
        # Ensure fitness_scores exists before trying to access it
        parent_correctness = parent.fitness_scores.get("correctness_score", 0.0) if parent.fitness_scores else 0.0

        if parent.errors and parent_correctness < 0.1: # More threshold for bug-fixing attempt
            first_error = parent.errors[0] if parent.errors else "Unknown error"
            # execution_details = next((e for e in parent.errors if "Raw Stderr" in e or "Raw Stdout" in e), None)
            # execution_details = "\n".join(parent.errors[1:]) if len(parent.errors) > 1 else None
            mutation_prompt = self.prompt_designer.design_bug_fix_prompt(
                program=parent, 
                error_message=first_error, 
                execution_output= "Review previous errors: " + "; ".join(parent.errors)
            )
            logger.info(f"Attempting bug fix for parent {parent.id} (Correctness: {parent_correctness:.2f})")
        else:
            # Pass recent eval feedback if available
            feedback = {"errors": parent.errors, "fitness_scores": parent.fitness_scores}
            mutation_prompt = self.prompt_designer.design_mutation_prompt(program=parent, evaluation_feedback=feedback)
            logger.info(f"Attempting mutation for parent {parent.id} (Correctness: {parent_correctness:.2f})")
        
        # Use a slightly lower temperature for mutation/bugfix than initial generation
        generated_code = await self.code_generator.generate_code(mutation_prompt, temperature=0.7)

        if not generated_code or generated_code.strip().startswith("# Error:") or "Error:" in generated_code: # Basic check for LLM error indication
            logger.warning(f"Failed to generate valid code for offspring of {parent.id}. LLM Output: {generated_code[:200]}")
            return Program(
                id=child_id,
                code=f"# Offspring generation failed for parent {parent.id}\n" + str(generated_code),
                generation=generation_num,
                parent_id=parent.id,
                status="failed_generation",
                errors=[f"LLM failed to produce valid offspring code: {generated_code[:200]}"]
            )

        return Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_id=parent.id,
            status="unevaluated"
        )

    async def execute(self) -> Optional[Program]: # Return type is single Program or None
        return await self.manage_evolutionary_cycle()

# Example Usage (for testing this agent directly):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Define a simple task
    sample_task = TaskDefinition(
        id="sum_list_task_tm_001",
        description="Write a Python function called `solve(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        function_name_to_evolve="solve", # Explicitly define the function name for the harness
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [-1, 0, 1], "output": 0},
            {"input": [10, 20, 30, 40, 50], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness_score", "goal": "maximize"},
        initial_code_prompt = "Please provide a Python function `solve(numbers)` that sums a list of integers. Handle empty lists by returning 0.",
        allowed_imports=[] # No special imports needed for this simple sum
    )
    
    task_manager = TaskManagerAgent(task_definition=sample_task) 

    # Reduce generations/population for quicker test
    task_manager.num_generations = 3 
    task_manager.population_size = 5 
    task_manager.num_parents_to_select = 2  
    settings.ELITISM_COUNT = 1 # For testing

    async def run_task():
        # Ensure LLM Provider (e.g. GEMINI_API_KEY or Ollama endpoint) is in your .env file or environment
        try:
            best_program = await task_manager.execute() # Call execute on the instance
            if best_program:
                print(f"\n*** Evolution Complete! Best program found: ***")
                print(f"ID: {best_program.id}")
                print(f"Generation: {best_program.generation}")
                print(f"Fitness: {best_program.fitness_scores}")
                print(f"Code:\n{best_program.code}")
            else:
                print("\n*** Evolution Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the task management cycle.", exc_info=True)

    asyncio.run(run_task())