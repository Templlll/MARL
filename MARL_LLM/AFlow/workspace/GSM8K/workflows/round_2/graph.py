from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Changed instruction to a math solver prompt to enforce step-by-step reasoning
        solution = await self.custom(input=problem, instruction=prompt_custom.MATH_SOLVER_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
