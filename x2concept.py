from typing import List
from ask_GPT import ask_GPT
from prompts.x2concept_prompt import x2concept_prompt

class X2Concept:
    def __init__(self, path=None, C_num_return=50) -> None:
        self.C_num_return = C_num_return
        if path is None:
            self.use_api = True
        else:
            self.model = None
        pass

    def get_concept_from_X_list(self, x_list:List[int]) -> List[str]:
        # TODO:
        # return a list of concepts with a number of self.C_num_return
        if self.use_api:
            system_prompt = f"Given a set of numbers, output {self.C_num_return} associated with these numbers. Try to generate diverse concepts without overlapping."
            user_prompt=x2concept_prompt(x_list)
            concepts=ask_GPT(system_prompt,user_prompt)
            concepts=concepts.split(",")
            return concepts
