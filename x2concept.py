from typing import List
from ask_GPT import ask_GPT
from prompts.x2concept_prompt import x2concept_prompt

class X2Concept:
    def __init__(self, path=None, C_num_return=50, fixed_return=None) -> None:
        self.C_num_return = C_num_return
        self.cache = {}
        if path is None:
            self.use_api = True
        else:
            self.model = None

        if fixed_return is not None:
            self.fixed = True
            self.concepts = fixed_return
        else:
            self.fixed = False
        pass

    def get_concept_from_X_list(self, x_list:List[int]) -> List[str]:
        # TODO:
        # return a list of concepts with a number of self.C_num_return
        if self.fixed:
            return self.concepts
        if self.use_api:
            x_list_str = ",".join([str(x) for x in x_list])
            if x_list_str in self.cache:
                concepts=self.cache[x_list_str]
                return concepts
            
            system_prompt, user_prompt=x2concept_prompt(x_list, self.C_num_return)
            concepts=ask_GPT(system_prompt,user_prompt)
            concepts=concepts.split(",")
            self.cache[x_list_str]=concepts
            return concepts
