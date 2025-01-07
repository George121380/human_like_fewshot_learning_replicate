from ask_GPT import ask_GPT
from prompts.concept2python_prompt import concept2python_prompt

class Concept2Python:
    def __init__(self, device) -> None:
        self.device = device
        self.cache = {}
        pass

    def get_program_from_concept(self, concept: str) -> str:
        if concept in self.cache:
            return self.cache[concept]
        system_prompt, user_prompt = concept2python_prompt(concept)
        function_response=ask_GPT(system_prompt, user_prompt)
        # print(function_response) # for debugging
        self.cache[concept]=function_response
        return function_response
        # return f"def test_function(x):\n    return x%2==0\n"
    