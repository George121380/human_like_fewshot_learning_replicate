from ask_GPT import ask_GPT
from prompts.concept2python_prompt import concept2python_prompt

class Concept2Python:
    def __init__(self, device) -> None:
        self.device = device
        pass

    def get_program_from_concept(self, concept: str) -> str:
        system_prompt = f"You need to transfer the concept to a python function that can be used to test the concept. Directly output the python function without any explaination and annotation."
        function_response=ask_GPT(system_prompt,concept2python_prompt(concept))
        # print(function_response) # for debugging
        return function_response
        # return f"def test_function(x):\n    return x%2==0\n"
    