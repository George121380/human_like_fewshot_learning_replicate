from ask_GPT import ask_GPT

class Concept2Python:
    def __init__(self, device) -> None:
        self.device = device
        pass

    def get_program_from_concept(self, concept: str) -> str:
        # TODO:
        system_prompt = f"You need to transfer the concept to a python function that can be used to test the concept. Directly output the python function without any explaination and annotation."

        user_prompt = f"""
## Concept
The concept is:{concept}

## Example:
# concept: even number
python function:
def test_function(x):
    return x%2==0

# concept: numbers larger than 5
python function:
def test_function(x):
    return x>5

# concept: Perfect squares less than 100
python function:
def test_funct(x):
    if n < 0:
        return False
    root = int(n**0.5)
    return root**2 == n and n < 100
        """
        function_response=ask_GPT(system_prompt,user_prompt)
        # print(function_response)
        return function_response
        # return f"def test_function(x):\n    return x%2==0\n"
    