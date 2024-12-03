

class Concept2Python:
    def __init__(self, device) -> None:
        self.device = device
        pass

    def get_program_from_concept(self, concept: str) -> str:
        # TODO:
        # return a python program that can be used to test the concept
        # 返回的测试函数的名字是test_function
        return f"def test_function(x):\n    return x%2==0\n"
    