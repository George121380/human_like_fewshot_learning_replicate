from typing import List

class X2Concept:
    def __init__(self, path=None, C_num_return=50) -> None:
        self.C_num_return = C_num_return
        if path is None:
            self.use_api = True
        else:
            self.model = None
        pass

    def get_concept_from_X_list(self, x_list:List[int]) -> List[str]:
        # return a list of concepts
        if self.use_api:
            concept_list = []
            for i in range(self.C_num_return):
                concept_list.append("Odd number")
            return concept_list
