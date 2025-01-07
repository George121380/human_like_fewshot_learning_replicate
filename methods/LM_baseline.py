from prompts.LM_baseline_prompt import LM_baseline_prompt
from ask_GPT import ask_GPT

class LM_Baseline:
    def __init__(self) -> None:
       pass

    def forward(self, x_list, x_test):
        while True:
            try:
                system_prompt, user_prompt = LM_baseline_prompt(x_test, x_list)
                p = float(ask_GPT(system_prompt, user_prompt))
                assert isinstance(p, float)
                return p
            
            except:
                print("Predicted probability is not a float number, retrying...")
                continue

    
    def inference(self, x_list, x_test):
        return self.forward(x_list, x_test)