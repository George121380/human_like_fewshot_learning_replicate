from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from utils import gather_log_probabilities
from typing import List

class PriorModelCodegen:
    def __init__(self, normalize_len=True, path="/media/george/Projects/Labs/CogSci_labs/models/codegen-350M-mono"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.normalize_len = normalize_len
        self.model.post_init()

    def get_likelihood(self, sentence: List[str]):
        input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids
        with torch.no_grad():
            output:CausalLMOutputWithPast = self.model(input_ids)
            logits = output.logits
            labels = input_ids.clone().detach()
            log_prob = gather_log_probabilities(logits[:, :-1], labels[:, 1:])
            seq_len = labels.size(1)
            if self.normalize_len:
                prob = log_prob.sum(dim=-1) / seq_len
            else:
                prob = log_prob.sum(dim=-1)
        return prob

priorModelCodegen = PriorModelCodegen(normalize_len=True)
text = "An odd number"
output = priorModelCodegen.get_likelihood(text)
print(output.item())