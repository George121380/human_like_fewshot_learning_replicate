from sentence_transformers import SentenceTransformer
import torch
from typing import List
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import gather_log_probabilities


class PriorModel:
    def __init__(self, device, mlp_path=None, model_path='/media/george/Projects/Labs/CogSci_labs/models/all-MiniLM-L6-v2', hidden_size=384):
        self.model = SentenceTransformer(model_path).to(device)
        self.mlp = torch.nn.Linear(hidden_size, 1).to(device)
        if mlp_path is not None:
            self.mlp.load_state_dict(torch.load(mlp_path))
        self.device = device
        
    def forward(self, sentences: List[str]):
        vector = self.model.encode(sentences)
        vector = torch.tensor(vector).to(self.device)
        prior = self.mlp(vector)
        return prior
    
    def save_mlp(self, path):
        torch.save(self.mlp.state_dict(), path)


class PriorModelCodegen:
    def __init__(self, device, normalize_len=True, path="/media/george/Projects/Labs/CogSci_labs/models/codegen-350M-mono"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.normalize_len = normalize_len
        self.model.to(device)
        self.device = device

    def forward(self, sentence: List[str]):
        input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids.to(self.device)
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
            prob = torch.exp(prob)
        return prob
        
def test():
    model = PriorModel()
    print(model.forward(['I am a sentence', 'I am another sentence']))

if __name__ == '__main__':
    test()