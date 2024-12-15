from prior_model import PriorModel, PriorModelCodegen
from dataset import Dataset
from tqdm import tqdm
from x2concept import X2Concept
from concept2Python import Concept2Python
from utils import remove_eligal_characters
import torch

class ProbModel:
    def __init__(self, device, eps=0.01, range=range(1,101), codegen=True) -> None:
        if codegen:
            self.prior_model = PriorModelCodegen(device=device)
        else:
            self.prior_model = PriorModel(device=device)
        self.x2concept = X2Concept(C_num_return=3)
        self.concept2python = Concept2Python(device=device)
        self.eps = eps
        self.range = range

    def forward(self, x_list, x_test):
        while True:
            try:
                concept_list = self.x2concept.get_concept_from_X_list(x_list)
                w_c_dict = dict()
                xtest_c_dict = dict()
                for concept in concept_list:
                    p_c = self.prior_model.forward(concept)
                    program = self.concept2python.get_program_from_concept(concept)
                    program = remove_eligal_characters(program)
                    # 创建独立的命名空间
                    local_namespace = {}
                    # 在命名空间中执行代码
                    exec(program, {}, local_namespace)
                    def test(x):
                        # 在特定环境中调用函数
                        if "test_function" in local_namespace:
                            result = local_namespace["test_function"](x)
                            return result
                        else:
                            print("函数未定义")
                    
                    select_total = 0 # number that satisfy the concept
                    for i in self.range:
                        if test(i):
                            select_total += 1
                    if test(x_test):
                        xtest_c_dict[concept] = 1
                    else:
                        xtest_c_dict[concept] = 0

                    w_c_dict[concept] = p_c
                    for x in x_list:
                        if test(x):
                            w_c_dict[concept] *= ((1-self.eps)/select_total + self.eps / 100)
                        else:
                            w_c_dict[concept] *= (self.eps / 100)

                w_total = 1e-20 # avoid zero division
                for concept in concept_list:
                    w_total += w_c_dict[concept]
                p = 0
                for concept in concept_list:
                    p += w_c_dict[concept] / w_total * xtest_c_dict[concept]
                return p
            except:
                with open("error.log", "a") as f:
                    f.write(f"x_list: {x_list}, x_test: {x_test}, concept_list: {concept_list}\n")
                    f.write(f"w_c_dict: {w_c_dict}, xtest_c_dict: {xtest_c_dict}\n")
                    f.write(f"p: {p}\n")
                    f.write(f"program: {program}\n")
                    f.write(f"local_namespace: {local_namespace}\n")
                    f.write(f"select_total: {select_total}\n")
                    f.write(f"concept: {concept}\n")
                    f.write(f"range: {self.range}\n")
                    f.write(f"error\n")
                    f.write("\n")
    
    def inference(self, x_list, x_test):
        with torch.no_grad():
            return self.forward(x_list, x_test)
        
    def loss(self, x_list, x_test, r):
        p = self.forward(x_list, x_test)
        return -(r*p+(1-r)*(1-p))

#TODO: Score function between output and target
def eval_score(p, r):
    return (r*p+(1-r)*(1-p))

def train(epochs=5):
    """
    Args:
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=False)
    dataset = Dataset()
    train_dataset, test_dataset = dataset.split()
    optimizer = torch.optim.Adam(prob_model.prior_model.mlp.parameters(), lr=0.01)
    for epoch in range(epochs):
        for idx in tqdm(range(train_dataset.get_length())):
            x_list, x_test, r = train_dataset.get_data(idx)
            loss = prob_model.loss(x_list, x_test, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        all_score = []
        for idx in tqdm(range(test_dataset.get_length())):
            x_list, x_test, r = test_dataset.get_data(idx)
            p = prob_model.inference(x_list, x_test)
            all_score.append(eval_score(p, r))
        print(f"Epoch {epoch+1} score: {(sum(all_score)/len(all_score)).item()}")

def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=True)
    dataset = Dataset()
    all_score = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        all_score.append(eval_score(p, r))
        print(eval_score(p, r))
    print(f"Final score: {(sum(all_score)/len(all_score)).item()}")

if __name__ == "__main__":
    train()