from prior_model import PriorModel, PriorModelCodegen
from dataset import Dataset
from tqdm import tqdm
from x2concept import X2Concept
from concept2Python import Concept2Python
from utils import remove_eligal_characters
import torch
import os
import matplotlib.pyplot as plt

class ProbModel:
    def __init__(self, device, eps=0.01, range=range(1,101), codegen=True, C_num_return=3, useMSE=False, fixed_return=None) -> None:
        if codegen:
            self.prior_model = PriorModelCodegen(device=device)
        else:
            self.prior_model = PriorModel(device=device)
        if fixed_return is not None:
            self.x2concept = X2Concept(C_num_return=C_num_return, fixed_return=fixed_return)
        else:
            self.x2concept = X2Concept(C_num_return=C_num_return)
        self.concept2python = Concept2Python(device=device)
        self.eps = eps
        self.range = range
        self.codegen = codegen
        if "logs" not in os.listdir():
            os.mkdir("logs")
        if codegen:
            self.info_logger = open(f"logs/{C_num_return}-codegen.log", "w")
        self.error_logger = open("error.log", "w")
        if useMSE:
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn.to(device)

    def forward(self, x_list, x_test):
        for i in range(3):
            # 尝试三次，如果失败则记录错误
            try:
                concept_list = self.x2concept.get_concept_from_X_list(x_list)
                if self.codegen:
                    self.info_logger.write(f"x_list: {x_list}\n")
                    self.info_logger.write(f"x_test: {x_test}\n")
                    self.info_logger.write(f"concept_list: {concept_list}\n")
                w_c_dict = dict()
                xtest_c_dict = dict()
                for concept in concept_list:
                    p_c = self.prior_model.forward(concept)
                    program = self.concept2python.get_program_from_concept(concept)
                    if self.codegen:
                        self.info_logger.write(f"concept: {concept} prior: {p_c.item()} program: {program}\n")
                    program = remove_eligal_characters(program)
                    # 创建独立的命名空间
                    local_namespace = {}
                    # 在命名空间中执行代码
                    try:
                        exec(program, {}, local_namespace)
                        def test(x):
                            # 在特定环境中调用函数
                            if "test_function" in local_namespace:
                                try:
                                    result = local_namespace["test_function"](x)
                                    return result
                                except:
                                    # self.info_logger.write(f"test_function error\n")
                                    return False
                            else:
                                print("函数未定义")
                    except:
                        if self.codegen:
                            self.info_logger.write(f"exec error\n")
                        def test(x):
                            return False
                    
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
                if self.codegen:
                    self.info_logger.write(f"p: {p.item()}\n")
                    self.info_logger.write("\n")
                return p
            except Exception as e:
                self.error_logger.write(e)
                with open("error.log", "a") as f:
                    f.write(f"x_list: {x_list}, x_test: {x_test}, concept_list: {concept_list}\n")
                    f.write(f"w_c_dict: {w_c_dict}, xtest_c_dict: {xtest_c_dict}\n")
                    f.write(f"program: {program}\n")
                    f.write(f"p: {p}\n")
                    f.write(f"local_namespace: {local_namespace}\n")
                    f.write(f"select_total: {select_total}\n")
                    f.write(f"concept: {concept}\n")
                    f.write(f"range: {self.range}\n")
                    f.write(f"error\n")
                    f.write("\n")
        raise Exception("Failed to get the result")
    
    def inference(self, x_list, x_test):
        with torch.no_grad():
            return self.forward(x_list, x_test)
        
    def loss(self, x_list, x_test, r):
        p = self.forward(x_list, x_test)
        r = torch.tensor([r]).to(p.device)
        return self.loss_fn(p, r)

#TODO: Score function between output and target
def eval_score(p, r):
    return (r*p+(1-r)*(1-p))

def train(epochs=100, C_num_return=5, useMSE=False):
    """
    Args:
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=False, C_num_return=C_num_return, useMSE=useMSE)
    dataset = Dataset()
    train_dataset, test_dataset = dataset.split()
    optimizer = torch.optim.Adam(prob_model.prior_model.mlp.parameters(), lr=0.01)
    logger = open(f"logs/{C_num_return}-tuning-mse.log", "w")
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
        logger.write(f"Epoch {epoch+1} score: {(sum(all_score)/len(all_score)).item()}\n")
        for key, value in prob_model.x2concept.cache.items():
            logger.write(f"x list: {key}\n")
            with torch.no_grad():
                for v in value:
                    prior = prob_model.prior_model.forward(v).item()
                    logger.write(f"concept: {v} prior: {prior}\n")
            
    pred = []
    target = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        all_score.append(eval_score(p, r))
        pred.append(p.item())
        target.append(r)
    final_score = (sum(all_score)/len(all_score)).item()
    print(f"Final score: {final_score}")
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(f"logs/{C_num_return}-tuning-mse.png")
    plt.close()
    return final_score

def eval(C_num_return=20, codegen=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=codegen, C_num_return=C_num_return)
    dataset = Dataset()
    all_score = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        all_score.append(eval_score(p, r))
    final_score = (sum(all_score)/len(all_score)).item()
    print(f"Final score: {final_score}")
    return final_score

def draw(C_num_return=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=True, C_num_return=C_num_return)
    dataset = Dataset()
    pred = []
    target = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        pred.append(p.item())
        target.append(r)
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(f"logs/{C_num_return}.png")

def eval_all():
    scores = []
    cnums = [1,2,3,5,10,30,100]
    for i in cnums:
        scores.append(eval(i))
    for i, score in zip(cnums, scores):
        print(f"C_num_return: {i}, score: {score}")

def train_all():
    scores = []
    cnums = [1,2,3,5,10,30,100]
    for i in cnums:
        scores.append(train(epochs=30, C_num_return=i))
    for i, score in zip(cnums, scores):
        print(f"C_num_return: {i}, score: {score}")

def test_fix(cs, file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=True, C_num_return=5, fixed_return=cs)
    dataset = Dataset(one_file=True)
    pred = []
    target = []
    all_score = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        all_score.append(eval_score(p, r))
        pred.append(p.item())
        target.append(r)
    final_score = (sum(all_score)/len(all_score)).item()
    print(f"Final score: {final_score}")
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(file_name)
    plt.close()

def test_concept():
    H = ["multiple of 8","perfect square","power of 2"]
    L = ["number larger than 1","number larger than 2", "number equal to 16"]
    test_fix(H, 'analysis/H.png')
    test_fix(L, 'analysis/L.png')
    test_fix(H+L, 'analysis/H+L.png')

if __name__ == "__main__":
    test_concept()