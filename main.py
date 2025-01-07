from dataset import Dataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from methods.prob_model import ProbModel
from methods.LM_baseline import LM_Baseline

#TODO: Score function between output and target
def eval_score(p, r):
    p = np.array(p)
    r = np.array(r)
    R2 = 1 - np.sum((p-r)**2)/np.sum((r-np.mean(r))**2)
    return R2

def train(epochs=100, C_num_return=5, useMSE=False):
    """
    Args:
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=False, C_num_return=C_num_return, useMSE=useMSE)
    dataset = Dataset()
    train_dataset, test_dataset = dataset.split()
    optimizer = torch.optim.Adam(prob_model.prior_model.mlp.parameters(), lr=0.01)
    if useMSE:
        logger = open(f"logs/{C_num_return}-tuning-MSE.log", "w")
    else:
        logger = open(f"logs/{C_num_return}-tuning.log", "w")
    
    pred = []
    target = []
    for idx in tqdm(range(test_dataset.get_length())):
        x_list, x_test, r = test_dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        pred.append(p.item())
        target.append(r)
    print(f"Epoch {0} score: {eval_score(pred, target)}")
    logger.write(f"Epoch {0} score: {eval_score(pred, target)}\n")
    for epoch in range(epochs):
        for idx in tqdm(range(train_dataset.get_length())):
            x_list, x_test, r = train_dataset.get_data(idx)
            loss = prob_model.loss(x_list, x_test, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = []
        target = []
        for idx in tqdm(range(test_dataset.get_length())):
            x_list, x_test, r = test_dataset.get_data(idx)
            p = prob_model.inference(x_list, x_test)
            pred.append(p.item())
            target.append(r)
        
        print(f"Epoch {epoch+1} score: {eval_score(pred, target)}")
        logger.write(f"Epoch {epoch+1} score: {eval_score(pred, target)}\n")
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
        pred.append(p.item())
        target.append(r)
    final_score = eval_score(pred, target)
    print(f"Final score: {final_score}")
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(f"logs/{C_num_return}-tuning.png")
    plt.close()
    return final_score

def eval(C_num_return=20, codegen=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=codegen, C_num_return=C_num_return)
    dataset = Dataset()
    pred = []
    target = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        pred.append(p.item())
        target.append(r)
    final_score = eval_score(pred, target)
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

# Given a list of concepts, test the performance of the model
def eval_all():
    scores = []
    cnums = [100]
    for i in cnums:
        scores.append(eval(i))
    for i, score in zip(cnums, scores):
        print(f"C_num_return: {i}, score: {score}")

# Given a list of concept numbers, train the model and return the scores
def train_all():
    scores = []
    cnums = [100]
    for i in cnums:
        scores.append(train(epochs=10, C_num_return=i))
    for i, score in zip(cnums, scores):
        print(f"C_num_return: {i}, score: {score}")

# Given a list of concepts, test the performance of the model
def test_fix(cs, file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model = ProbModel(device=device, codegen=True, C_num_return=5, fixed_return=cs)
    dataset = Dataset(one_file=True)
    pred = []
    target = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        pred.append(p.item())
        target.append(r)
    final_score = eval_score(pred, target)
    print(f"Final score: {final_score}")
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(file_name)
    plt.close()

def test_concept():
    H = ["multiple of 8","perfect square","power of 2"]
    # L = ["number larger than 1","number larger than 2", "number equal to 16"]
    L = ["number larger than 16","number larger than 26", "number equal to 36", "number larger than 46", "number larger than 56", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66", "number larger than 66"]
    test_fix(H, 'analysis/H.png')
    test_fix(L, 'analysis/L.png')
    test_fix(H+L, 'analysis/H+L.png')

def baseline_lm(file_name="analysis/baseline_lm.png"):
    prob_model = LM_Baseline()
    dataset = Dataset()
    pred = []
    target = []
    data = []
    for idx in tqdm(range(dataset.get_length())):
        x_list, x_test, r = dataset.get_data(idx)
        p = prob_model.inference(x_list, x_test)
        pred.append(p)
        target.append(r)
        data.append((x_list, x_test))
    final_score = eval_score(pred, target)
    print(f"Final score: {final_score}")
    plt.scatter(pred, target)
    plt.xlabel("Predict")
    plt.ylabel("Target")
    plt.savefig(file_name)
    plt.close()
    with open('analysis/baseline_lm.txt', 'w') as f:
        for d, t, p in zip(data, target, pred):
            f.write(f"x_list: {d[0]} x_test: {d[1]} target: {t} predict: {p}\n")

if __name__ == "__main__":
    baseline_lm()