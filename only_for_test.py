import torch


if __name__ == "__main__":
    a = torch.tensor([1.]).to(torch.device("cuda"))
    print(a.item())