from x2concept import X2Concept

if __name__ == "__main__":
    model = X2Concept(C_num_return=20)
    x_list = [2, 4, 6, 8, 10]
    x_list = [16]
    concepts = model.get_concept_from_X_list(x_list)
    print(", ".join(concepts))