import os
import csv

class Dataset:
    def __init__(self, dir_path='/media/george/Projects/Labs/CogSci_labs/tenenbaum_data', data=None):
        if data is not None:
            self.data = data
            return
        file_list = os.listdir(dir_path)
        self.data = []
        for file in file_list:
            file_path = os.path.join(dir_path, file)
            # load csv data
            all_num = []
            csv_reader = csv.reader(open(file_path, 'r'))
            for row in csv_reader:
                num = int(row[0])
                rate = float(row[1])
                self.data.append((
                    all_num.copy(),
                    num,
                    rate
                ))
                all_num.append(num)
        print(f"Loaded {len(self.data)} data points")
    def get_length(self):
        return len(self.data)
    
    def get_data(self, idx):
        return self.data[idx]

    def split(self, ratio=0.9):
        split_idx = int(len(self.data) * ratio)
        return Dataset(data=self.data[:split_idx]), Dataset(data=self.data[split_idx:])