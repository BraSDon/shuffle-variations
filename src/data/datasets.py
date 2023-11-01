import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class SUSYDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()

        self.train = train
        self.transform = transform

        features = [
            "class",
            "lepton 1 pT",
            "lepton 1 eta",
            "lepton 1 phi",
            "lepton 2 pT",
            "lepton 2 eta",
            "lepton 2 phi",
            "missing energy magnitude",
            "missing energy phi",
            "MET_rel",
            "axial MET",
            "M_R",
            "M_TR_2",
            "R",
            "MT2",
            "S_R",
            "M_Delta_R",
            "dPhi_r_b",
            "cos(theta_r1)",
        ]
        df = pd.read_csv(root + "/SUSY.csv", header=None)
        df.columns = features

        y = df["class"]
        X = df.drop("class", axis=1)

        # Perform stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.9, stratify=y
        )

        if train:
            X_train = (X_train - X_train.mean()) / X_train.std()
            self.data = X_train
            self.targets = y_train
        else:
            X_test = (X_test - X_test.mean()) / X_test.std()
            self.data = X_test
            self.targets = y_test

        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.targets = torch.tensor(self.targets.values, dtype=torch.int64)

    def __getitem__(self, index):
        date, target = self.data[index], self.targets[index]

        if self.transform is not None:
            date = self.transform(date)

        return date, target

    def __len__(self):
        return len(self.data)
