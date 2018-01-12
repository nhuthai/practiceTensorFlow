import numpy as np
from math import floor

class Data:
    def __init__(self):
        self.createData()

    def normalize(self, vector):
        return (vector - vector.mean()) / vector.std()

    def createData(self):
        self.n_house = 160

        # Generate data
        np.random.seed(42)
        house_size = np.random.randint(low=1000, high=3500, size=self.n_house)
    
        np.random.seed(42)
        house_price = house_size*100 + np.random.randint(low=20000, high=70000, size=self.n_house)

        # Normalized data
        house_size_norm = self.normalize(house_size)
        house_price_norm = self.normalize(house_price)

        # Training data
        self.n_train = floor(0.7 * self.n_house)
        self.train_house_size = np.array(house_size_norm[:self.n_train])
        self.train_house_price = np.array(house_price_norm[:self.n_train])

        # Testing data
        self.test_house_size = np.array(house_size_norm[self.n_train:])
        self.test_house_price = np.array(house_price_norm[self.n_train:])