import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, name, *params):
        self.supported = ["gaussian"]
        if name not in self.supported:
            raise ValueError(f"`{name}` is not supported")
        self.name = name
        self.n_params = len(params)
        self.d_params = params


    @property
    def params(self):
        return self.d_params
    

    def gaussian_prob(self, x):
        return x + 3
    

    def weibull_prob(self, x):
        return x - 5


    def prob(self, x):
        return getattr(self, f"{self.name}_prob")(x)
    

    def survival(self, x):
        return getattr(self, f"{self.name}_survival")(x)
    

    def log_prob(self, x):
        return getattr(self, f"{self.name}_log_prob")(x)
    

    def log_survival(self, x):
        return getattr(self, f"{self.name}_log_survival")(x)


gaussian = Distribution("gaussian", [2, 3], [1, 2])
print(gaussian.params)
print(gaussian.prob(2))