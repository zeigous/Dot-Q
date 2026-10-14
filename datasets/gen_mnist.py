import os
import numpy as np
from torchvision.datasets import MNIST

os.makedirs("MNIST", exist_ok=True)
for split, train in (("train", True), ("test", False)):
    ds = MNIST("./data", train=train, download=True)
    x = ds.data.numpy().reshape(len(ds.data), -1)
    y = ds.targets.numpy().reshape(-1, 1)
    np.savetxt(f"MNIST/mnist_{split}.csv", np.hstack([y, x]), fmt="%d", delimiter=",")