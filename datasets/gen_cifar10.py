import os
import numpy as np
from torchvision.datasets import CIFAR10

os.makedirs("CIFAR-10", exist_ok=True)
for split, train in (("train", True), ("test", False)):
    ds = CIFAR10("./data", train=train, download=True)
    x = ds.data.reshape(len(ds.data), -1)
    y = np.array(ds.targets).reshape(-1, 1)
    np.savetxt(f"CIFAR-10/cifar10_{split}.csv", np.hstack([y, x]), fmt="%d", delimiter=",")