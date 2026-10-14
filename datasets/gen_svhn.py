import os
import numpy as np
from torchvision.datasets import SVHN

os.makedirs("SVHN", exist_ok=True)
for split in ("train", "test"):
    ds = SVHN("./data", split=split, download=True)
    x = ds.data.transpose(0, 2, 3, 1).reshape(len(ds.data), -1)
    y = ds.labels.reshape(-1, 1)
    np.savetxt(f"SVHN/svhn_{split}.csv", np.hstack([y, x]), fmt="%d", delimiter=",")