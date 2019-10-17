
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = torch.randn(2, 3 * 2, 12, 12)
conv = nn.Conv2d(3, 4, 3, 1, 1)
glu = nn.GLU(dim=1)
print(glu(x).shape)

# scales = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# nosinn = [0.9605, 0.9711, 0.9746, 0.9746, 0.9758, 0.9756]
# bl_col = [0.1107, 0.2263, 0.3457, 0.4444, 0.5654, 0.6357]
# bl_gs = [0.5877, 0.7021, 0.7686, 0.8254, 0.9173, 0.938]
#
# plt.style.use("seaborn")
# plt.plot(scales, nosinn, label="NoSINN (Ours)")
# plt.plot(scales, bl_col, label="CNN")
# plt.plot(scales, bl_gs, label="CNN (Greyscale)")
#
# plt.xlabel(r"$\sigma$", size=14)
# plt.ylabel("Accuracy", size=14)
# plt.xticks(size=12)
# plt.yticks(np.arange(0, 1.1, 0.1), size=12)
# plt.xlim(0, 0.5)
# plt.ylim(0, 1)
# plt.legend(fontsize=12)
# plt.savefig("CMNIST")
