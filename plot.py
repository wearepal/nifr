from pathlib import Path

from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd


def main():
#     scale = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
#     nosinn = [96.05, 97.11, 97.46, 97.46, 97.58, 97.56]

#     vae = [98.5, 98.5, 98.5, 96.9, 96.3, 93.1]

#     style.use("seaborn")
#     fig, plot = plt.subplots(figsize=(6, 4), dpi=200)
#     plot.plot(scale, nosinn, label="NoSINN", marker="v")
#     plot.plot(scale, vae, label="CVAE", marker="s")
#     plot.set_xlabel(r"$\sigma$")
#     plot.set_ylabel("Accuracy")
#     plot.set_title("CMNIST")
#     plot.set_ylim(90, 100)
#     plot.legend()
#     fig.savefig(Path(".") / "cmnist.pdf")
#     plt.show()


    nosinn = pd.read_excel("LocalDataHD/mb715/Documents/Fairness/FINN/results/adult")



if __name__ == "__main__":
    main()
