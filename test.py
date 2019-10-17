import matplotlib.pyplot as plt

scale = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
nosinn = [
    96.05,
    97.11,
    97.46,
    97.46,
    97.58,
    97.56
]

vae = [
    98.5,
    98.5,
    98.5,
    96.9,
    96.3,
    93.1
]


plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.plot(scale, nosinn, label="NoSINN", marker="v")
ax.plot(scale, vae, label="CVAE", marker="s")
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel('Accuracy')
ax.set_title("CMNIST")
plt.ylim(90, 100)
ax.legend()
fig.savefig("cmnist.pdf")
plt.show()
