from setuptools import setup, find_packages

setup(
    name="NoSINN",
    version="0.2.0",
    author="T. Kehrenberg, M. Bartlett, O. Thomas",
    packages=find_packages(),
    description="Invertible Networks for Learning Fair Representations",
    python_requires=">=3.6",
    package_data={"nosinn": ["py.typed"]},
    install_requires=[
        "captum",
        "numpy >= 1.15",
        "pandas >= 0.24",
        "pillow < 7.0",
        "gitpython",
        "scikit-image >= 0.14",
        "scikit-learn >= 0.20",
        "scipy >= 1.2.1",
        "torch >= 1.2",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31",
        "typed-argument-parser == 1.4",
        "typing-extensions >= 3.7.4",
        "typing-inspect >= 0.5",
        "wandb >= 0.8, != 0.8.5, != 0.8.6, != 0.8.7, != 0.8.8, != 0.8.9, "
        "!= 0.8.10, != 0.8.11, != 0.8.12, != 0.8.13, != 0.8.14, != 0.8.15, "
        "!= 0.8.16, != 0.8.17, != 0.8.18, != 0.8.19, != 0.8.20, != 0.8.21, "
        "!= 0.8.22",
        "EthicML == 0.1.0a5",
    ],
)
