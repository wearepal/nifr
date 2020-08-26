from setuptools import find_packages, setup

setup(
    name="NIFR",
    version="0.2.0",
    author="T. Kehrenberg, M. Bartlett, O. Thomas",
    packages=find_packages(),
    description="Null-sampling for Interpretable and Fair Representations",
    python_requires=">=3.8",
    package_data={"nifr": ["py.typed"]},
    install_requires=[
        "captum",
        "EthicML",
        "gitpython",
        "numpy >= 1.15",
        "pandas >= 0.24",
        "scikit-image >= 0.14",
        "scikit-learn >= 0.20",
        "scipy >= 1.2.1",
        "torch >= 1.2",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31",
        "typed-argument-parser == 1.4",
        "typing-extensions >= 3.7.4",
        "typing-inspect >= 0.5",
        "wandb == 0.8.27",
    ],
)
