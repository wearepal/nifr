from setuptools import find_packages, setup

setup(
    name="NSFIAIR",
    version="0.2.0",
    author="T. Kehrenberg, M. Bartlett, O. Thomas",
    packages=find_packages(),
    description="Null-sampling for Invariant and Interpretable Representations",
    python_requires=">=3.6",
    package_data={"nsfiair": ["py.typed"]},
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
        "wandb == 0.8.27",
        "EthicML @ git+https://github.com/predictive-analytics-lab/EthicML.git",
    ],
)
