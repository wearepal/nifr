from setuptools import setup, find_packages

setup(
    name='FINN',
    version='0.1.0',
    author='T. Kehrenbergg, M. Bartlett, O. Thomas',
    packages=find_packages(),
    description='Invertible Networks for Learning Fair Representations',
    python_requires=">=3.6",
    install_requires=[
        "comet-ml >= 1.0.51",
        "numpy >= 1.15",
        "pandas >= 0.24",
        "scikit-learn >= 0.20",
        "scikit-image >= 0.14",
        "tqdm >= 4.31",
        "torch >= 1.2",
        "scipy >= 1.2.1",
        "torchvision >= 0.4.0"
    ],
)
