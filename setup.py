from setuptools import setup

setup(
    name='FINN',
    version='0.1.0',
    author='T. Kehrenbergg, M. Bartlet, O. Tomas',
    packages=['finn'],
    description='Invertible Networks for Learning Fair Representations',
    python_requires=">=3.6",
    install_requires=[
        "comet-ml >= 1.0.51",
        "numpy >= 1.15",
        "pandas >= 0.24",
        'pyro-ppl >= 0.3.0',
        "scikit-learn >= 0.20",
        "scikit-image >= 0.14",
        "tqdm >= 4.31",
        "torch >= 1.0",
        "scipy >= 1.2.1",
        "torchvision >= 0.2.2.post3"
    ],
)
