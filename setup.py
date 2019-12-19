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
        "pillow",
        "scikit-image >= 0.14",
        "scikit-learn >= 0.20",
        "scipy >= 1.2.1",
        "torch >= 1.2",
        "torchvision >= 0.4.0",
        "tqdm >= 4.31",
        "typed-argument-parser "
        "@ git+https://github.com/predictive-analytics-lab/typed-argument-parser@v1.4",
        "typing-extensions >= 3.7.4",
        "typing-inspect >= 0.5",
        "wandb >= 0.8",
        "ethicml @ git+https://github.com/predictive-analytics-lab/EthicML@r0.1.0a4",
    ],
)
