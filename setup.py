from setuptools import setup, find_packages

setup(
    name="counterfusion",
    version="0.1.1",
    author="Lixian Wang",
    author_email="lixianphys@gmail.com",
    description="A light-weight package to solve large systems of interactions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LarsonLaugh/Counterfusion",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.1",
        "matplotlib>=3.4.3",
    ],
)
