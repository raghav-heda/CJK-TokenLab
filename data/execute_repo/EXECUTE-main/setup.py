# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


install_requires = [
    "datasets>=2.17.1",
    "sacremoses>=0.1.1",
    "sentence-splitter>=1.4",
    "pycountry>=22.3.5",
    "googletrans>=4.0.2",
    "tqdm>=4.65.0",
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "bitsandbytes>=0.43.1",
    "psutil>=5.9.8",
    "tiktoken>=0.6.0",
]
dependency_links = []


class PostInstall(install):
    def run(self):
        install.run(self)


setup(
    name="execute",
    version="0.1.0",
    author="anonymous",
    description="A multilingual version of CUTE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="anonymous",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    dependency_links=dependency_links,
    cmdclass={"install": PostInstall},
)