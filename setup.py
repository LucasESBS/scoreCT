import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="scorect",
    version="0.0.2",
    author="Lucas Seninge",
    author_email="lseninge@ucsc.edu",
    description="scoreCT: Automated cell type annotation in scRNA-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucasESBS/scoreCT",
    packages=setuptools.find_packages(),
    classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ),
)
