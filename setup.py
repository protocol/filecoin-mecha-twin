import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mechaFIL",
    version="1.2",
    author="Maria Silva, Tom Mellan, Kiran Karra",
    author_email="misilva73@gmail.com, t.mellan@imperial.ac.uk, kiran.karra@gmail.com",
    description="Deterministic model for the Filecoin Economy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/protocol/filecoin-mecha-twin",
        "Source": "https://github.com/protocol/filecoin-mecha-twin",
    },
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.23.1", "pandas>=1.4.3", "requests>=2.28.1"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
