from setuptools import setup, find_packages

def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="freeflux",
    version="0.1.0",
    author="litao",
    author_email="",  # Add your email here
    description="A package for efficient quantization and inference acceleration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/litaotju/flux-4bit.git",  # Update with your repository URL
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'freeflux-app=freeflux.app:main',
            'freeflux-gen=freeflux.cli:main',
            'freeflux-prompt=freeflux.prompts_gen:main',
        ],
    },
)
