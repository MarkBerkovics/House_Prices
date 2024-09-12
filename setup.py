from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [req.strip() for req in content]

setup(
    name="houses",
      description="This package is for predicting housr prices",
      packages=find_packages(),
      install_requires=requirements
)
