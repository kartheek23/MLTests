from setuptools import setup

with open("README.md","r",encoding='utf-8') as f:
    long_description = f.read()

REPO_NAME="MLTests"
AUTHOR_USER_NAME="kartheek23"
SRC_REPO="src"
LIST_OF_REQUIREMENTS=[
"pandas",
"numpy",
"Scipy",
"scikit-learn==0.22.1",
"pytest",
"feature_engine>=0.3.1,<0.4.0",
"marshmallow>=3.2.2,<4.0",
"tid-regression-model>=2.0.20,<2.1.0",
'joblib>=0.14.1,<0.15.0']

setup(
    name=SRC_REPO,
    version="0.0.1",
    author= AUTHOR_USER_NAME,
    description="A framework for testing ML projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="challakartheek@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)
