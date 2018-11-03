from setuptools import setup, find_packages

setup(
    name='keras-bert',
    version='0.19.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-bert',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='BERT implemented in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
        'keras-transformer==0.10.0',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
