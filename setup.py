from setuptools import find_packages, setup

setup(
    name='kooopman_learning_and_control',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cvxpy',
        'keras',
        'backcall',
        'certifi==2020.12.5',
        'cvxpy',
        'cycler==0.10.0',
        'decorator',
        'dill',
        'ecos',
        'ipykernel',
        'ipython',
        'ipython-genutils',
        'jedi==0.17.0',
        'joblib',
        'jupyter-client',
        'jupyter-core',
        'kiwisolver',
        'llvmlite==0.36.0',
        'matplotlib',
        'mkl-fft==1.3.0',
        'mkl-random==1.1.1',
        'mkl-service==2.3.0',
        'numba==0.53.0',
        'numba-scipy==0.2.0',
        'numpy',
        'olefile==0.46',
        'osqp',
        'parso',
        'pexpect',
        'pickleshare',
        'Pillow',
        'prompt-toolkit',
        'ptyprocess',
        'Pygments',
        'pyparsing',
        'python-dateutil',
        'pyzmq==20.0.0',
        'qdldl',
        'scikit-learn',
        'scipy',
        'scs',
        'six',
        'tabulate==0.8.9',
        'threadpoolctl',
        'torch==1.8.0',
        'torchaudio==0.8.0a0+a751e1d',
        'torchvision==0.9.0',
        'tornado',
        'traitlets',
        'typing-extensions',
        'wcwidth',
        'ray[default]',
        'numba'
    ],
    extra_require={
        'dev': ['pytest']
    }
)
