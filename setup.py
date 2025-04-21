from setuptools import setup

setup(
    name='fNIRS_PFC_2025',
    version='0.1.0',
    packages=['fnirs_PFC_2025', 'fnirs_PFC_2025.cli', 'fnirs_PFC_2025.viz', 'fnirs_PFC_2025.read',
              'fnirs_PFC_2025.processing', 'fnirs_PFC_2025.preprocessing'],
    url='https://github.com/tsujik2024',
    license='MIT',
    author='Keiko Tsuji',
    author_email='tsujik@ohsu.edu',
    description='A Python pipeline for preprocessing and analyzing fNIRS data from the prefrontal cortex.',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'openpyxl',     # for Excel support
        'tqdm',
        'natsort',
        'setuptools'])
