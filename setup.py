from setuptools import setup, find_packages

setup(
        name='retroeval',
        version='1.0',
        description='Simple evaluation tools for retrosynthesis models.',
        packages=find_packages(exclude=[]),
        python_requires='>=3.8',
        # Note: python 3.8 is required aizynthfinder
)
