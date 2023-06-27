from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ieseg_credscore',
    version='0.23.11',
    license='MIT',
    author="Philipp Borchert",
    author_email='p.borchert@ieseg.fr',
    packages=find_packages(),
    description = 'Credit Scoring - IESEG School of Management',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pnborchert',
    keywords='Credit Scoring IESEG',
    install_requires=[
          'pandas',
          'numpy',
      ],

)