from setuptools import setup, find_packages

setup(name='sencoder',
      packages=find_packages(),
      version="0.1.0",
      description='Autoencoder for sentences',
      author='Satchel Grant',
      author_email='grantsrb@gmail.com',
      url='',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
