from setuptools import setup,find_packages

setup(name='QCircLearning',
      version='0.5',
      description='Qiskit based variational circuit',
      url='https://github.com/Canoming/QCircLearning',
      author='Canoming',
      author_email='canoming@163.com',
      license='MIT',
      packages=find_packages(exclude=["*.tests","*.tests.*","tests.*","tests"]),
      install_requires=[
          'qiskit >= 1.0.0',
          'torch >= 2.0.0',
          'pylatexenc >= 2.8',
          'matplotlib >= 3.0.0',
      ],
      zip_safe=False)