from setuptools import setup, find_packages

setup(name='gym_fastsim',
      version='0.0.2',
      install_requires=['gym>=0.2.3','pyfastsim'],
      packages=find_packages(include=['gym_fastsim', 'gym_fastsim.*']),
      package_data={'gym_fastsim':['simple_nav/assets/*']},
      author='Alex Coninx',
      author_email='coninx@isir.upmc.fr'
)
