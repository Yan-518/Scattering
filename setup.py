from setuptools import setup, find_packages

setup(
    name='NRCS',
    version='20.01',
    packages=['NRCS',
              'NRCS.bistatic',
              'NRCS.constants',
#               'NRCS.current',
              'NRCS.LUT',
              'NRCS.model',
              'NRCS.spec',
              'NRCS.spread',
              'NRCS.modulation',
              'NRCS.validation'],
    # url='https://github.com/Yan-518/NRCS.git',
    author='Yan Yuan and others',
    author_email='y826302970y@outlook.com',
    description='',
    install_requires=['numpy', 'scipy', 'matplotlib', 'stereoid', 'drama']
)
