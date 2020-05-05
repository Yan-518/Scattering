from setuptools import setup, find_packages

setup(
    name='NRCS',
    version='20.01',
    packages=['NRCS',
              'NRCS.constants',
              'NRCS.constants',
              'NRCS.model',
              'NRCS.spec',
              'NRCS.spread',
              'NRCS.modulation',
              'NRCS.validation'],
    # url='https://github.com/Yan-518/NRCS.git',
    author='Yan Yuan and other',
    author_email='y826302970y@outlook.com',
    description='',
    install_requires=['numpy', 'scipy', 'matplotlib']
)