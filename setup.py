from setuptools import setup

setup(
    name='vesicle_edge_extractor',
    version='1.0',
    description='Find the edge of a single vesicle from phase contrast microscope .nd2 files',
    url='https://github.com/BranniganLab/vesicle_edge_extractor',
    author='Brannigan Lab',
    author_email='grace.brannigan@rutgers.edu',
    packages=['vesicle_edge_extractor'],
    install_requires=['numpy>=2.0.0', 'opencv-python-headless', 'matplotlib>=3.9.1', 'scikit-image', 'nd2', 'scipy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
)
