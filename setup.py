from setuptools import setup

setup(
    name='VesMod',
    version='1.0',
    description='Extract the bending MODulus from a video of a VESicle',
    url='https://github.com/BranniganLab/VesMod',
    author='Brannigan Lab',
    author_email='grace.brannigan@rutgers.edu',
    packages=['vesmod'],
    install_requires=['numpy>=2.0.0', 'opencv-python-headless', 'matplotlib>=3.9.1', 'scikit-image', 'nd2', 'scipy', 'lmfit'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
)
