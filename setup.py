from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cluster_work',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.2.4',

    description='A framework to run experiments on an computing cluster.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/gregorgebhardt/cluster_work',

    # Author details
    author='Gregor Gebhardt',
    author_email='gregor.gebhardt@gmail.com',

    # Choose your license
    license='BSD-3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: System :: Distributed Computing',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Education',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Environment :: Console'
    ],

    python_requires='>=3',

    # What does your project relate to?
    keywords='',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['examples', 'experiments', 'notebooks']),


    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    py_modules=["cluster_work", "plot_work", "cluster_work_tools"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['PyYAML', 'pandas', 'matplotlib'],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'hostfileconv=cluster_work_tools:convert_hostfile',
        ],
    },
)
