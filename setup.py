from setuptools import setup
import io

# some influences here came from https://github.com/audreyr/cookiecutter/blob/master/setup.py

version = '0.2.0'


with io.open('README.rst', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name     = 'asap_stereo',
    version  = version,
    packages = ['asap_stereo'],
    include_package_data=True,
    license  = 'BSD-3-Clause',
    description = 'A high level CLI and reproduceable workflow for the Ames Stereo Pipeline',
    long_description = readme,
    # Author details
    author='Andrew M. Annex',
    author_email='ama6fy@virginia.edu',
    url='https://github.com/AndrewAnnex/asap_stereo/',

    install_requires=['requests', 'fire', 'moody>=0.0.4', 'sh', 'papermill', 'rasterio', 'pyproj'],

    entry_points={
        'console_scripts': [
            'asap = asap_stereo.asap:main'
        ]
    },

    keywords=['mars', 'nasa', 'asp', 'ames', 'stereo', 'pipeline', 'cli', 'tool', 'workflow'],

    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: GIS'
    ]
)
