from setuptools import setup
import io

# some influences here came from https://github.com/audreyr/cookiecutter/blob/master/setup.py

version = '0.0.1dev'


with io.open('README.rst', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name     = 'asap_stereo',
    version  = version,
    packages = ['asap_stereo'],
    license  = 'MIT',
    description = 'A CLI interface to asp_scripts',
    long_description = readme,
    # Author details
    author='Andrew Annex',
    author_email='annex@jhu.edu',
    url='https://github.com/andrewannex/jhu_asp_scripts',

    install_requires=['requests', 'fire', 'moody', 'sh'],

    dependency_links = [
        "git+https://github.com/andrewannex/moody.git"
    ],

    scripts = [
        'asap_stereo/asp_tools/ctx/asp_ctx_lev1eo2dem.sh',
        'asap_stereo/asp_tools/ctx/ctx_pipeline_part_one.sh',
        'asap_stereo/asp_tools/ctx/pedr_bin4pc_align.sh',
        'asap_stereo/asp_tools/ctx/asp_ctx_map_ba_pc_align2dem.sh',
        'asap_stereo/asp_tools/ctx/ctx_pipeline_part_two.sh',
        'asap_stereo/asp_tools/ctx/asp_ctx_step2_map2dem.sh',
        'asap_stereo/asp_tools/ctx/ctxedr2lev1eo.sh',
        'asap_stereo/asp_tools/hirise/asp_hirise_map2dem.sh',
        'asap_stereo/asp_tools/hirise/asp_hirise_prep.sh',
        'asap_stereo/asp_tools/hirise/hirise_pipeline_part_two.sh',
        'asap_stereo/asp_tools/hirise/asp_hirise_pc_align2dem.sh',
        'asap_stereo/asp_tools/hirise/hirise_pipeline_part_one.sh'
    ],

    entry_points={
        'console_scripts': [
            'asap = asap_stereo.asap:main'
        ]
    },

    keywords=['mars', 'nasa', 'asp', 'ames', 'stereo', 'pipeline', 'cli', 'tool'],

    classifiers=[
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: GIS'
    ]
)