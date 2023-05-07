# Change Log   
All notable changes to ASAP-Stereo will be documented here

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project tries to adhere to [Semantic Versioning](http://semver.org/).

## [0.3.1] - 2023-05-07
Bugfix release for dependency updates

## Changed
- sh version now required to be newer than or equal to 2.0.0

### Fixed
- thread parameter for parallel_bundle_adjust, now checks the help to use an available parameter name 
- changes in sh py 2.0.0 that affected camrange parsing, now using pvl only


## [0.3.0] - 2023-03-22
Release with lots of changes due to ISIS and ASP developments.
### Added
- HiPROC workflow
- use of ALE/CSM camera models for ISIS
- compute footprints function to common tools
- drg to cog command to common tools
- lots of caches to github workflows for speed
- better command logging
- full ctx notebook test

### Changed
- all steps use numeric order in names (#28)
- bumped moody version
- assumed ISIS >= 7.1.0
- assumed ASP >= 3.1.0
- Now uses CSM camera models
- simplifcations for pc_align 

### Fixed
- various issues in github CI workflows
- various upgrades for ISIS >= 7.1.0
- various upgrades for ASP >= 3.2.0
- missing affine import 

## [0.2.1] - 2021-06-28
### Added
- Georef module for making GCPs for georegistration and bundle adjustment
- gsd argument for hirise step5 cam2map4stereo.py to allow downscaled hirise
- dois to banners
- more preview images for hirise workflow

### Changed
- bumped moody version

### Fixed
- various typos in hirise workflow

## [0.2.0] - 2021-05-31
### Added
- Stereo quality report functions, adds pvl as dependency
- Downscale parameter to both pipelines for faster processing
- maxdisp estimation using geodiff for ctx and hirise

### Changed
- Changed 2 stage pipeline to use geodiff for max disparity, simplifying pipeline to single step
- Documentation updates for new workflows/install guide
- renamed github workflows

### Removed
- removed 2nd pc_align step for both ctx/hirise, single step pipeline

### Fixed
- fix for negatives in PEDR file names

## [0.1.0] - 2020-10-31
### Added
- Docs!
- Install guide
- CTX walkthrough
- docstrings!
- mola pedr pre step 10

### Fixed
- step twelve postix

### Removed
- asp_scripts bash files

## [0.0.4] - 2020-08-06
### Added
- ci tests for ctx via github action
- info and banner
- rescaling cub function
- webservice enable for ctx spiceinit
- flag to disable high quality pc_align

### Fixed
- bug with par_do 

## [0.0.3] - 2020-06-18
First release of asap_stereo

### Added
- many things

### Changed
- the game

### Deprecated
- nothing important

### Removed
- what had to go

### Fixed
- it
