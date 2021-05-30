# Change Log   
All notable changes to ASAP-Stereo will be documented here

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project tries to adhere to [Semantic Versioning](http://semver.org/).

## [0.2.0] - 2021-05-30
### Added
- Stereo quality report, adds pvl as dependency
- Downscale parameter to both pipelines for faster processing

### Changed
- Changed 2 stage pipeline to use geodiff for max disparity, simplifying pipeline to single step

### Removed
- removed 2nd pc_align step for both ctx/hirise


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
