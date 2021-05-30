# BSD 3-Clause License
#
# Copyright (c) 2020, Andrew Michael Annex
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import fire
import sh
from sh import Command
from contextlib import contextmanager
import functools
import os
import sys
import datetime
import itertools
from typing import Optional, Dict, List, Tuple, Union, Callable
import moody
import re
from pathlib import Path
from threading import Semaphore
import math
import json
import warnings
import papermill as pm

here = os.path.dirname(__file__)

cores = os.cpu_count()
if not cores:
    cores = 16

_threads_singleprocess = cores # 24, 16
_threads_multiprocess  = _threads_singleprocess // 2 if _threads_singleprocess > 1 else 1 # 12, 8
_processes             = _threads_multiprocess // 4 if _threads_multiprocess > 3 else 1 # 3, 2


pool = Semaphore(cores)

def done(cmd, success, exit_code):
    pool.release()

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    try:
        if newdir:
            os.chdir(newdir)
            print(f'cd {newdir}', flush=True)
        yield
    finally:
        os.chdir(prevdir)
        print(f'cd {prevdir}', flush=True)

@contextmanager
def silent_cd(newdir):
    prevdir = os.getcwd()
    try:
        if newdir:
            os.chdir(newdir)
        yield
    finally:
        os.chdir(prevdir)

def cmd_to_string(command: sh.RunningCommand) -> str:
    """
    Converts the running command into a single string of the full command call for easier logging
    
    :param command: a command from sh.py that was run
    :return: string of bash command
    """
    return  " ".join((_.decode("utf-8") for _ in command.cmd))

def clean_args(args):
    return list(itertools.chain.from_iterable([x.split(' ') if isinstance(x, str) else (x,) for x in args]))

def clean_kwargs(kwargs: Dict)-> Dict:
    cleaned = {}
    for key in kwargs.keys():
        new_key = str(key)
        if not key.startswith('--') and len(key) > 1:
            new_key = f'--{key}'
        elif not key.startswith('-'):
            new_key = f'-{key}'
        new_key = new_key.replace('_', '-')
        cleaned[new_key] = kwargs[key]
    return cleaned

def kwargs_to_args(kwargs: Dict)-> List:
    keys = []
    # ensure keys start with '--' for asp scripts
    for key in kwargs.keys():
        key = str(key)
        if key not in ('--t_srs', '--t_projwin'):
            key = key.replace('_', '-')
        if not key.startswith('--') and len(key) > 1:
            keys.append(f'--{key}')
        elif not key.startswith('-'):
            keys.append(f'-{key}')
        else:
            keys.append(key)
    return [x for x in itertools.chain.from_iterable(itertools.zip_longest(keys, kwargs.values())) if x]

def isis3_to_dict(instr: str)-> Dict:
    """
    Given a stdout string from ISIS3, return a Dict version
    
    :param instr:
    :return: dictionary of isis output
    """
    groups = re.findall(r'Group([\S\s]*?)End_Group', instr)
    out = {}
    for group in groups:
        lines = [x.replace('=', '').split() for x in group.split('\\n')]
        group_name = lines[0][0]
        out[group_name] = {t[0]: t[1] for t in lines[1:-1]}
    return out

def rich_logger(func: Callable):
    """
    rich logger decorator, wraps a function and writes nice log statements
    
    :param func: function to wrap
    :return: wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        # grab the function name first
        func_name = func.__name__
        # check if we are running a sh/bash command or a normal python function
        if '/bin/' not in func_name:
            if func.__doc__ is None:
                name_line = f'{func_name}'
            else:
                # grab the first doc line for the pretty name, make sure all functions have docs!
                pretty_name = func.__doc__.splitlines()[1].strip()
                # generate the name line
                name_line = f'{func_name} ({pretty_name})'
        else:
            # else we have a bash command and won't have a pretty name
            name_line = func_name
        # log out the start line with the name line and start time
        print(f"""# Started: {name_line}, at: {start_time.isoformat(" ")}""", flush=True)
        # call the function and get the return
        ret = func(*args, **kwargs)
        # if we had a running command log out the call
        if ret is not None and isinstance(ret, sh.RunningCommand):
            print(f'# Ran Command: {cmd_to_string(ret)}', flush=True)
        # else just get the time duraton
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        # log out the execution time
        print(f"""# Finished: {name_line}, at: {end_time.isoformat(" ")}, duration: {str(duration)}""", flush=True)
        # no return at this point
    return wrapper


def par_do(func, all_calls_args):
    """
    Parallel execution helper function for sh.py Commands
    
    :param func: func to call
    :param all_calls_args: args to pass to each call of the func
    :return: list of called commands
    """
    procs = []

    def do(*args):
        pool.acquire()
        return func(*args, _bg=True, _done=done)

    for call_args in all_calls_args:
        if ' ' in call_args:
            # if we are running a command with multiple args, sh needs different strings
            call_args = call_args.split(' ')
            procs.append(do(*call_args))
        else:
            procs.append(do(call_args))

    return [p.wait() for p in procs]

class CommonSteps(object):
    r"""
    ASAP Stereo Pipeline - Common Commands

    █████████████████████████████████████████████████████████████

              ___   _____ ___    ____
             /   | / ___//   |  / __ \
            / /| | \__ \/ /| | / /_/ /
           / ___ |___/ / ___ |/ ____/
          /_/  |_/____/_/  |_/_/      𝑆 𝑇 𝐸 𝑅 𝐸 𝑂

          asap_stereo (0.2.0)

          Github: https://github.com/AndrewAnnex/asap_stereo

    █████████████████████████████████████████████████████████████
    """

    # defaults for first 3 steps parallel stereo
    defaults_ps1 = {
            '--processes': _processes,
            '--threads-singleprocess': _threads_singleprocess,
            '--threads-multiprocess': _threads_multiprocess,
            '--stop-point': 4,
            '--bundle-adjust-prefix': 'adjust/ba'
        }

    # defaults for first last step parallel stereo (triangulation)
    defaults_ps2 = {
            '--processes'            : _threads_singleprocess, # use more cores for triangulation!
            '--threads-singleprocess': _threads_singleprocess,
            '--threads-multiprocess' : _threads_multiprocess,
            '--entry-point'          : 4,
            '--bundle-adjust-prefix' : 'adjust/ba'
        }

    # default eqc Iau projections, eventually replace with proj4 lookups
    projections = {
        "IAU_Mars": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=3396190 +b=3396190 +units=m +no_defs",
        "IAU_Moon": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs",
        "IAU_Mercury": "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=2439700 +b=2439700 +units=m +no_defs"
    }

    def __init__(self):
        self.parallel_stereo = Command('parallel_stereo').bake(_out=sys.stdout, _err=sys.stderr)
        self.point2dem   = Command('point2dem').bake(_out=sys.stdout, _err=sys.stderr)
        self.pc_align    = Command('pc_align').bake('--save-inv-transform', _out=sys.stdout, _err=sys.stderr)
        self.dem_geoid   = Command('dem_geoid').bake(_out=sys.stdout, _err=sys.stderr)
        self.geodiff     = Command('geodiff').bake('--float', _out=sys.stdout, _err=sys.stderr, _tee=True)
        self.mroctx2isis = Command('mroctx2isis').bake(_out=sys.stdout, _err=sys.stderr)
        self.spiceinit   = Command('spiceinit').bake(_out=sys.stdout, _err=sys.stderr)
        self.spicefit    = Command('spicefit').bake(_out=sys.stdout, _err=sys.stderr)
        self.cubreduce   = Command('reduce').bake(_out=sys.stdout, _err=sys.stderr)
        self.ctxcal      = Command('ctxcal').bake(_out=sys.stdout, _err=sys.stderr)
        self.ctxevenodd  = Command('ctxevenodd').bake(_out=sys.stdout, _err=sys.stderr)
        self.hillshade   = Command('gdaldem').hillshade.bake(_out=sys.stdout, _err=sys.stderr)
        self.mapproject  = Command('mapproject').bake(_out=sys.stdout, _err=sys.stderr)
        try:
            # try to use parallel bundle adjustment
            self.ba = Command('parallel_bundle_adjust').bake(
                '--threads-singleprocess', _threads_singleprocess,
                '--threads-multiprocess', _threads_multiprocess
            )
        except sh.CommandNotFound:
            # if not fall back to regular bundle adjust
            self.ba = Command('bundle_adjust')
        finally:
            self.ba = self.ba.bake('--threads', cores, _out=sys.stdout, _err=sys.stderr)

    @staticmethod
    def get_stereo_quality_report(cub1, cub2) -> str:
        """
        Get the stereo quality report for two cub files
        The cub files must be Level1 images (Spiceinit'ed but not map-projected).

        The quality values reported by this program are based on the
        recommendations and limitations in Becker et al. (2015).  They have
        a value of one for an ideal value, between zero and one for a value
        within the acceptable limits, and less than zero (the more negative,
        the worse) if the value is beyond the acceptable limit.
        # TODO refactor into more granular bits
        :param cub1: path
        :param cub2: path
        :return:
        """
        from .stereo_quality import get_report
        report = get_report(cub1, cub2)
        return report

    @staticmethod
    def get_cam_info(img) -> Dict:
        """
        Get the camera information dictionary from ISIS using camrange

        :param img: path to image as a string
        :return: dictionary of info
        """
        wd = str(Path(img).absolute().parent)
        with silent_cd(wd):
            # currently have to cd into the directory to minify the length
            # of the file name parameter, isis3 inserts additional new lines to wrap
            # words in the terminal that will mess up isis3 to dict without management
            camrange = Command('camrange')
            out = str(camrange(f'from={str(Path(img).name)}').stdout)
            out_dict = isis3_to_dict(out)
        return out_dict

    @staticmethod
    def get_image_gsd(img, opinion='lower')-> float:
        gdalinfocmd = Command('gdalinfo').bake('-json')
        gdal_info = json.loads(str(gdalinfocmd(img)))
        if "geoTransform" in gdal_info:
            transform = gdal_info["geoTransform"]
            res1, res2  = math.fabs(transform[1]), math.fabs(transform[-1])
        else:
            cam_info = CommonSteps.get_cam_info(img)
            if "PixelResolution" not in cam_info:
                raise RuntimeError("Could not find pixel size for input using gdal or camrange. Check if image is valid.")
            res1, res2 = math.fabs(float(cam_info["PixelResolution"]["Lowest"])), math.fabs(float(cam_info["PixelResolution"]["Highest"]))
        if opinion.lower() == 'lower':
            return min(res1, res2)
        elif opinion.lower() == 'higher':
            return max(res1, res2)
        elif opinion.lower() == 'average':
            return (res1+res2)/2
        else:
            raise RuntimeError(f'Opinion {opinion} is not valid, must be "lower" or "higher" or "average".')

    @staticmethod
    def get_srs_info(img, use_eqc: str = None)-> str:
        if use_eqc:
            print(f'Using user provided projection {use_eqc}')
            return use_eqc
        proj4str = sh.gdalsrsinfo(str(img), o='proj4')
        if len(proj4str) <= 10: # arbitrary length picked here
            out_dict = CommonSteps.get_cam_info(img)
            lon = (float(out_dict['UniversalGroundRange']['MinimumLongitude']) + float(out_dict['UniversalGroundRange']['MaximumLongitude'])) / 2
            proj4str = f"+proj=sinu +lon_0={lon} +x_0=0 +y_0=0 +a={out_dict['Target']['RadiusA']} +b={out_dict['Target']['RadiusB']} +units=m +no_defs"
        return str(proj4str).rstrip('\n\' ').lstrip('\'')

    @staticmethod
    def get_map_info(img, key: str, group='UniversalGroundRange')-> str:
        out_dict = CommonSteps.get_cam_info(img)
        return out_dict[group][key]

    @staticmethod
    def parse_stereopairs():
        left, right, both = sh.cat('./stereopairs.lis').strip().split(' ')
        assert both is not None
        return left, right, both

    @staticmethod
    def create_stereopairs_lis():
        left, right, _ = sh.cat('./pair.lis').split('\n')
        with open('./stereopairs.lis', 'w') as out:
            out.write(f'{left} {right} {left}_{right}')

    @staticmethod
    def create_stereodirs_lis():
        with open('./stereodirs.lis', 'w') as out:
            _, _, left_right = CommonSteps.parse_stereopairs()
            out.write(left_right)

    @staticmethod
    def create_sterodirs():
        Path(sh.cat('./stereodirs.lis').strip()).mkdir(exist_ok=True)

    @staticmethod
    def create_stereopair_lis():
        left, right, left_right = CommonSteps.parse_stereopairs()
        with open(f'./{left_right}/stereopair.lis', 'w') as out:
            out.write(f'{left} {right}')

    @staticmethod
    def get_img_crs(img):
        """
        Get CRS of the image

        uses rasterio
        :param img: path to image
        :return: CRX of image
        """
        import rasterio as rio
        with rio.open(img) as i:
            return i.crs

    @staticmethod
    def get_img_bounds(img):
        """
        Get the bounds of the image

        uses rasterio
        :param img: path to image
        :return: bounds tuple
        """
        import rasterio as rio
        with rio.open(img) as i:
            return i.bounds

    def transform_bounds_and_buffer(self, img1, img2, factor=2.0):
        """
        Get bounds of img2 based on centroid of img1 surrounded by a buffer
        the size of the maximum dimension of img1 (scaled by a factor)

        ie if img1 is hirise and img2 is ctx, we find the center point of img1 in img2
        then create a bounding box that is buffered (in radius) by the height of the hirise image
        technically the buffered box would be 2x the height of the hirise which is fine

        :param img1: img to find the bounds in img2 space
        :param img2: crs we are interested in finding the expanded bounds of img1 in
        :param factor: how big we want it (radius is longest dim in img1)
        :return: xmin_img2, ymin_img2, xmax_img2, ymax_img2
        """
        from pyproj import transform
        img1_bounds = self.get_img_bounds(img1)
        img1_crs    = self.get_img_crs(img1)
        img2_crs    = self.get_img_crs(img2)
        # get the buffer radius
        buffer_radius = max((abs(img1_bounds.top-img1_bounds.bottom), abs(img1_bounds.left - img1_bounds.right))) * factor
        # get the centroid of img1
        img1_center = (0.0, (img1_bounds.top + img1_bounds.bottom)/2)
        # transform the centroid
        img1_center_t = transform(img1_crs, img2_crs, *img1_center)
        # use the transformed center to get new xmin ymin xmax ymax
        xmin_img2 = img1_center_t[0] - buffer_radius
        ymin_img2 = img1_center_t[1] - buffer_radius
        xmax_img2 = img1_center_t[0] + buffer_radius
        ymax_img2 = img1_center_t[1] + buffer_radius
        return xmin_img2, ymin_img2, xmax_img2, ymax_img2

    def crop_by_buffer(self, img1, img2, factor=2.0):
        """
        use gdal warp to crop img2 by a buffer around img1

        :param img1: first image defines buffer area
        :param img2: second image is cropped by buffer from first
        :param factor: factor to buffer with
        """
        xmin, ymin, xmax, ymax = self.transform_bounds_and_buffer(img1, img2, factor=factor)
        img2_path = Path(img2).absolute()
        new_name = img2_path.stem + '_clipped.tif'
        return sh.gdalwarp('-te', xmin, ymin, xmax, ymax, img2_path, new_name)

    def check_mpp_against_true_gsd(self, path, mpp):
        """
        Get the GSD of the image, and warn if it is less than 3 * the gsd

        :param path: path to image
        :param mpp: proposed mpp for image
        """
        true_gsd = self.get_image_gsd(path)
        if mpp < true_gsd * 3:
            warnings.warn(f"True image GSD is possibly too big for provided mpp value of {mpp} (compare to 3xGSD={true_gsd * 3})",
                          category=RuntimeWarning)

    @staticmethod
    def get_mpp_postfix(mpp: Union[int, float, str]) -> str:
        """
        get the mpp postfix

        :param mpp: mpp value
        :return: postfix as a string
        """
        return str(float(mpp)).replace('.', '_')

    @rich_logger
    def get_pedr_4_pcalign_w_moody(self, cub_path, proj = None, https=True):
        """
        Python replacement for pedr_bin4pc_align.sh
        that uses moody and the PDS geosciences node REST API

        :param proj: optional projection override
        :param https: optional way to disable use of https
        :param cub_path: path to input file to get query geometry
        """
        cub_path = Path(cub_path).absolute()
        out_name = cub_path.parent.name
        cwd = cub_path.parent
        with cd(cwd):
            out_dict = CommonSteps.get_cam_info(cub_path)['UniversalGroundRange']
            minlon, maxlon, minlat, maxlat = out_dict['MinimumLongitude'], out_dict['MaximumLongitude'], out_dict['MinimumLatitude'], out_dict['MaximumLatitude']
            # use moody to get the pedr in shape file form, we export a csv for what we need to align to
            moody.ODE(https=https).pedr(minlon=float(minlon), minlat=float(minlat), maxlon=float(maxlon), maxlat=float(maxlat), ext='shp')
            shpfile = next(Path.cwd().glob('*z.shp'))
            sql_query = f'SELECT Lat, Lon, Planet_Rad - 3396190.0 AS Datum_Elev, Topography FROM "{shpfile.stem}"'
            # create the minified file just for pc_align
            sh.ogr2ogr('-f', 'CSV', '-sql', sql_query, f'./{out_name}_pedr4align.csv', shpfile.name)
            # get projection info
            projection = self.get_srs_info(cub_path, use_eqc=proj)
            print(projection)
            # reproject to image coordinates for some gis tools
            sh.ogr2ogr('-t_srs', projection, '-sql', sql_query, f'./{out_name}_pedr4align.shp', shpfile.name)

    @rich_logger
    def get_pedr_4_pcalign(self, cub_path, pedr_path, proj):
        """
        Python replacement for pedr_bin4pc_align.sh
        hopefully this will be replaced by a method that queries the mars ODE rest API directly or
        uses a spatial index on the pedr files for speed

        :param cub_path:
        :param pedr_path:
        """
        from string import Template
        from textwrap import dedent
        pedr_path = Path(pedr_path).absolute()
        cub_path = Path(cub_path).absolute()
        out_name = cub_path.parent.name
        cwd = cub_path.parent
        with cd(cwd):
            out_dict = CommonSteps.get_cam_info(cub_path)['UniversalGroundRange']
            minlon, maxlon, minlat, maxlat = out_dict['MinimumLongitude'], out_dict['MaximumLongitude'], out_dict['MinimumLatitude'], out_dict['MaximumLatitude']
            # make the string template
            pedr2tab_template = Template(
                """\
                T # lhdr
                T # 0: shot longitude, latitude, topo, range, planetary_radius,ichan,aflag
                F # 1: MGS_longitude, MGS_latitude, MGS_radius
                F # 2: offnadir_angle, EphemerisTime, areodetic_lat,areoid
                T # 3: ishot, iseq, irev, gravity model number
                F # 4: local_time, solar_phase, solar_incidence
                F # 5: emission_angle, Range_Correction,Pulse_Width_at_threshold,Sigma_optical,E_laser,E_recd,Refl*Trans
                F # 6: bkgrd,thrsh,ipact,ipwct
                F # 7: range_window, range_delay
                F #   All shots, regardless of shot_classification_code
                T # F = noise or clouds, T = ground returns
                T # do crossover correction
                T "${out_name}_pedr.asc" # Name of file to write output to (must be enclosed in quotes).
        
                $minlon     # ground_longitude_min
                $maxlon     # ground_longitude_max
                $minlat      # ground_latitude_min
                $maxlat      # ground_latitude_max
        
                192.    # flattening used for areographic latitude
                """)
            #
            completed_template = pedr2tab_template.safe_substitute(out_name=out_name, minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat)
            # write out the template to a file
            with open('./PEDR2TAB.PRM', 'w') as o:
                print(dedent(completed_template), file=o)
            # copy the pedr list to this directory
            sh.cp('-f', str(pedr_path), './')
            # run pedr2tab using the template
            print('Running pedr2tab, this may take some time...')
            sh.pedr2tab(f'./{pedr_path.name}', _out=f'./{out_name}_pedr2tab.log')
            print('Finished pedr2tab')
            # run a bunch of post processing steps from the script, eventually figure out how to do this within python not using sed and awk
            sh.sed(sh.awk(sh.sed('-e', 's/^ \+//', '-e', 's/ \+/,/g', f"./{out_name}_pedr.asc"), '-F,', 'NR > 2 {print($1","$2","($5 - 3396190)","$1","$2","$10)}'), 's/,/\t/g', _out=f'./{out_name}_pedr.tab')
            # get projection info
            projection = self.get_srs_info(cub_path, use_eqc=proj)
            print(projection)
            # convert using proj, need to split the args for sh!
            proj_tab = sh.proj(*str(projection).split(' '), f'./{out_name}_pedr.tab')
            # write out header
            sh.echo("#Latitude,Longitude,Datum_Elevation,Easting,Northing,Orbit", _out=f'./{out_name}_pedr4align.csv')
            # run through a bunch of steps, please re-write this in python!
            sh.awk(sh.sed(proj_tab, 's/\\t/,/g'),'-F,','{print($5","$4","$3","$1","$2","$6)}', _out=f'./{out_name}_pedr4align.csv')

    @rich_logger
    def bundle_adjust(self, postfix='_RED.map.cub', bundle_adjust_prefix='adjust/ba', **kwargs):
        """
        Bundle adjustment wrapper

        :param postfix: postfix of images to bundle adjust
        :param bundle_adjust_prefix: where to save out bundle adjust results
        :param kwargs: kwargs to pass to bundle_adjust
        :return: RunningCommand
        """
        defaults = {
            '--datum': "D_MARS",
            '--max-iterations': 100
        }
        left, right, both = self.parse_stereopairs()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.ba(f'{left}{postfix}',f'{right}{postfix}', '-o', bundle_adjust_prefix, '--save-cnet-as-csv', *args)

    @rich_logger
    def stereo_1(self, stereo_conf: str, postfix='.lev1eo.cub', **kwargs):
        """
        Step 1 of parallel stereo

        :param postfix:
        :param stereo_conf:
        :param kwargs:
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        stereo_conf = Path(stereo_conf).absolute()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.defaults_ps1, **clean_kwargs(kwargs)})
            return self.parallel_stereo(*args, f'{left}{postfix}', f'{right}{postfix}', '-s', stereo_conf, f'results_ba/{both}_ba')

    @rich_logger
    def stereo_2(self, stereo_conf: str, postfix='.lev1eo.cub', **kwargs):
        """
        Step 2 of parallel stereo

        :param postfix:
        :param stereo_conf:
        :param kwargs:
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        stereo_conf = Path(stereo_conf).absolute()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.defaults_ps2, **clean_kwargs(kwargs)})
            return self.parallel_stereo(*args, f'{left}{postfix}', f'{right}{postfix}', '-s', stereo_conf, f'results_ba/{both}_ba')

    @rich_logger
    def rescale_cub(self, src_file: str, factor=4, overwrite=False, dst_file=None):
        """
        rescale an ISIS3 cub file using the 'reduce' command
        given a factor, optionaly do not overwrite file

        :param src_file: path to src cub file
        :param factor: reduction factor (number [lines, samples] / factor)
        :param overwrite: if true overwrite the src file
        :param dst_file: destination file name, append `rescaled_` if not specified
        """
        c = Path(src_file)
        if not dst_file:
            dst_file = f'{c.with_name("rescaled_"+c.name)}'
        self.cubreduce(f'from={c}', f'to={dst_file}', f'sscale={factor}', f'lscale={factor}')
        if overwrite:
            sh.mv(dst_file, src_file)

    def rescale_and_overwrite(self, factor, postfix='.lev1eo.cub'):
        """
        Rescale the left and right images

        :param factor: factor to reduce each dimension by
        :param postfix: file postfix
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            self.rescale_cub(f'{left}{postfix}', factor=factor, overwrite=True)
            self.rescale_cub(f'{right}{postfix}', factor=factor, overwrite=True)

    def get_pedr_4_pcalign_common(self, postfix, proj, https, pedr_list=None):
        left, right, both = self.parse_stereopairs()
        with cd(Path.cwd() / both):
            if pedr_list:
                self.get_pedr_4_pcalign(f'{left}{postfix}.cub', pedr_list, proj)
            else:
                self.get_pedr_4_pcalign_w_moody(f'{left}{postfix}.cub', proj=proj, https=https)

    def get_geo_diff(self, ref_dem, src_dem=None):
        left, right, both = self.parse_stereopairs()
        ref_dem = Path(ref_dem).absolute()
        with cd(Path.cwd() / both):
            args = []
            if not src_dem:
                src_dem = next(Path.cwd().glob('*_pedr4align.csv'))
            src_dem = str(src_dem)
            if src_dem.endswith('.csv'):
                args.extend(['--csv-format', '1:lat 2:lon 3:height_above_datum'])
            res = self.geodiff(*args, ref_dem, src_dem)
            res = str(res).splitlines()
            res = {k.strip(): v.strip() for k, v in [l.split(':') for l in res]}
            return res
    
    def estimate_max_disparity(self, ref_dem, src_dem=None):
        """
        Estimate the absolute value of the maximum observed displacement
        between two point clouds, and the standard deviation of the differences
        
        if not applying an initial transform to pc_align, use the max_d value 
        if expecting to apply a transform first and you are 
        interested in the maximum displacement after an initial transform, then
        use the std_d returned (likely 3X it)
        """
        vals = self.get_geo_diff(ref_dem, src_dem)
        max_d = float(vals['Max difference'])
        min_d = float(vals['Min difference'])
        std_d = float(vals['StdDev of difference'])
        absmax_d = max(abs(max_d), abs(min_d))
        return absmax_d, max_d, min_d, std_d
    
    def estimate_median_disparity(self, ref_dem, src_dem=None):
        vals = self.get_geo_diff(ref_dem, src_dem)
        med_d = float(vals['Median difference'])       
        return med_d, abs(med_d)
        


class CTX(object):
    r"""
    ASAP Stereo Pipeline - CTX workflow

    █████████████████████████████████████████████████████████████

              ___   _____ ___    ____
             /   | / ___//   |  / __ \
            / /| | \__ \/ /| | / /_/ /
           / ___ |___/ / ___ |/ ____/
          /_/  |_/____/_/  |_/_/      𝑆 𝑇 𝐸 𝑅 𝐸 𝑂

          asap_stereo (0.2.0)

          Github: https://github.com/AndrewAnnex/asap_stereo

    █████████████████████████████████████████████████████████████
    """

    def __init__(self, https=False, datum="D_MARS", proj: Optional[str] = None):
        self.cs = CommonSteps()
        self.https = https
        self.datum = datum
        # if proj is not none, get the corresponding proj or else override with proj,
        # otherwise it's a none so remain a none
        self.proj = CommonSteps.projections.get(proj, proj)

    def get_full_ctx_id(self, pid):
        res = str(moody.ODE(self.https).get_ctx_meta_by_key(pid, 'ProductURL'))
        return res.split('=')[1].split('&')[0]

    def get_ctx_emission_angle(self, pid):
        return float(moody.ODE(self.https).get_ctx_meta_by_key(pid, 'Emission_angle'))

    def get_ctx_order(self, one, two):
        em_one = self.get_ctx_emission_angle(one)
        em_two = self.get_ctx_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def generate_ctx_pair_list(self, one, two):
        order = self.get_ctx_order(one, two)
        full_ids = [self.get_full_ctx_id(pid) for pid in order]
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in full_ids:
                o.write(pid)
                o.write('\n')

    @staticmethod
    def notebook_pipeline_make_dem(left: str, right: str, config1: str, pedr_list: str = None, downsample: int = None, working_dir ='./', 
                                   config2: Optional[str] = None, dem_gsd = 24.0, img_gsd = 6.0, maxdisp = None, out_notebook=None, **kwargs):
        """
        First step in CTX DEM pipeline that uses papermill to persist log

        this command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command
        
        :param out_notebook: output notebook log file name, defaults to log_asap_notebook_pipeline_make_dem.ipynb
        :param config2: ASP config file to use for second processing pass
        :param working_dir: Where to execute the processing, defaults to current directory
        :param config1: ASP config file to use for first processing pass
        :param pedr_list: Path to PEDR files, defaults to None to use ODE Rest API
        :param left: First image id
        :param right: Second image id
        :param maxdisp: Maximum expected displacement in meters, use None to determine it automatically 
        :param downsample: Factor to downsample images for faster production
        :param dem_gsd: desired GSD of output DEMs (4x image GSD)
        :param img_gsd: desired GSD of output ortho images
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem.ipynb'
        pm.execute_notebook(
            f'{here}/asap_ctx_workflow.ipynb',
            out_notebook,
            parameters={
                'left' : left,
                'right': right,
                'peder_list': pedr_list,
                'config1': config1,
                'config2': config2,
                'output_path' : working_dir,
                'maxdisp': maxdisp,
                'dem_gsd' : dem_gsd,
                'img_gsd' : img_gsd,
                'downsample' : downsample,
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @rich_logger
    def step_one(self, one: str, two: str, cwd: Optional[str] = None) -> None:
        """
        Download CTX EDRs from the PDS

        :param one: first CTX image id
        :param two: second CTX image id
        :param cwd:
        """
        with cd(cwd):
            self.generate_ctx_pair_list(one, two)
            # download files
            moody.ODE(self.https).ctx_edr(one)
            moody.ODE(self.https).ctx_edr(two)

    @rich_logger
    def step_two(self, with_web=False):
        """
        ISIS3 CTX preprocessing, replaces ctxedr2lev1eo.sh

        :param with_web: if true attempt to use webservices for SPICE kernel data
        """
        imgs = [*Path.cwd().glob('*.IMG'), *Path.cwd().glob('*.img')]
        par_do(self.cs.mroctx2isis, [f'from={i.name} to={i.stem}.cub' for i in imgs])
        cubs = list(Path.cwd().glob('*.cub'))
        par_do(self.cs.spiceinit, [f'from={c.name}{" web=yes" if with_web else ""}' for c in cubs])
        par_do(self.cs.spicefit, [f'from={c.name}' for c in cubs])
        par_do(self.cs.ctxcal, [f'from={c.name} to={c.stem}.lev1.cub' for c in cubs])
        for cub in cubs:
            cub.unlink()
        lev1cubs = list(Path.cwd().glob('*.lev1.cub'))
        par_do(self.cs.ctxevenodd, [f'from={c.name} to={c.stem}eo.cub' for c in lev1cubs])
        for lc in lev1cubs:
            lc.unlink()

    @rich_logger
    def step_three(self):
        """
        Create various processing files for future steps
        """
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_sterodirs()
        self.cs.create_stereopair_lis()
        # copy the cub files into the both directory
        _, _, both = self.cs.parse_stereopairs()
        return sh.mv('-n', sh.glob('./*.cub'), f'./{both}/')

    @rich_logger
    def step_four(self, bundle_adjust_prefix='adjust/ba', **kwargs)-> sh.RunningCommand:
        """
        Bundle Adjust CTX

        Run bundle adjustment on the CTX map projected data

        :param bundle_adjust_prefix: prefix for bundle adjust output
        """
        return self.cs.bundle_adjust(postfix='.lev1eo.cub', bundle_adjust_prefix=bundle_adjust_prefix, **kwargs)

    @rich_logger
    def step_five(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 1

        Run first part of parallel_stereo asp_ctx_lev1eo2dem.sh
        """
        return self.cs.stereo_1(stereo_conf, postfix='.lev1eo.cub', **kwargs)

    @rich_logger
    def step_six(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 2

        Run second part of parallel_stereo, asp_ctx_lev1eo2dem.sh stereo is completed after this step
        """
        return self.cs.stereo_2(stereo_conf, postfix='.lev1eo.cub', **kwargs)

    @rich_logger
    def step_seven(self, mpp=24, just_dem=False, folder='results_ba', **kwargs):
        """
        Produce preview DEMs/Orthos

        Produce dem from point cloud, by default 24mpp for ctx for max-disparity estimation

        :param folder:
        :param just_dem: set to True if you only want the DEM and no other products like the ortho and error images
        :param mpp:
        """
        left, right, both = self.cs.parse_stereopairs()
        mpp_postfix = self.cs.get_mpp_postfix(mpp)
        post_args = []
        if not just_dem:
            left_image = next((Path.cwd() / both / folder).glob('*L.tif'))
            post_args.extend(['-n', '--errorimage', '--orthoimage', left_image.name])
        with cd(Path.cwd() / both / folder):
            # check the GSD against the MPP
            self.cs.check_mpp_against_true_gsd(f'../{left}.lev1eo.cub', mpp)
            # get the projection info
            proj     = self.cs.get_srs_info(f'../{left}.lev1eo.cub', use_eqc=self.proj)
            defaults = {
                '--t_srs'      : proj,
                '-r'           : 'mars',
                '--dem-spacing': mpp,
                '--nodata'     : -32767,
                '--output-prefix': f'dem/{both}_ba_{mpp_postfix}',
            }
            pre_args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.point2dem(*pre_args, f'{both}_ba-PC.tif', *post_args)

    @rich_logger
    def step_eight(self, folder='results_ba', output_folder='dem'):
        """
        hillshade First step in asp_ctx_step2_map2dem script

        :param output_folder:
        :param folder:
        :param mpp:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / folder / output_folder):
            dem = next(Path.cwd().glob('*DEM.tif'))
            self.cs.hillshade(dem.name, f'./{dem.stem}-hillshade.tif')

    @rich_logger
    def step_nine(self, refdem=None, mpp=6):
        """
        Mapproject the left and right ctx images against the reference DEM

        :param refdem: reference dem to map project using
        :param mpp: target GSD
        """
        left, right, both = self.cs.parse_stereopairs()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        else:
            refdem = Path(refdem).absolute()
        with cd(Path.cwd() / both):
            # double check provided gsd
            self.cs.check_mpp_against_true_gsd(f'{left}.lev1eo.cub', mpp)
            self.cs.check_mpp_against_true_gsd(f'{right}.lev1eo.cub', mpp)
            # map project both ctx images against the reference dem
            # might need to do par do here
            self.cs.mapproject('-t', 'isis', refdem, f'{left}.lev1eo.cub', f'{left}.ba.map.tif', '--mpp', mpp, '--bundle-adjust-prefix', 'adjust/ba')
            self.cs.mapproject('-t', 'isis', refdem, f'{right}.lev1eo.cub', f'{right}.ba.map.tif', '--mpp', mpp, '--bundle-adjust-prefix', 'adjust/ba')

    @rich_logger
    def step_ten(self, stereo_conf, refdem=None, **kwargs):
        """
        Second stereo first step

        :param stereo_conf:
        :param refdem: path to reference DEM or PEDR csv file
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        stereo_conf = Path(stereo_conf).absolute()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        else:
            refdem = Path(refdem).absolute()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.cs.defaults_ps1, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}.ba.map.tif', f'{right}.ba.map.tif', f'{left}.lev1eo.cub', f'{right}.lev1eo.cub',
                                           '-s', stereo_conf, f'results_map_ba/{both}_ba', refdem)

    @rich_logger
    def step_eleven(self, stereo_conf, refdem=None, **kwargs):
        """
        Second stereo second step

        :param stereo_conf:
        :param refdem: path to reference DEM or PEDR csv file
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        stereo_conf = Path(stereo_conf).absolute()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        else:
            refdem = Path(refdem).absolute()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.cs.defaults_ps2, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}.ba.map.tif', f'{right}.ba.map.tif', f'{left}.lev1eo.cub', f'{right}.lev1eo.cub',
                                           '-s', stereo_conf, f'results_map_ba/{both}_ba', refdem)

    @rich_logger
    def step_twelve(self, pedr_list=None, postfix='.lev1eo'):
        """
        Get MOLA PEDR data to align the CTX DEM to

        :param postfix: postfix for file, minus extension
        :param pedr_list: path local PEDR file list, default None to use REST API
        """
        self.cs.get_pedr_4_pcalign_common(postfix, self.proj, self.https, pedr_list=pedr_list)

    @rich_logger
    def step_thirteen(self, maxd: float = None, pedr4align = None, highest_accuracy = True, **kwargs):
        """
        PC Align CTX

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs
        
        #TODO: use the DEMs instead of the point clouds

        :param highest_accuracy: Use the maximum accuracy mode
        :param maxd: Maximum expected displacement in meters
        :param pedr4align: path to pedr csv file
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        if not pedr4align:
            pedr4align = str(Path.cwd() / both / f'{both}_pedr4align.csv')
        if not maxd:
            dem = next((Path.cwd() / both / 'results_map_ba' / 'dem').glob(f'{both}*DEM.tif'))
            # todo implement a new command or path to do a initial NED translation with this info
            maxd, _, _, _ = self.cs.estimate_max_disparity(dem, pedr4align)
        defaults = {
            '--num-iterations': 4000,
            '--threads': _threads_singleprocess,
            '--datum'  : self.datum,
            '--max-displacement': maxd,
            '--output-prefix': f'dem_align/{both}_map_ba_align'
        }
        with cd(Path.cwd() / both / 'results_map_ba'):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            hq = ['--highest-accuracy'] if highest_accuracy else []
            return self.cs.pc_align(*args, *hq, f'{both}_ba-PC.tif', pedr4align)

    @rich_logger
    def step_fourteen(self, mpp=24.0, just_ortho=False, output_folder='dem_align', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products

        :param mpp:
        :param just_ortho:
        :param output_folder:
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        gsd_postfix = str(float(mpp)).replace('.', '_')
        add_params = []
        if just_ortho:
            add_params.append('--no-dem')
        else:
            add_params.extend(['-n', '--errorimage',])

        with cd(Path.cwd() / both / 'results_map_ba'):
            proj     = self.cs.get_srs_info(f'../{left}.lev1eo.cub', use_eqc=self.proj)
            if not just_ortho:
                # check the GSD against the MPP
                self.cs.check_mpp_against_true_gsd(f'../{left}.lev1eo.cub', mpp)
            with cd(output_folder):
                point_cloud = next(Path.cwd().glob('*trans_reference.tif'))
                defaults = {
                    '--t_srs'         : proj,
                    '-r'              : 'mars',
                    '--nodata'        : -32767,
                    '--orthoimage'    : f'../{both}_ba-L.tif',
                    '--output-prefix' : f'{both}_map_ba_align_{gsd_postfix}',
                    '--dem-spacing'   : mpp
                }
                args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
                return self.cs.point2dem(*args, str(point_cloud.name), *add_params)

    @rich_logger
    def step_fifteen(self, output_folder='dem_align', **kwargs):
        """
        Adjust DEM to geoid

        Run geoid adjustment on dem for final science ready product
        :param output_folder:
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / 'results_map_ba' / output_folder):
            file = next(Path.cwd().glob('*-DEM.tif'))
            args = kwargs_to_args(clean_kwargs(kwargs))
            return self.cs.dem_geoid(*args, file, '-o', f'{file.stem}')

class HiRISE(object):
    r"""
    ASAP Stereo Pipeline - HiRISE workflow

    █████████████████████████████████████████████████████████████

              ___   _____ ___    ____
             /   | / ___//   |  / __ \
            / /| | \__ \/ /| | / /_/ /
           / ___ |___/ / ___ |/ ____/
          /_/  |_/____/_/  |_/_/      𝑆 𝑇 𝐸 𝑅 𝐸 𝑂

          asap_stereo (0.2.0)

          Github: https://github.com/AndrewAnnex/asap_stereo

    █████████████████████████████████████████████████████████████
    """

    def __init__(self, https=False, threads=cores, datum="D_MARS", proj: Optional[str] = None):
        self.https = https
        self.cs = CommonSteps()
        self.threads = threads
        self.datum = datum
        self.hiedr = sh.Command('hiedr2mosaic.py')
        self.cam2map = sh.Command('cam2map')
        self.cam2map4stereo = sh.Command('cam2map4stereo.py')
        # if proj is not none, get the corresponding proj or else override with proj,
        # otherwise it's a none so remain a none
        self.proj = CommonSteps.projections.get(proj, proj)

    def get_hirise_emission_angle(self, pid: str)-> float:
        """
        Use moody to get the emission angle of the provided HiRISE image id

        :param pid: HiRISE image id
        :return: emission angle
        """

        return float(moody.ODE(self.https).get_hirise_meta_by_key(f'{pid}_R*', 'Emission_angle'))

    def get_hirise_order(self, one: str, two: str) -> Tuple[str, str]:
        """
        Get the image ids sorted by lower emission angle

        :param one: first image
        :param two: second image
        :return: tuple of sorted images
        """

        em_one = self.get_hirise_emission_angle(one)
        em_two = self.get_hirise_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def generate_hirise_pair_list(self, one, two):
        """
        Generate the hirise pair.lis file for future steps

        :param one: first image id
        :param two: second image id
        """
        order = self.get_hirise_order(one, two)
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in order:
                o.write(pid)
                o.write('\n')

    @staticmethod
    def notebook_pipeline_make_dem(left: str, right: str, config: str, refdem: str, maxdisp: float = None, downsample: int = None, demgsd: float = 1.0, imggsd: float = 0.25, alignment_method = 'rigid', working_dir ='./', out_notebook=None, **kwargs):
        """
        First step in HiRISE DEM pipeline that uses papermill to persist log

        This command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command, even more so for HiRISE!

        :param out_notebook: output notebook log file name, defaults to log_asap_notebook_pipeline_make_dem_hirise.ipynb
        :param working_dir: Where to execute the processing, defaults to current directory
        :param config:  ASP config file to use for processing
        :param left: first image id
        :param right: second image id
        :param alignment_method: alignment method to use for pc_align
        :param downsample: Factor to downsample images for faster production
        :param refdem: path to reference DEM or PEDR csv file
        :param maxdisp: Maximum expected displacement in meters, specify none to determine it automatically
        :param demgsd: desired GSD of output DEMs (4x image GSD)
        :param imggsd: desired GSD of output ortho images
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem_hirise.ipynb'
        pm.execute_notebook(
            f'{here}/asap_hirise_workflow.ipynb',
            out_notebook,
            parameters={
                'left' : left,
                'right': right,
                'config': config,
                'output_path' : working_dir,
                'refdem'          : refdem,
                'maxdisp'         : maxdisp,
                'demgsd'          : demgsd,
                'imggsd'          : imggsd,
                'alignment_method': alignment_method,
                'downsample': downsample,
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @rich_logger
    def step_one(self, one, two, cwd: Optional[str] = None):
        """
        Download HiRISE EDRs

        Download two HiRISE images worth of EDR files to two folders

        :param one: first image id
        :param two: second image id
        :param cwd:
        """
        with cd(cwd):
            self.generate_hirise_pair_list(one, two)
            # download files
            Path(one).mkdir(exist_ok=True)
            with cd(one):
                moody.ODE(self.https).hirise_edr(f'{one}_R*')

            Path(two).mkdir(exist_ok=True)
            with cd(two):
                moody.ODE(self.https).hirise_edr(f'{two}_R*')

    @rich_logger
    def step_two(self):
        """
        Metadata init

        Create various files with info for later steps

        """
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_sterodirs()
        self.cs.create_stereopair_lis()

    @rich_logger
    def step_three(self):
        """
        Hiedr2mosaic preprocessing

        Run hiedr2mosaic on all the data

        """

        def hiedr2mosaic(*im):
            # hiedr2moasic is given a glob of tifs
            pool.acquire()
            return self.hiedr(*im, '--threads', self.threads, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(Path(left)):
            procs.append(hiedr2mosaic(*list(Path('./').glob('*.IMG'))))
        with cd(Path(right)):
            procs.append(hiedr2mosaic(*list(Path('./').glob('*.IMG'))))
        _ = [p.wait() for p in procs]

    @rich_logger
    def step_four(self):
        """
        Move hieder2mosaic files

        Move the hiedr2mosaic output to the location needed for cam2map4stereo

        """
        left, right, both = self.cs.parse_stereopairs()
        sh.mv(next(Path(f'./{left}/').glob(f'{left}*.mos_hijitreged.norm.cub')), both)
        sh.mv(next(Path(f'./{right}/').glob(f'{right}*.mos_hijitreged.norm.cub')), both)

    @rich_logger
    def step_five(self):
        """
        Map project HiRISE data for stereo processing

        Run cam2map4stereo on the data

        """

        def par_cam2map(argstring):
            pool.acquire()
            return self.cam2map(argstring.split(' '), _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        with cd(both):
            left_im  = next(Path('.').glob(f'{left}*.mos_hijitreged.norm.cub'))
            right_im = next(Path('.').glob(f'{right}*.mos_hijitreged.norm.cub'))
            response = str(self.cam2map4stereo('-n', left_im, right_im))
            left_cam2map_call, right_cam2map_call = response.split('\n')[-4:-2]
            print(left_cam2map_call, flush=True)
            print(right_cam2map_call, flush=True)
            # double check to make sure we got the right lines, maybe replace above with regex sometime
            if 'cam2map from=' not in left_cam2map_call or 'cam2map from=' not in right_cam2map_call:
                raise RuntimeError(f'Got bad call responses for cam2map from cam2map4stereo.py, \n see left_cam2map_call: {left_cam2map_call}, right_cam2map_call: {right_cam2map_call}')
            # we are good now, call the cam2map simultaneously
            left_cam2map_call  =  left_cam2map_call.strip('cam2map ')
            right_cam2map_call = right_cam2map_call.strip('cam2map ')
            procs = [par_cam2map(left_cam2map_call), par_cam2map(right_cam2map_call)]
            _ = [p.wait() for p in procs]

    @rich_logger
    def step_six(self, bundle_adjust_prefix='adjust/ba', **kwargs)-> sh.RunningCommand:
        """
        Bundle Adjust HiRISE

        Run bundle adjustment on the HiRISE map projected data

        :param bundle_adjust_prefix:
        """
        return self.cs.bundle_adjust(postfix='_RED.map.cub', bundle_adjust_prefix=bundle_adjust_prefix, **kwargs)

    @rich_logger
    def step_seven(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 1

        Run first part of parallel_stereo
        """
        return self.cs.stereo_1(stereo_conf, postfix='_RED.map.cub', **kwargs)

    @rich_logger
    def step_eight(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 2

        Run second part of parallel_stereo, stereo is completed after this step
        """
        return self.cs.stereo_2(stereo_conf, postfix='_RED.map.cub', **kwargs)

    @rich_logger
    def step_nine(self, mpp=2, just_dem=False, **kwargs):
        """
        Produce preview DEMs/Orthos

        Produce dem from point cloud, by default 2mpp for hirise for max-disparity estimation

        :param just_dem: set to True if you only want the DEM and no other products like the ortho and error images
        :param mpp:
        """
        left, right, both = self.cs.parse_stereopairs()
        mpp_postfix = self.cs.get_mpp_postfix(mpp)
        post_args = []
        if not just_dem:
            post_args.extend(['-n', '--errorimage', '--orthoimage', f'{both}_ba-L.tif'])
        with cd(Path.cwd() / both / 'results_ba'):
            # check the GSD against the MPP
            self.cs.check_mpp_against_true_gsd(f'../{left}_RED.map.cub', mpp)
            proj     = self.cs.get_srs_info(f'../{left}_RED.map.cub', use_eqc=self.proj)
            defaults = {
                '--t_srs'      : proj,
                '-r'           : 'mars',
                '--dem-spacing': mpp,
                '--nodata'     : -32767,
                '--output-prefix': f'dem/{both}_{mpp_postfix}',
            }
            pre_args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.point2dem(*pre_args, f'{both}_ba-PC.tif', *post_args)

    def _gdal_hirise_rescale(self, mpp):
        """
        Hillshade using gdaldem instead of asp

        :param mpp:
        """
        left, right, both = self.cs.parse_stereopairs()
        mpp_postfix = self.cs.get_mpp_postfix(mpp)
        with cd(Path.cwd() / both / 'results_ba' / 'dem'):
            # check the GSD against the MPP
            self.cs.check_mpp_against_true_gsd(f'../../{left}_RED.map.cub', mpp)
            in_dem = next(Path.cwd().glob('*-DEM.tif')) # todo: this might not always be the right thing to do...
            return sh.gdal_translate('-r', 'cubic', '-tr', float(mpp), float(mpp), in_dem, f'./{both}_{mpp_postfix}-DEM.tif')

    @rich_logger
    def pre_step_ten(self, refdem, alignment_method='translation', do_resample='gdal', **kwargs):
        """
        Hillshade Align before PC Align

        Automates the procedure to use ipmatch on hillshades of downsampled HiRISE DEM
        to find an initial transform

        :param do_resample:  can be: 'gdal' or 'asp' or anything else for no resampling
        :param alignment_method: can be 'similarity' 'rigid' or 'translation'
        :param refdem: path to reference DEM or PEDR csv file
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        defaults = {
            '--max-displacement'                  : -1,
            '--num-iterations'                    : '0',
            '--ipmatch-options'                   : '--debug-image',
            '--ipfind-options'                    : '--ip-per-image 2000000 --ip-per-tile 4000 --interest-operator sift --descriptor-generator sift --debug-image 2',
            '--threads'                           : self.threads,
            '--initial-transform-from-hillshading': alignment_method,
            '--datum'                             : self.datum,
            '--output-prefix'                     : 'hillshade_align/out'
        }
        refdem = Path(refdem).absolute()
        refdem_mpp = math.ceil(self.cs.get_image_gsd(refdem))
        refdem_mpp_postfix = self.cs.get_mpp_postfix(refdem_mpp)
        # create the lower resolution hirise dem to match the refdem gsd
        if do_resample.lower() == 'asp':
            # use the image in a call to pc_align with hillshades, slow!
            self.step_nine(mpp=refdem_mpp, just_dem=True)
        elif do_resample.lower() == 'gdal':
            # use gdal translate to resample hirise dem down to needed resolution
            self._gdal_hirise_rescale(refdem_mpp)
        else:
            print('Not resampling HiRISE per user request')
        #TODO: auto crop the reference dem to be around hirise more closely
        with cd(Path.cwd() / both / 'results_ba'):
            lr_hirise_dem = Path.cwd() / 'dem' / f'{both}_{refdem_mpp_postfix}-DEM.tif'
            args    = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            cmd_res = self.cs.pc_align(*args, lr_hirise_dem, refdem)
            # done! log out to user that can use the transform
        out_dir = Path.cwd() / both / 'results_ba' / 'hillshade_align'
        print(f"Completed Pre step nine, view output in {str(out_dir)}", flush=True)
        print(f"Use transform: 'hillshade_align/out-transform.txt'", flush=True)
        print("as initial_transform argument in step ten", flush=True)
        return cmd_res

    @rich_logger
    def pre_step_ten_pedr(self, pedr_list=None, postfix='_RED.map.cub'):
        """
        Use MOLA PEDR data to align the HiRISE DEM to in case no CTX DEM is available

        :param pedr_list: path local PEDR file list, default None to use REST API
        :param postfix: postfix for the file
        """
        self.cs.get_pedr_4_pcalign_common(postfix, self.proj, self.https, pedr_list=pedr_list)

    @rich_logger
    def step_ten(self, maxd, refdem, highest_accuracy=True, **kwargs):
        """
        PC Align HiRISE

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs

        :param maxd: Maximum expected displacement in meters
        :param refdem: path to reference DEM or PEDR csv file
        :param highest_accuracy: use highest precision alignment (more memory and cpu intensive)
        :param kwargs: kwargs to pass to pc_align, use to override ASAP defaults
        """
        left, right, both = self.cs.parse_stereopairs()
        if not maxd:
            dem = next((Path.cwd() / both / 'results_ba' / 'dem').glob(f'{both}*DEM.tif'))
            # todo implement a new command or path to do a initial NED translation with this info
            maxd, _, _, _  = self.cs.estimate_max_disparity(dem, refdem)
        
        defaults = {
            '--num-iterations': 2000,
            '--threads': self.threads,
            '--datum'  : self.datum,
            '--max-displacement': maxd,
            '--output-prefix': f'dem_align/{both}_align'
        }
        refdem = Path(refdem).absolute()
        with cd(Path.cwd() / both):
            with cd('results_ba'):
                args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
                hq = ['--highest-accuracy'] if highest_accuracy else []
                return self.cs.pc_align(*args, *hq, f'{both}_ba-PC.tif', refdem)

    @rich_logger
    def step_eleven(self, mpp=1.0, just_ortho=False, output_folder='dem_align', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products

        :param mpp: Desired GSD (meters per pixel)
        :param just_ortho: if True, just render out the ortho images
        :param output_folder: output folder name
        :param kwargs: any other kwargs you want to pass to point2dem
        """
        left, right, both = self.cs.parse_stereopairs()
        gsd_postfix = str(float(mpp)).replace('.', '_')
        add_params = []
        if just_ortho:
            add_params.append('--no-dem')
        else:
            add_params.extend(['-n', '--errorimage',])

        with cd(Path.cwd() / both / 'results_ba'):
            proj     = self.cs.get_srs_info(f'../{left}_RED.map.cub', use_eqc=self.proj)
            if not just_ortho:
                # check the GSD against the MPP
                self.cs.check_mpp_against_true_gsd(f'../{left}_RED.map.cub', mpp)
            with cd(output_folder):
                point_cloud = next(Path.cwd().glob('*trans_reference.tif'))
                defaults = {
                    '--t_srs'         : proj,
                    '-r'              : 'mars',
                    '--nodata'        : -32767,
                    '--orthoimage'    : f'../{both}_ba-L.tif',
                    '--output-prefix' : f'{both}_align_{gsd_postfix}',
                    '--dem-spacing'   : mpp
                }
                args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
                return self.cs.point2dem(*args, str(point_cloud.name), *add_params)

    @rich_logger
    def step_twelve(self, output_folder='dem_align', **kwargs):
        """
        Adjust DEM to geoid

        Run geoid adjustment on dem for final science ready product

        :param output_folder:
        :param kwargs:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / 'results_ba' / output_folder):
            file = next(Path.cwd().glob('*-DEM.tif'))
            args = kwargs_to_args(clean_kwargs(kwargs))
            return self.cs.dem_geoid(*args, file, '-o', f'{file.stem}')


class ASAP(object):
    r"""
    ASAP Stereo Pipeline

    █████████████████████████████████████████████████████████████
                                                                 
              ___   _____ ___    ____                            
             /   | / ___//   |  / __ \                           
            / /| | \__ \/ /| | / /_/ /                           
           / ___ |___/ / ___ |/ ____/                            
          /_/  |_/____/_/  |_/_/      𝑆 𝑇 𝐸 𝑅 𝐸 𝑂               

          asap_stereo (0.2.0)

          Github: https://github.com/AndrewAnnex/asap_stereo

    █████████████████████████████████████████████████████████████
    """

    def __init__(self, https=False, datum="D_MARS"):
        self.https  = https
        self.hirise = HiRISE(self.https, datum=datum)
        self.ctx    = CTX(self.https, datum=datum)
        self.common = CommonSteps()
        self.get_srs_info = self.common.get_srs_info
        self.get_map_info = self.common.get_map_info

    def ctx_one(self, left, right, cwd: Optional[str] = None):
        """
        Run first stage of CTX pipeline

        This command runs steps 1-3 of the CTX pipeline

        :param left: left image id
        :param right: right image id
        :param cwd: directory to run process within (default to CWD)
        """
        with cd(cwd):
            self.ctx.step_one(left, right)
            # ctxedr2lev1eo steps
            self.ctx.step_two()
            # move things
            self.ctx.step_three()

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
        """
        Run Second stage of CTX pipeline

        This command runs steps 4-12 of the CTX pipeline

        :param stereo: ASP stereo config file to use
        :param pedr_list: Path to PEDR files, defaults to None to use ODE Rest API
        :param stereo2: 2nd ASP stereo config file to use, if none use first stereo file again
        :param cwd: directory to run process within (default to CWD)
        """
        with cd(cwd):
            self.ctx.step_four()
            self.ctx.step_five(stereo)
            self.ctx.step_six(stereo)
            # asp_ctx_step2_map2dem steps
            self.ctx.step_seven(mpp=100, just_dem=True, dem_hole_fill_len=50)
            self.ctx.step_eight()
            self.ctx.step_nine()
            self.ctx.step_ten(stereo2 if stereo2 else stereo)
            self.ctx.step_eleven(stereo2 if stereo2 else stereo)
            self.ctx.step_seven(folder='results_map_ba')
            self.ctx.step_eight(folder='results_map_ba')
            self.ctx.step_twelve(pedr_list)

    def ctx_three(self, max_disp: float = None, demgsd: float = 24, imggsd: float = 6, cwd: Optional[str] = None, **kwargs) -> None:
        """
        Run third and final stage of the CTX pipeline

        This command runs steps 13-15 of the CTX pipeline

        :param max_disp: Maximum expected displacement in meters
        :param demgsd: GSD of final Dem, default is 1 mpp
        :param imggsd: GSD of full res image
        :param cwd: directory to run process within (default to CWD)
        :param kwargs:
        """
        with cd(cwd):
            self.ctx.step_thirteen(max_disp, **kwargs)
            self.ctx.step_fourteen(mpp=demgsd, **kwargs)
            self.ctx.step_fifteen(**kwargs)
            # go back and make final orthos and such
            self.ctx.step_fourteen(mpp=imggsd, just_ortho=True, **kwargs)
            self.ctx.step_eight(folder='results_map_ba', output_folder='dem_align')

    def hirise_one(self, left, right):
        """
        Download the EDR data from the PDS, requires two HiRISE Id's
        (order left vs right does not matter)

        This command runs step 1 of the HiRISE pipeline

        :param left: HiRISE Id
        :param right: HiRISE Id
        """
        self.hirise.step_one(left, right)

    def hirise_two(self, stereo, mpp=2, bundle_adjust_prefix='adjust/ba', max_iterations=50) -> None:
        """
        Run various calibration steps then:
        bundle adjust, produce DEM, render low res version for inspection
        This will take a while (sometimes over a day), use nohup!

        This command runs steps 2-9 of the HiRISE pipeline

        :param stereo: ASP stereo config file to use
        :param mpp: preview DEM GSD, defaults to 2 mpp
        :param bundle_adjust_prefix: bundle adjust prefix, defaults to 'adjust/ba'
        :param max_iterations: number of iterations for HiRISE bundle adjustment, defaults to 50
        """
        self.hirise.step_two()
        self.hirise.step_three()
        self.hirise.step_four()
        self.hirise.step_five()
        self.hirise.step_six(bundle_adjust_prefix=bundle_adjust_prefix, max_iterations=max_iterations)
        self.hirise.step_seven(stereo)
        self.hirise.step_eight(stereo)
        self.hirise.step_nine(mpp=mpp)

    def hirise_three(self, max_disp, ref_dem, demgsd: float = 1, imggsd: float = 0.25, **kwargs) -> None:
        """
        Given estimate of max disparity between reference elevation model
        and HiRISE output, run point cloud alignment and
        produce the final DEM/ORTHO data products.

        This command runs steps 10-12 of the HiRISE pipeline

        :param max_disp: Maximum expected displacement in meters
        :param ref_dem: Absolute path the reference dem
        :param demgsd: GSD of final Dem, default is 1 mpp
        :param imggsd: GSD of full res image
        """
        self.hirise.step_ten(max_disp, ref_dem, **kwargs)
        self.hirise.step_eleven(mpp=demgsd, **kwargs)
        self.hirise.step_twelve(**kwargs)
        # if user wants a second image with same res as step
        # eleven don't bother as prior call to eleven did the work
        if not math.isclose(imggsd, demgsd):
            self.hirise.step_eleven(mpp=imggsd, just_ortho=True)

    def info(self):
        """
        Get the number of threads and processes as a formatted string

        :return: str rep of info
        """
        return f"threads sp: {_threads_singleprocess}\nthreads mp: {_threads_multiprocess}\nprocesses: {_processes}"


def main():
    fire.Fire(ASAP)

if __name__ == '__main__':
    main()

