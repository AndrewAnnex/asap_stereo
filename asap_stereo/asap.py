import fire
import sh
from sh import Command
from contextlib import contextmanager
import functools
import os
import sys
import datetime
import itertools
from typing import Optional, Dict, List
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
_threads_multiprocess  = _threads_singleprocess // 2 # 12, 8
_processes             = _threads_multiprocess // 4 # 3, 2



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

def cmd_to_string(command: sh.RunningCommand):
    """
    Converts the running command into a single string of the full command call for easier logging
    :param command:
    :return:
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
    groups = re.findall(r'Group([\S\s]*?)End_Group', instr)
    out = {}
    for group in groups:
        lines = [x.replace('=', '').split() for x in group.split('\\n')]
        group_name = lines[0][0]
        out[group_name] = {t[0]: t[1] for t in lines[1:-1]}
    return out

def rich_logger(func):
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


class CommonSteps(object):

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

    @staticmethod
    def par_do(func, all_calls_args):
        procs = []

        def do(*args):
            pool.acquire()
            return func(*args)

        for call_args in all_calls_args:
            if ' ' in call_args:
                # if we are running a command with multiple args, sh needs different strings
                call_args = call_args.split(' ')
                procs.append(do(*call_args))
            else:
                procs.append(do(call_args))

        return [p.wait() for p in procs]

    def __init__(self):
        self.parallel_stereo = Command('parallel_stereo').bake(_out=sys.stdout, _err=sys.stderr)
        self.point2dem   = Command('point2dem').bake(_out=sys.stdout, _err=sys.stderr)
        self.pc_align    = Command('pc_align').bake('--highest-accuracy', '--save-inv-transform', _out=sys.stdout, _err=sys.stderr)
        self.dem_geoid   = Command('dem_geoid').bake(_out=sys.stdout, _err=sys.stderr)
        self.mroctx2isis = Command('mroctx2isis').bake(_out=sys.stdout, _err=sys.stderr)
        self.spiceinit   = Command('spiceinit').bake(_out=sys.stdout, _err=sys.stderr)
        self.spicefit    = Command('spicefit').bake(_out=sys.stdout, _err=sys.stderr)
        self.ctxcal      = Command('ctxcal').bake(_out=sys.stdout, _err=sys.stderr)
        self.ctxevenodd  = Command('ctxevenodd').bake(_out=sys.stdout, _err=sys.stderr)
        self.hillshade   = Command('gdaldem').hillshade.bake(_out=sys.stdout, _err=sys.stderr)
        self.mapproject  = Command('mapproject').bake(_out=sys.stdout, _err=sys.stderr)
        self.pedr_bin4pc_align = Command('pedr_bin4pc_align.sh').bake(_out=sys.stdout, _err=sys.stderr)
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
    def get_cam_info(img) -> Dict:
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
        import rasterio as rio
        with rio.open(img) as i:
            return i.crs

    @staticmethod
    def get_img_bounds(img):
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
        :param img1:
        :param img2:
        :param factor:
        :return:
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
        :return:
        """
        true_gsd = self.get_image_gsd(path)
        if mpp < true_gsd * 3:
            warnings.warn(f"True image GSD is possibly too big for provided mpp value of {mpp} (compare to 3xGSD={true_gsd * 3})",
                          category=RuntimeWarning)

    @staticmethod
    def get_mpp_postfix(mpp):
        return str(float(mpp)).replace('.', '_')

    @rich_logger
    def get_pedr_4_pcalign_w_moody(self, cub_path, proj = None, https=True):
        """
        Python replacement for pedr_bin4pc_align.sh
        hopefully this will be replaced by a method that queries the mars ODE rest API directly or
        uses a spatial index on the pedr files for speed
        :param proj: optional projection override
        :param https: optional way to disable use of https
        :param cub_path: path to input file to get query geometry
        :return:
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
            sql_query = f'SELECT Lat, Lon, Planet_Rad - 3396190.0 AS Datum_Elev, Topography FROM {shpfile.stem}'
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
        :return:
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
            # clean up files
            #sh.rm(f'./{out_name}_pedr.tab')
            #sh.rm(f'./{out_name}_pedr.asc')
            # ## TODO: Add functionality to build VRT for CSV and then convert to shapefile using ogr2ogr

    @rich_logger
    def bundle_adjust(self, postfix='_RED.map.cub', bundle_adjust_prefix='adjust/ba', **kwargs):
        """
        Bundle adjustment wrapper
        :param postfix:
        :param bundle_adjust_prefix:
        :param kwargs:
        :return:
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
        :param left:
        :param right:
        :param kwargs:
        :return:
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.defaults_ps1, **clean_kwargs(kwargs)})
            return self.parallel_stereo(*args, f'{left}{postfix}', f'{right}{postfix}', '-s', Path(stereo_conf).absolute(), f'results_ba/{both}_ba')

    @rich_logger
    def stereo_2(self, stereo_conf: str, postfix='.lev1eo.cub', **kwargs):
        """
        Step 2 of parallel stereo
        :param postfix:
        :param stereo_conf:
        :param left:
        :param right:
        :param kwargs:
        :return:
        """
        left, right, both = self.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.defaults_ps2, **clean_kwargs(kwargs)})
            return self.parallel_stereo(*args, f'{left}{postfix}', f'{right}{postfix}', '-s', Path(stereo_conf).absolute(), f'results_ba/{both}_ba')

class CTX(object):

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
    def old_ctx_step_one(stereo: str, ids: str, pedr_list: str, stereo2: Optional[str] = None) -> None:
        old_ctx_one = Command('ctx_pipeline_part_one.sh')
        if stereo2:
            old_ctx_one(stereo, stereo2, ids, pedr_list, _fg=True)
        else:
            old_ctx_one(stereo, ids, pedr_list, _fg=True)

    @staticmethod
    def old_ctx_step_two(stereodirs: str, max_disp: int, demgsd: float) -> None:
        old_ctx_two = Command('ctx_pipeline_part_two.sh')
        old_ctx_two(stereodirs, max_disp, demgsd, _fg=True)

    @staticmethod
    def notebook_pipeline_make_dem(left: str, right: str, pedr_list: str, config1: str, working_dir ='./', config2: Optional[str] = None, out_notebook=None, **kwargs):
        """
        First step in CTX DEM pipeline that uses papermill to persist log

        this command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command
        :param out_notebook:
        :param config2:
        :param working_dir:
        :param config1:
        :param pedr_list:
        :param left:
        :param right:
        :return:
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem.ipynb'
        pm.execute_notebook(
            f'{here}/asap_ctx.ipynb',
            out_notebook,
            parameters={
                'left' : left,
                'right': right,
                'peder_list': pedr_list,
                'config1': config1,
                'config2': config2,
                'output_path' : working_dir,
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @staticmethod
    def notebook_pipeline_align_dem(maxdisp = 500, demgsd = 24.0, imggsd = 6.0, working_dir ='./', out_notebook=None, **kwargs):
        """
        Second and final step in CTX DEM pipeline that uses papermill to persist log

        this command aligns the CTX DEM produced in step 1 to the Mola Datum
        I recommend strongly to use nohup with this command
        :param maxdisp:
        :param demgsd:
        :param imggsd:
        :param working_dir:
        :param out_notebook:
        :param kwargs:
        :return:
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_align_dem.ipynb'
        pm.execute_notebook(
            f'{here}/asap_ctx_pc_align.ipynb',
            out_notebook,
            parameters={
                'maxdisp' : maxdisp,
                'demgsd' : demgsd,
                'imggsd' : imggsd
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @rich_logger
    def step_one(self, one: str, two: str, cwd: Optional[str] = None) -> None:
        """
        Download CTX EDRs

        :param one:
        :param two:
        :param cwd:
        :return:
        """
        with cd(cwd):
            self.generate_ctx_pair_list(one, two)
            # download files
            moody.ODE(self.https).ctx_edr(one)
            moody.ODE(self.https).ctx_edr(two)

    @rich_logger
    def step_two(self):
        """
        ISIS3 CTX preprocessing
        replaces ctxedr2lev1eo.sh
        :return:
        """
        _cwd = Path.cwd()
        imgs = list(_cwd.glob('*.IMG')) + list(_cwd.glob('*.img'))
        self.cs.par_do(self.cs.mroctx2isis, [f'from={i.name} to={i.stem}.cub' for i in imgs])
        cubs = list(Path.cwd().glob('*.cub'))
        self.cs.par_do(self.cs.spiceinit, [f'from={c.name}' for c in cubs])
        self.cs.par_do(self.cs.spicefit, [f'from={c.name}' for c in cubs])
        self.cs.par_do(self.cs.ctxcal, [f'from={c.name} to={c.stem}.lev1.cub' for c in cubs])
        lev1cubs = list(Path.cwd().glob('*.lev1.cub'))
        self.cs.par_do(self.cs.ctxevenodd, [f'from={c.name} to={c.stem}eo.cub' for c in lev1cubs])
        for cub in cubs:
            cub.unlink()
        for lc in lev1cubs:
            lc.unlink()

    @rich_logger
    def step_three(self):
        """
        Do a bunch of checks for asp_ctx_lev1eo2dem.sh
        :return:
        """
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_sterodirs()
        self.cs.create_stereopair_lis()
        # copy the cub files into the both directory
        _, _, both = self.cs.parse_stereopairs()
        return sh.cp('-n', sh.glob('./*.cub'), f'./{both}/')

    @rich_logger
    def step_four(self, bundle_adjust_prefix='adjust/ba', **kwargs)-> sh.RunningCommand:
        """
        Bundle Adjust CTX
        asp_ctx_lev1eo2dem.sh
        Run bundle adjustment on the CTX map projected data
        :param bundle_adjust_prefix:
        :return:
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
        :return:
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
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / folder / output_folder):
            dem = next(Path.cwd().glob('*DEM.tif'))
            self.cs.hillshade(dem.name, f'./{dem.stem}-hillshade.tif')

    @rich_logger
    def step_nine(self, refdem=None):
        """
        Mapproject the left and right ctx images against the reference DEM
        :param refdem:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        with cd(Path.cwd() / both):
            # map project both ctx images against the reference dem
            # might need to do par do here
            self.cs.mapproject('-t', 'isis', refdem, f'{left}.lev1eo.cub', f'{left}.ba.map.tif', '--mpp', 6, '--bundle-adjust-prefix', 'adjust/ba')
            self.cs.mapproject('-t', 'isis', refdem, f'{right}.lev1eo.cub', f'{right}.ba.map.tif', '--mpp', 6, '--bundle-adjust-prefix', 'adjust/ba')

    @rich_logger
    def step_ten(self, stereo_conf, refdem=None, **kwargs):
        """
        second stereo first step
        :param stereo_conf:
        :param refdem:
        :param kwargs:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.cs.defaults_ps1, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}.ba.map.tif', f'{right}.ba.map.tif', f'{left}.lev1eo.cub', f'{right}.lev1eo.cub',
                                           '-s', Path(stereo_conf).absolute(), f'results_map_ba/{both}_ba', refdem)

    @rich_logger
    def step_eleven(self, stereo_conf, refdem=None, **kwargs):
        """
        second stereo second step
        :param stereo_conf:
        :param refdem:
        :param kwargs:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        if not refdem:
            refdem = Path.cwd() / both / 'results_ba' / 'dem' / f'{both}_ba_100_0-DEM.tif'
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**self.cs.defaults_ps2, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}.ba.map.tif', f'{right}.ba.map.tif', f'{left}.lev1eo.cub', f'{right}.lev1eo.cub',
                                           '-s', Path(stereo_conf).absolute(), f'results_map_ba/{both}_ba', refdem)

    @rich_logger
    def step_twelve(self, pedr_list):
        """
        Run pedr_bin4pc_align
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both):
            if pedr_list:
                self.cs.get_pedr_4_pcalign(f'{left}.lev1eo.cub', pedr_list, self.proj)
            else:
                self.cs.get_pedr_4_pcalign_w_moody(f'{left}.lev1eo.cub', proj=self.proj, https=self.https)

    @rich_logger
    def step_thirteen(self, maxd, pedr4align=None, **kwargs):
        """
        PC Align CTX

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs
        :param maxd:
        :param refdem:
        :param kwargs:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        defaults = {
            '--num-iterations': 4000,
            '--threads': _threads_singleprocess,
            '--datum'  : self.datum,
            '--max-displacement': maxd,
            '--output-prefix': f'dem_align/{both}_map_ba_align'
        }
        if not pedr4align:
            pedr4align = f'../{both}_pedr4align.csv'
        with cd(Path.cwd() / both / 'results_map_ba'):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.pc_align(*args, '--save-inv-trans', '--highest-accuracy', f'{both}_ba-PC.tif', pedr4align)

    @rich_logger
    def step_fourteen(self, mpp=24.0, just_ortho=False, output_folder='dem_align', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products
        :param mpp:
        :param just_ortho:
        :param output_folder:
        :param kwargs:
        :return:
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
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / 'results_map_ba' / output_folder):
            file = next(Path.cwd().glob('*-DEM.tif'))
            args = kwargs_to_args(clean_kwargs(kwargs))
            return self.cs.dem_geoid(*args, file, '-o', f'{file.stem}')

class HiRISE(object):

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

    def get_hirise_emission_angle(self, pid):
        return float(moody.ODE(self.https).get_hirise_meta_by_key(f'{pid}_R*', 'Emission_angle'))

    def get_hirise_order(self, one, two):
        em_one = self.get_hirise_emission_angle(one)
        em_two = self.get_hirise_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def generate_hirise_pair_list(self, one, two):
        order = self.get_hirise_order(one, two)
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in order:
                o.write(pid)
                o.write('\n')

    @staticmethod
    def old_hirise_step_two(stereodirs: str, max_disp: int, ref_dem: str, demgsd: float, imggsd: float) -> None:
        old_hirise_two = Command('hirise_pipeline_part_two.sh')
        old_hirise_two(stereodirs, max_disp, ref_dem, demgsd, imggsd, _fg=True)

    @staticmethod
    def notebook_pipeline_make_dem(left: str, right: str, config: str, working_dir ='./', out_notebook=None, **kwargs):
        """
        First step in HiRISE DEM pipeline that uses papermill to persist log

        This command does most of the work, so it is long running!
        I recommend strongly to use nohup with this command, even more so for HiRISE!
        :param out_notebook:
        :param working_dir:
        :param config:
        :param left:
        :param right:
        :return:
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_make_dem_hirise.ipynb'
        pm.execute_notebook(
            f'{here}/asap_hirise.ipynb',
            out_notebook,
            parameters={
                'left' : left,
                'right': right,
                'config': config,
                'output_path' : working_dir,
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @staticmethod
    def notebook_pipeline_align_dem(refdem, maxdisp = 500, demgsd = 1.0, imggsd = 0.25, alignment_method = 'rigid', working_dir ='./', out_notebook=None, **kwargs):
        """
        Second and final step in HiRISE DEM pipeline that uses papermill to persist log

        This pipeline aligns the HiRISE DEM produced to an input DEM, typically a CTX DEM.

        It will first attempt to do this using point alignment on hillshaded views of the dems.

        I recommend strongly to use nohup with this command
        :param alignment_method:
        :param refdem:
        :param maxdisp:
        :param demgsd:
        :param imggsd:
        :param working_dir:
        :param out_notebook:
        :param kwargs:
        :return:
        """
        if not out_notebook:
            out_notebook = f'{working_dir}/log_asap_notebook_pipeline_align_dem_hirise.ipynb'
        pm.execute_notebook(
            f'{here}/asap_hirise_pc_align.ipynb',
            out_notebook,
            parameters={
                'refdem' : refdem,
                'maxdisp' : maxdisp,
                'demgsd' : demgsd,
                'imggsd' : imggsd,
                'alignment_method' : alignment_method,
            },
            request_save_on_cell_execute=True,
            **kwargs
        )

    @rich_logger
    def step_one(self, one, two, cwd: Optional[str] = None):
        """
        Download HiRISE EDRs

        Download two HiRISE images worth of EDR files to two folders
        :param one:
        :param two:
        :param cwd:
        :return:
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
        :return:
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
        :return:
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
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        sh.mv(next(Path(f'./{left}/').glob(f'{left}*.mos_hijitreged.norm.cub')), both)
        sh.mv(next(Path(f'./{right}/').glob(f'{right}*.mos_hijitreged.norm.cub')), both)

    @rich_logger
    def step_five(self):
        """
        Cam2map4Stereo

        Run cam2map4stereo on the data
        :return:
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
        :param max_iterations:
        :return:
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
        :return:
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
        #todo this just rescales the image, did I forget to hillshade or does pc_align do that for me?
        :param mpp:
        :return:
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
        :param refdem:
        :param kwargs:
        :return:
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
    def step_ten(self, maxd, refdem, **kwargs):
        """
        PC Align HiRISE

        Run pc_align using provided max disparity and reference dem
        optionally accept an initial transform via kwargs
        :param maxd:
        :param refdem:
        :param kwargs:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        defaults = {
            '--num-iterations': 2000,
            '--threads': self.threads,
            '--datum'  : self.datum,
            '--max-displacement': maxd,
            '--output-prefix': f'dem_align/{both}_align'
        }
        with cd(Path.cwd() / both):
            with cd('results_ba'):
                args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
                return self.cs.pc_align(*args, f'{both}_ba-PC.tif', refdem)

    @rich_logger
    def step_eleven(self, mpp=1.0, just_ortho=False, output_folder='dem_align', **kwargs):
        """
        Produce final DEMs/Orthos

        Run point2dem on the aligned output to produce final science ready products
        :param mpp:
        :param just_ortho:
        :param output_folder:
        :param kwargs:
        :return:
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
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both / 'results_ba' / output_folder):
            file = next(Path.cwd().glob('*-DEM.tif'))
            args = kwargs_to_args(clean_kwargs(kwargs))
            return self.cs.dem_geoid(*args, file, '-o', f'{file.stem}')


class ASAP(object):

    def __init__(self, https=False, datum="D_MARS"):
        self.https  = https
        self.hirise = HiRISE(self.https, datum=datum)
        self.ctx    = CTX(self.https, datum=datum)
        self.common = CommonSteps()
        self.get_srs_info = self.common.get_srs_info
        self.get_map_info = self.common.get_map_info

    def ctx_one(self, left, right, cwd: Optional[str] = None):
        with cd(cwd):
            self.ctx.step_one(left, right)
            # ctxedr2lev1eo steps
            self.ctx.step_two()
            # move things
            self.ctx.step_three()

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
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

    def ctx_three(self, max_disp, demgsd: float = 24, cwd: Optional[str] = None, **kwargs) -> None:
        with cd(cwd):
            self.ctx.step_thirteen(max_disp, **kwargs)
            self.ctx.step_fourteen(mpp=demgsd, **kwargs)
            self.ctx.step_fifteen(**kwargs)
            self.ctx.step_fourteen(mpp=6, just_ortho=True, **kwargs)
            self.ctx.step_eight(folder='results_map_ba', output_folder='dem_align')

    def hirise_one(self, left, right):
        """
        Download the EDR data from the PDS, requires two HiRISE Id's
        (order left vs right does not matter)
        :param left: HiRISE Id
        :param right: HiRISE Id
        """
        self.hirise.step_one(left, right)

    def hirise_two(self, stereo, mpp=2, bundle_adjust_prefix='adjust/ba', max_iterations=50) -> None:
        """
        Run various calibration steps then:
        bundle adjust, produce DEM, render low res version for inspection
        This will take a while (sometimes over a day), use nohup!
        :param stereo:
        :param mpp:
        :param bundle_adjust_prefix:
        :param max_iterations:
        :return:
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
        :param max_disp: Disparity in meters
        :param ref_dem: absolute path the reference dem
        :param demgsd: GSD of final Dem, default is 1 mpp
        :param imggsd: GSD of full res image
        :return:
        """
        self.hirise.step_ten(max_disp, ref_dem, **kwargs)
        self.hirise.step_eleven(mpp=demgsd, **kwargs)
        self.hirise.step_twelve(**kwargs)
        # if user wants a second image with same res as step
        # eleven don't bother as prior call to eleven did the work
        if not math.isclose(imggsd, demgsd):
            self.hirise.step_eleven(mpp=imggsd, just_ortho=True)


def main():
    fire.Fire(ASAP)

if __name__ == '__main__':
    main()

