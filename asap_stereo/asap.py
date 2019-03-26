import fire
import sh
from sh import Command
from contextlib import contextmanager
import functools
import os
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

def cmd_to_string(command: sh.RunningCommand):
    """
    Converts the running command into a single string of the full command call for easier logging
    :param command:
    :return:
    """
    return  " ".join((_.decode("utf-8") for _ in command.cmd))

def clean_kwargs(kwargs: Dict)-> Dict:
    cleaned = {}
    for key in kwargs.keys():
        new_key = str(key)
        if not key.startswith('--'):
            new_key = f'--{key}'
        new_key = new_key.replace('_', '-')
        cleaned[new_key] = kwargs[key]
    return cleaned

def kwargs_to_args(kwargs: Dict)-> List:
    keys = []
    # ensure keys start with '--' for asp scripts
    for key in kwargs.keys():
        key = str(key)
        key = key.replace('_', '-')
        if not key.startswith('--'):
            keys.append(f'--{key}')
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
        # grab the first doc line for the pretty name, make sure all functions have docs!
        pretty_name = func.__doc__.splitlines()[0]
        print(f"""Started : {func.__name__}({pretty_name}), at: {start_time.isoformat(" ")}""", flush=True)
        #
        v = func(*args, **kwargs)
        if v is not None:
            try:
                print(f'Ran Command: {cmd_to_string(v)}', flush=True)
            except BaseException:
                pass
        #
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"""Finished: {func.__name__}({pretty_name}), at: {end_time.isoformat(" ")}, duration: {str(duration)}""", flush=True)
        return v
    return wrapper


class CommonSteps(object):

    @staticmethod
    def par_do(func, calls):
        procs = []

        def do(thiscall):
            pool.acquire()
            return func(thiscall)

        for call in calls:
            procs.append(do(call))

        return [p.wait() for p in procs]

    def __init__(self):
        self.ba          = Command('bundle_adjust').bake(_fg=True)
        self.parallel_stereo = Command('parallel_stereo').bake(_fg=True)
        self.point2dem   = Command('point2dem').bake(_fg=True)
        self.pc_align    = Command('pc_align').bake('--highest-accuracy', '--save-inv-transform', _fg=True)
        self.dem_geoid   = Command('dem_geoid').bake(_fg=True)
        self.mroctx2isis = Command('mroctx2isis').bake(_fg=True)
        self.spiceinit   = Command('spiceinit').bake(_fg=True)
        self.spicefit    = Command('spicefit').bake(_fg=True)
        self.ctxcal      = Command('ctxcal').bake(_fg=True)
        self.ctxevenodd  = Command('ctxevenodd').bake(_fg=True)

    @staticmethod
    def get_cam_info(img) -> Dict:
        camrange = Command('camrange')
        out = str(camrange(f'from={img}').stdout)
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
            res1, res2 = math.fabs(cam_info["PixelResolution"]["Lowest"]), math.fabs(cam_info["PixelResolution"]["Highest"])
        if opinion.lower() == 'lower':
            return min(res1, res2)
        elif opinion.lower() == 'higher':
            return max(res1, res2)
        elif opinion.lower() == 'average':
            return (res1+res2)/2
        else:
            raise RuntimeError(f'Opinion {opinion} is not valid, must be "lower" or "higher" or "average".')

    @staticmethod
    def get_srs_info(img)-> str:
        proj4str = sh.gdalsrsinfo(img, o='proj4')
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
        return sh.cat('./stereopairs.lis').strip().split(' ')

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


class CTX(object):

    def __init__(self, https=False):
        self.cs = CommonSteps()
        self.https = https

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

    @rich_logger
    def ctx_one(self, one: str, two: str, cwd: Optional[str] = None) -> None:
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

        :return:
        """
        imgs = list(Path.cwd().glob('*.IMG|*.img'))
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


class HiRISE(object):

    def __init__(self, https=False, threads=cores):
        self.https = https
        self.cs = CommonSteps()
        self.threads = threads

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
        hiedr = sh.Command('hiedr2mosaic.py')

        def hiedr2mosaic(*im):
            # hiedr2moasic is given a glob of tifs
            pool.acquire()
            return hiedr(*im, '--threads', self.threads, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(Path(left)):
            procs.append(hiedr2mosaic(*list(Path('./').glob('*.IMG'))))
        with cd(Path(right)):
            procs.append(hiedr2mosaic(*list(Path('./').glob('*.IMG'))))
        _ = [p.wait() for p in procs]
        print('Finished hiedr2mosaic on images')

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
        cam2map = sh.Command('cam2map4stereo.py')

        def par_cam2map(left_im, right_im):
            pool.acquire()
            return cam2map(left_im, right_im, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(both):
            left_im  = next(Path('.').glob(f'{left}*.mos_hijitreged.norm.cub'))
            right_im = next(Path('.').glob(f'{right}*.mos_hijitreged.norm.cub'))
            procs.append(par_cam2map(left_im, right_im))
        _ = [p.wait() for p in procs]
        sh.echo('Finished cam2map4stereo on images', _fg=True)

    @rich_logger
    def step_six(self, bundle_adjust_prefix='adjust/ba', **kwargs)-> sh.RunningCommand:
        """
        Bundle Adjust HiRISE

        Run bundle adjustment on the HiRISE map projected data
        :param bundle_adjust_prefix:
        :param max_iterations:
        :return:
        """
        defaults = {
            '--threads': self.threads,
            '--datum': 'D_MARS',
            '--max-iterations': 30
        }
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.ba(f'{left}_RED.map.cub', f'{right}_RED.map.cub', '-o', bundle_adjust_prefix, *args, _fg=True)

    @rich_logger
    def step_seven(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 1

        Run first part of parallel_stereo
        """
        defaults = {
            '--processes': _processes,
            '--threads-singleprocess': _threads_singleprocess,
            '--threads-multiprocess': _threads_multiprocess,
            '--stop-point': 4,
            '--bundle-adjust-prefix': 'adjust/ba'
        }
        left, right, both = self.cs.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}_RED.map.cub', f'{right}_RED.map.cub',
                                    '-s', Path(stereo_conf).absolute(), f'results/{both}')

    @rich_logger
    def step_eight(self, stereo_conf, **kwargs):
        """
        Parallel Stereo Part 2

        Run second part of parallel_stereo, stereo is completed after this step
        """
        defaults = {
            '--processes'            : _processes,
            '--threads-singleprocess': _threads_singleprocess,
            '--threads-multiprocess' : _threads_multiprocess,
            '--entry-point'          : 4,
            '--bundle-adjust-prefix' : 'adjust/ba'
        }
        left, right, both = self.cs.parse_stereopairs()
        assert both is not None
        with cd(Path.cwd() / both):
            args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
            return self.cs.parallel_stereo(*args, f'{left}_RED.map.cub', f'{right}_RED.map.cub',
                                    '-s', Path(stereo_conf).absolute(), f'results/{both}')

    @rich_logger
    def step_nine(self, mpp=2, just_dem=False):
        """
        Produce preview DEMs/Orthos

        Produce dem from point cloud, by default 2mpp for hirise for max-disparity estimation
        :param just_dem: set to True if you only want the DEM and no other products like the ortho and error images
        :param mpp:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        mpp_postfix = str(float(mpp)).replace('.', '_')
        add_params = []
        if not just_dem:
            add_params.extend(['-n', '--errorimage', '--orthoimage', f'{both}-L.tif'])

        with cd(Path.cwd() / both / 'results'):
            true_gsd = self.cs.get_image_gsd(f'../{left}_RED.map.cub')
            proj = self.cs.get_srs_info(f'../{left}_RED.map.cub')
            if mpp < true_gsd*3:
                warnings.warn(f"True image GSD is possibly too big for provided mpp value of {mpp} (compare to 3xGSD={true_gsd*3})", category=RuntimeWarning)

            return self.cs.point2dem('--t_srs', f'{proj}', '-r', 'mars', '--nodata', -32767,
                              '-s', mpp, f'{both}-PC.tif', '-o', f'dem/{both}_{mpp_postfix}')

    @rich_logger
    def pre_step_ten(self, refdem):
        """
        Hillshade Align before PC Align

        Automates the procedure to use ipmatch on hillshades of downsampled HiRISE DEM
        to find an initial transform
        :param refdem:
        :param kwargs:
        :return:
        """
        left, right, both = self.cs.parse_stereopairs()
        refdem_mpp = math.ceil(self.cs.get_image_gsd(refdem))
        # create the lower resolution hirise dem to match the refdem gsd
        self.step_nine(mpp=refdem_mpp, just_dem=True)
        # use the image in a call to pc_align with hillshades
        #TODO: auto crop the reference dem to be around hirise more closely\
        cmd_res = None
        with cd(Path.cwd() / both / 'results'):
            lr_hirise_dem = Path.cwd() / 'dem' / f'{both}_{refdem_mpp}-DEM.tif'
            cmd_res = self.cs.pc_align('--max-displacement', -1, '--num-iterations', 0, '--threads', self.threads,
                             '--initial-transform-from-hillshading', '\"similarity\"', '--datum', 'D_MARS',
                              lr_hirise_dem, refdem, '-o', 'hillshade_align/out')
            # done! log out to user that can use the transform
        out_dir = Path.cwd() / both / 'results' / 'hillshade_align'
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
            '--threads': self.threads,
            '--datum'  : 'D_MARS',
            '--max-displacement': maxd,
        }
        with cd(Path.cwd() / both):
            with cd('results'):
                args = kwargs_to_args({**defaults, **clean_kwargs(kwargs)})
                return self.cs.pc_align('--highest-accuracy', *args, f'{both}-PC.tif', refdem, '-o', f'dem_align/{both}_align')

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

        with cd(Path.cwd() / both / 'results'):
            proj = self.cs.get_srs_info(f'../{left}_RED.map.cub')
            true_gsd = self.cs.get_image_gsd(f'../{left}_RED.map.cub')
            if mpp < true_gsd*3 and not just_ortho:
                warnings.warn(f"True image GSD is possibly too big for provided mpp value of {mpp} (compare to 3xGSD={true_gsd*3})",
                              category=RuntimeWarning)
            with cd(output_folder):
                point_cloud = next(Path.cwd().glob('*trans_reference.tif'))
                return self.cs.point2dem('--t_srs', proj, '-r', 'mars', '--nodata', -32767, '-s', mpp, str(point_cloud.name),
                                  '--orthoimage', f'../{both}-L.tif', '-o', f'{both}_align_{gsd_postfix}', *add_params)

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
        with cd(Path.cwd() / both / 'results' / output_folder):
            file = next(Path.cwd().glob('*-DEM.tif'))
            return self.cs.dem_geoid(file, '-o', f'{file.stem}')


class ASAP(object):

    def __init__(self, https=False):
        self.https = https
        self.hirise = HiRISE(self.https)
        self.ctx = CTX(self.https)
        self.common = CommonSteps()
        self.get_srs_info = self.common.get_srs_info
        self.get_map_info = self.common.get_map_info

    @staticmethod
    def _ctx_step_one(stereo: str, ids: str, pedr_list: str, stereo2: Optional[str] = None) -> None:
        old_ctx_one = Command('ctx_pipeline_part_one.sh')
        if stereo2:
            old_ctx_one(stereo, stereo2, ids, pedr_list, _fg=True)
        else:
            old_ctx_one(stereo, ids, pedr_list, _fg=True)

    @staticmethod
    def _ctx_step_two(stereodirs: str, max_disp: int, demgsd: float) -> None:
        old_ctx_two = Command('ctx_pipeline_part_two.sh')
        old_ctx_two(stereodirs, max_disp, demgsd, _fg=True)

    @staticmethod
    def _hirise_step_two(stereodirs: str, max_disp: int, ref_dem: str, demgsd: float, imggsd: float) -> None:
        old_hirise_two = Command('hirise_pipeline_part_two.sh')
        old_hirise_two(stereodirs, max_disp, ref_dem, demgsd, imggsd, _fg=True)

    def ctx_one(self, left, right, cwd: Optional[str] = None):
        with cd(cwd):
            self.ctx.ctx_one(left, right)

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_one(stereo, './pair.lis', pedr_list, stereo2=stereo2)

    def ctx_three(self, max_disp, demgsd: float = 24, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_two('./stereodirs.lis', max_disp, demgsd)

    def hirise_one(self, left, right):
        """
        Download the EDR data from the PDS, requires two HiRISE Id's
        (order left vs right does not matter)
        :param left: HiRISE Id
        :param right: HiRISE Id
        """
        self.hirise.step_one(left, right)

    def hirise_two(self, stereo, mpp=2, bundle_adjust_prefix='adjust/ba', max_iterations=20) -> None:
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

