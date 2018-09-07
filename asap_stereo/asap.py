import fire
import sh
from sh import Command
from contextlib import contextmanager
import os
from typing import Optional, Dict
import moody
import re
from pathlib import Path
from threading import Semaphore


pool = Semaphore(16)

def done(cmd, success, exit_code):
    pool.release()

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    try:
        if newdir:
            os.chdir(newdir)
        yield
    finally:
        os.chdir(prevdir)

def isis3_to_dict(instr: str)-> Dict:
    groups = re.findall(r'Group([\S\s]*?)End_Group', instr)
    out = {}
    for group in groups:
        lines = [x.replace('=', '').split() for x in group.split('\\n')]
        group_name = lines[0][0]
        out[group_name] = {t[0]: t[1] for t in lines[1:-1]}
    return out

def get_cam_info(img)-> Dict:
    camrange = Command('camrange')
    out = str(camrange(f'from={img}').stdout)
    out_dict = isis3_to_dict(out)
    return out_dict


class CommonSteps(object):

    def __init__(self):
        self.ba = Command('bundle_adjust').bake(_fg=True)
        self.parallel_stereo = Command('parallel_stereo').bake(_fg=True)
        self.point2dem = Command('point2dem').bake(_fg=True)
        self.pc_align = Command('pc_align').bake('--highest-accruacy', '--save-inv-transform', _fg=True)
        self.dem_geoid = Command('dem_geoid').bake(_fg=True)

    @staticmethod
    def get_srs_info(img)-> str:
        out_dict = get_cam_info(img)
        lon = (float(out_dict['UniversalGroundRange']['MinimumLongitude']) + float(out_dict['UniversalGroundRange']['MaximumLongitude'])) / 2
        proj4str = f"+proj=sinu +lon_0={lon} +x_0=0 +y_0=0 +a={out_dict['Target']['RadiusA']} +b={out_dict['Target']['RadiusB']} +units=m +no_defs"
        return proj4str

    @staticmethod
    def get_map_info(img, key: str, group='UniversalGroundRange')-> str:
        out_dict = get_cam_info(img)
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

    def ctx_one(self, one: str, two: str, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self.generate_ctx_pair_list(one, two)
            # download files
            moody.ODE(self.https).ctx_edr(one)
            moody.ODE(self.https).ctx_edr(two)


class HiRISE(object):

    def __init__(self, https=False, threads=16):
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

    def step_one(self, one, two, cwd: Optional[str] = None):
        with cd(cwd):
            self.generate_hirise_pair_list(one, two)
            # download files
            Path(one).mkdir(exist_ok=True)
            with cd(one):
                moody.ODE(self.https).hirise_edr(f'{one}_R*')

            Path(two).mkdir(exist_ok=True)
            with cd(two):
                moody.ODE(self.https).hirise_edr(f'{two}_R*')

    def step_two(self):
        self.cs.create_stereopairs_lis()
        self.cs.create_stereodirs_lis()
        self.cs.create_sterodirs()
        self.cs.create_stereopair_lis()

    def step_three(self):
        """
        Run hiedr2mosic on all the data
        :return:
        """
        hiedr = sh.Command('hiedr2moasic.py')

        def hiedr2mosaic(*im):
            # hiedr2moasic is given a glob of tifs
            pool.acquire()
            return hiedr(*im, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(Path(left)):
            procs.append(hiedr2mosaic(Path('./').glob('*.IMG')))
        with cd(Path(right)):
            procs.append(hiedr2mosaic(Path('./').glob('*.IMG')))
        _ = [p.wait() for p in procs]
        print('Finished hiedr2mosaic on images')

    def step_four(self):
        left, right, both = self.cs.parse_stereopairs()
        sh.mv(next(Path(f'./{left}/').glob(f'{left}*.mos_hijitreged.norm.cub')), both)
        sh.mv(next(Path(f'./{right}/').glob(f'{right}*.mos_hijitreged.norm.cub')), both)

    def step_five(self):
        cam2map = sh.Command('cam2map4stereo.py')

        def par_cam2map(im):
            pool.acquire()
            return cam2map(im, _bg=True, _done=done)

        left, right, both = self.cs.parse_stereopairs()
        procs = []
        with cd(both):
            procs.append(par_cam2map(next(Path('.').glob(f'{left}*.mos_hijitreged.norm.cub'))))
            procs.append(par_cam2map(next(Path('.').glob(f'{right}*.mos_hijitreged.norm.cub'))))
        _ = [p.wait() for p in procs]
        print('Finished cam2map4stereo on images')

    def step_six(self, bundle_adjust_prefix='adjust/ba'):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path.cwd() / both):
            sh.echo(f"Begin bundle_adjust at {sh.date()}", _fg=True)
            self.cs.ba(f'{left}_RED.map.cub', f'{right}_RED.map.cub', '-o', bundle_adjust_prefix, '--threads', self.threads, _fg=True)
            sh.echo(f"End   bundle_adjust at {sh.date()}", _fg=True)

    def step_seven(self, stereo_conf, processes=2, threads_multiprocess=8, threads_singleprocess=16, bundle_adjust_prefix='adjust/ba'):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path('.') / both):
            self.cs.parallel_stereo('--processes'            , processes,
                                    '--threads-singleprocess', threads_singleprocess,
                                    '--threads-multiprocesss', threads_multiprocess,
                                    '--stop-point'           , 4,
                                    f'{left}_RED.map.cub'    , f'{right}_RED.map.cub'
                                    '-s'                     , Path(stereo_conf).absolute(),
                                    f'results/{both}'        ,
                                    '--bundle-adjust-prefix' , bundle_adjust_prefix)

    def step_eight(self, stereo_conf, processes=16, threads_multiprocess=8, threads_singleprocess=16, bundle_adjust_prefix='adjust/ba'):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path('.') / both):
            self.cs.parallel_stereo('--processes'            , processes,
                                    '--threads-singleprocess', threads_singleprocess,
                                    '--threads-multiprocesss', threads_multiprocess,
                                    '--entry-point'           , 4,
                                    f'{left}_RED.map.cub'    , f'{right}_RED.map.cub'
                                    '-s'                     , Path(stereo_conf).absolute(),
                                    f'results/{both}'        ,
                                    '--bundle-adjust-prefix' , bundle_adjust_prefix)

    def step_nine(self, mpp=2):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path('.') / both / 'results'):
            proj = self.cs.get_srs_info(f'../{left}_RED.map.cub')
            self.cs.point2dem('--t_srs', proj, '-r', 'mars', '--nodata', -32767, '-s', mpp, '-n', '--errorimage', f'{both}-PC.tif',
                              '--orthoimage', f'{both}-L.tif', '-o', f'dem/{both}')

    def step_ten(self, maxd, refdem):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path('.') / both):
            with cd('results'):
                self.cs.pc_align('--mac-displacement', maxd, '--threads', self.threads, f'{both}-PC.tif', refdem, '--datum', 'D_MARS', '-o', f'dem_align/{both}_align')

    def step_eleven(self, gsd, just_ortho=False):
        left, right, both = self.cs.parse_stereopairs()
        gsd_postfix = str(float(gsd)).replace('.','_')

        add_params = []
        if just_ortho:
            add_params.append('--no-dem')

        with cd(Path('.') / both / 'results'):
            proj = self.cs.get_srs_info(f'../{left}_RED.map.cub')
            with cd('dem_align'):
                self.cs.point2dem('--t_srs', proj, '-r', 'mars', '--nodata', -32767, '-s', gsd, '-n', '--errorimage', f'{both}_align-trans_reference.tif',
                                  '--orthoimage', f'../{both}-L.tif', '-o', f'{both}_align_{gsd_postfix}', *add_params)

    def step_twelve(self):
        left, right, both = self.cs.parse_stereopairs()
        with cd(Path('.') / both / 'results' / 'dem_align'):
            file = next(Path('.').glob('-DEM.tif'))
            self.cs.dem_geoid(file, '-o', f'{file.stem}')


class ASAP(object):

    def __init__(self, https=False):
        self.https = https
        self.hirise = HiRISE(self.https)
        self.ctx = CTX(self.https)
        self.common = CommonSteps()

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

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_one(stereo, './pair.lis', pedr_list, stereo2=stereo2)

    def ctx_three(self, max_disp, demgsd: float = 24, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_two('./stereodirs.lis', max_disp, demgsd)

    def hirise_one(self, left, right):
        self.hirise.step_one(left, right)

    def hirise_two(self, stereo, cwd: Optional[str] = None, **kwargs) -> None:
        self.hirise.step_two()
        self.hirise.step_three()
        self.hirise.step_four()
        self.hirise.step_five()
        self.hirise.step_six()
        self.hirise.step_seven(stereo)
        self.hirise.step_eight(stereo)
        self.hirise.step_nine()

    def hirise_three(self, max_disp, ref_dem, demgsd: float = 1, imggsd: float = 0.25, cwd: Optional[str] = None) -> None:
        self.hirise.step_ten(max_disp, ref_dem)
        self.hirise.step_eleven(demgsd)
        self.hirise.step_twelve()
        self.hirise.step_eleven(imggsd, just_ortho=True)



def main():
    fire.Fire(ASAP)

if __name__ == '__main__':
    main()

