import fire
from sh import Command
from contextlib import contextmanager
import os
from typing import Optional, Dict
import moody
import re


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

class ASAP(object):

    def __init__(self, https=False):
        self.https = https

    @staticmethod
    def _ctx_step_one(stereo: str, ids: str, pedr_list: str, stereo2: Optional[str] = None) -> None:
        old_ctx_one = Command('ctx_pipeline_part_one.sh')
        if stereo2:
            old_ctx_one(stereo, stereo2, ids, pedr_list, _fg=True)
        else:
            old_ctx_one(stereo, ids, pedr_list, _fg=True)

    @staticmethod
    def _ctx_step_two(stereodirs: str, max_disp: int) -> None:
        old_ctx_two = Command('ctx_pipeline_part_two.sh')
        old_ctx_two(stereodirs, max_disp, _fg=True)

    @staticmethod
    def _hirise_step_one(stereo: str, ids: str) -> None:
        old_hirise_one = Command('hirise_pipeline_part_one.sh')
        old_hirise_one(stereo, ids, _fg=True)

    @staticmethod
    def _hirise_step_two(stereodirs: str, max_disp: int, ref_dem: str) -> None:
        old_hirise_two = Command('hirise_pipeline_part_two.sh')
        old_hirise_two(stereodirs, max_disp, ref_dem, _fg=True)

    def get_full_ctx_id(self, pid):
        res = str(moody.ODE(self.https).get_ctx_meta_by_key(pid, 'ProductURL'))
        return res.split('=')[1].split('&')[0]

    def get_ctx_emission_angle(self, pid):
        return float(moody.ODE(self.https).get_ctx_meta_by_key(pid, 'Emission_angle'))

    def get_hirise_emission_angle(self, pid):
        return float(moody.ODE(self.https).get_hirise_meta_by_key(f'{pid}_R*', 'Emission_angle'))

    def get_ctx_order(self, one, two):
        em_one = self.get_ctx_emission_angle(one)
        em_two = self.get_ctx_emission_angle(two)
        if em_one <= em_two:
            return one, two
        else:
            return two, one

    def get_hirise_order(self, one, two):
        em_one = self.get_hirise_emission_angle(one)
        em_two = self.get_hirise_emission_angle(two)
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

    def generate_hirise_pair_list(self, one, two):
        order = self.get_hirise_order(one, two)
        with open('pair.lis', 'w', encoding='utf') as o:
            for pid in order:
                o.write(pid)
                o.write('\n')

    def ctx_one(self, one: str, two: str, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self.generate_ctx_pair_list(one, two)
            # download files
            moody.ODE(self.https).ctx_edr(one)
            moody.ODE(self.https).ctx_edr(two)

    def ctx_two(self, stereo: str, pedr_list: str, stereo2: Optional[str] = None, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_one(stereo, './pair.lis', pedr_list, stereo2=stereo2)

    def ctx_three(self, max_disp, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._ctx_step_two('./stereodirs.lis', max_disp)

    def hirise_one(self, one, two, cwd: Optional[str] = None) -> None:
        from pathlib import Path
        with cd(cwd):
            self.generate_hirise_pair_list(one, two)
            # download files
            Path(one).mkdir(exist_ok=True)
            with cd(one):
                moody.ODE(self.https).hirise_edr(f'{one}_R*')

            Path(two).mkdir(exist_ok=True)
            with cd(two):
                moody.ODE(self.https).hirise_edr(f'{two}_R*')

    def hirise_two(self, stereo, cwd: Optional[str] = None) -> None:
        with cd(cwd):
               self._hirise_step_one(stereo, './pair.lis')

    def hirise_three(self, max_disp, ref_dem, cwd: Optional[str] = None) -> None:
        with cd(cwd):
            self._hirise_step_two('./stereodirs.lis', max_disp, ref_dem)

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




def main():
    fire.Fire(ASAP)

if __name__ == '__main__':
    main()

