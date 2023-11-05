from glob import glob
from os import path
from statistics import mean

from exp.utility import read_yaml


class ResultData:

    def __init__(self, directory):
        self.base_fn = ""
        self.raw_rata = []
        self.directory = directory
        for file in glob(path.join(self.directory, "*.yaml")):
            # noinspection PyBroadException
            try:
                record = read_yaml(file)
                if record:
                    self.raw_rata.append(record)
            # flake8: noqa: E722
            except:
                pass

    @property
    def n_results(self):
        return len(self.raw_rata)

    @staticmethod
    def arr_mean(x):
        return 100 * mean(x)

    @staticmethod
    def fold_mean(x, r):
        d, v = sum(x), sum(r['folds']['n_records'])
        return 100 * (d / v)

    @staticmethod
    def name(r):
        e = r['experiment']
        return e['name'] if 'name' in e else e['dataset']

    @staticmethod
    def cls(r):
        return r['classifier']['name']

    @staticmethod
    def acc(r):
        return ResultData.arr_mean(r['folds']['accuracy'])

    @staticmethod
    def valid(r):
        return ResultData.fold_mean(r['folds']['n_valid_evades'], r)

    @staticmethod
    def evades(r):
        return ResultData.fold_mean(r['folds']['n_evasions'], r)

    @staticmethod
    def attack(r):
        tmp = r['attack']['name']
        if tmp == 'ProjectedGradientDescent':
            return 'PGD'
        if tmp == 'ZooAttack':
            return 'Zoo'
        if tmp == 'HopSkipJump':
            return 'HSJ'
        if tmp == 'CPGD':
            if not r['attack']['config']['args']['enable_constraints']:
                return 'CPGDᴿ'
        return tmp

    def fn_pattern(self, file_ext, pattern, out_dir=None):
        flat_name = self.directory.replace('/', '_')
        file_name = f'__{pattern}_{flat_name}'
        return path.join(out_dir, f'{file_name}.{file_ext}')
