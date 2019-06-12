import numpy as np
import pandas as pd

class SymbolFromTransformLog(object):
    def __init__(self, symbol_list):
        self._symbol_list = symbol_list
        self._symbol_assignment_list = []

    def log_symbols(self, rigid_transform_dict):
        sym_assign = np.array([sym(rigid_transform_dict) for sym in self._symbol_list], dtype=np.bool)
        self._symbol_assignment_list.append(sym_assign)

    def to_pandas_dataframe(self):
        sym_name = [sym.name for sym in self._symbol_list]
        sym_assign = np.array(self._symbol_assignment_list, dtype=np.bool)
        data_frame_dict = {}
        for i, sym in enumerate(sym_name):
            data_frame_dict[sym] = sym_assign[:, i]
        return pd.DataFrame(data_frame_dict)


class SymbolFromTransform( object ):
    def __init__(self, name):
        self.name = name

    def __call__(self, rigid_transform_dict):
        '''

        :param rigid_transform_dict: A dictionary where the key is the object name and the value is a drake rigid
            transform object.
        :return:
        '''
        raise NotImplementedError


class SymbolL2Close(SymbolFromTransform):
    def __init__(self, name, object1, object2, l2_thresh):
        SymbolFromTransform.__init__(name)
        self._object1 = object1
        self._object2 = object2
        self._l2_thresh = l2_thresh

    def __call__(self, rigid_transform_dict):
        t1 = rigid_transform_dict[self._object1].translation()
        t2 = rigid_transform_dict[self._object2].translation()
        return np.linalg.norm(t1 - t2) < self._l2_thresh

class SymbolL2Close(SymbolFromTransform):
    def __init__(self, name, object_name, position, delta):
        SymbolFromTransform.__init__(name)
        self._object_name = object_name
        self._position = position
        self._delta = delta

    def __call__(self, rigid_transform_dict):
        t = rigid_transform_dict[self._object_name].translation()
        return np.linalg.norm(t - self._position) < self._delta
