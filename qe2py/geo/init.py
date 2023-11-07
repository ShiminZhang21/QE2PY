import numpy as np

#Initializatino class atom geo

def __init__(self, ion=None, par=None, pos=None, par_units=None, pos_units=None):
    ''' initialize atomgeo instance '''
    self._ion = np.array(ion, dtype=object)
    self._par = np.array(par, dtype=float)
    self._pos = np.array(pos, dtype=float)
    self._par_units = str(par_units).lower()
    self._pos_units = str(pos_units).lower()
    #self._rec = _calc_rec(self._par) #reciprocal lattice cell