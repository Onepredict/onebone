"""Transform PRPS(Phase Resolved Pulse Sequence) format pd data
to PRPD(Phase Resolved Partial Discharge) format.

- Author: Hyunjae Kim
- Contact: hyunjae.kim@onepredict.com
"""

from typing import Tuple

import numpy as np


def ps2pd(ps, range_amp: Tuple[int, int] = (0, 256), resol_amp: int = 128) -> np.ndarray:
    """
    Transform prps(phase resolved pulse sequance) to a prpd(phaes resolved partial discharge)
    by marginalizing time dimension.

    Parameters
    ----------
    ps : array_like of shape (n_resolution_phase, n_timestep)
        The data. Ex: kepco standard=(3600, 128)
    range_amp : tuple (min, max), default=(0, 256)
        Measurement range of PD DAQ. Refers to DAQ manufacture.
    resol_amp : int, default=128
        Desired resolution of amplitude resolution for transformd prpd.

    Returns
    -------
    pd : numpy.ndarray of shape (n_resolution_phase, n_resolution_amplitude)
        The transformed prpd.

    Examples
    --------
    >>> ps = np.random.random([3600,128])
    >>> ps2pd(ps)
    array([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.]])
    """
    if len(ps.shape) != 2:
        raise ValueError("`ps` has to be 2-dimensions.")

    imat = np.eye(resol_amp)
    imat[0, 0] = 0
    bins = np.linspace(range_amp[0], range_amp[1], resol_amp)
    p_digit = np.digitize(ps, bins) - 1
    p_digit_logit = imat[p_digit]
    pd = np.sum(p_digit_logit, axis=0)
    pd = np.fliplr(pd).transpose()
    return pd
