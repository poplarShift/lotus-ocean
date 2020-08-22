from copy import deepcopy
import numpy as np


def get_transfer_coefficients(state):
    """
    Output scalar interface transfer coefficients.

    This is a standard scenario for how coefficients can be chosen,
    and should be sufficient for most use cases.

    Arguments
    ---------
        state:

    Returns
    -------
        chs, cht: Salt/heat transfer coefficients

    Notes
    -----
        References: TBD book pages where this is discussed!
    """
    if state.wb0 >= 0:
        return dict(cht=0.0093, chs=0.0093/35)
    elif state.wb0 < 0:
        return dict(cht=0.0057, chs=0.0057)
    else:
        raise ValueError('Need to specify wb0')

def convective_vel_scale(ustar, wb, cml, dml):
    """
    Eqn. 5.8

    TBD use lmax or cml*dml?
    """
    return (ustar**3 - wb * cml * dml)**(1/3)


def alpha_nr(Ri):
    """
    Return the ratio of eddy diffusivities.

    Uses a parameterization of Turner's (1973) relation (McPhee 2008 pp 149, 161)
    with Rc=0.2, b=1.4.

    TBD check those parameters!
    """
    # TBD. what to do with Ri < 0.0079??

    with np.errstate(invalid='ignore'):
        alpha = np.where(Ri>5,
                         0.0039,
                         np.where(Ri>0.079,
                                  np.exp(-1.5 * np.sqrt(Ri-0.079)),
                                  1.
                                  )
                         )

    return alpha
