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
        return dict(cht=0.0093, chs=0.0093 / 35)
    elif state.wb0 < 0:
        return dict(cht=0.0057, chs=0.0057)
    else:
        raise ValueError("Need to specify wb0")


def convective_vel_scale(ustar, wb, cml, dml):
    """
    Eqn. 5.8

    TBD use lmax or cml*dml?
    """
    return (ustar**3 - wb * cml * dml) ** (1 / 3)


def alpha_nr(Ri):
    """
    Return the ratio of eddy diffusivities.

    Uses a parameterization of Turner's (1973) relation (McPhee 2008 pp 149, 161)
    with Rc=0.2, b=1.4.

    TBD check those parameters!
    """
    # TBD. what to do with Ri < 0.0079??

    with np.errstate(invalid="ignore"):
        alpha = np.where(
            Ri > 5,
            0.0039,
            np.where(Ri > 0.079, np.exp(-1.5 * np.sqrt(Ri - 0.079)), 1.0),
        )

    return alpha


def get_mixed_layer_depth_N2lim(state, mld_min=4, n2_lim=5e-5):
    """
    Calculate mixed layer depth using a critical N2.

    Arguments
    ---------
        mld_min: upper limit for mixed layer depth (m)
        n2_lim: limiting value for N^2 in the mixed layer (s-2)

    Returns
    -------
        Last depth on mean value grid in the mixed layer

    Notes
    -----
        If the entire model domain is unstratified, returns None.
        Returns depth of mixed layer below ice, not below ocean surface.
    """
    zz, z, n2 = state.zz, state.z, state.n2

    with np.errstate(invalid="ignore"):
        ix_pc = (z >= mld_min) & (n2 > n2_lim)

    if any(ix_pc):
        # iml is the index of the first occurrence of N2>n2_lim on the z grid
        iml = np.argmax(ix_pc)  # True>False; argmax gives first occurence
        # zz grid is staggered: Get last zz value where N2<=n2_lim
        mld = zz[iml - 1]
        return mld
    else:
        return None


def get_mixed_layer_depth_sigth_crit(state, sigth_crit=0.03):
    """
    Calculate mixed layer depth using a critical density increase.

    Arguments
    ---------
        sigth_crit: critical density difference between base of mixed layer
                    and surface density

    Returns
    -------
        Last depth on mean value grid in the mixed layer

    Notes
    -----
        If the entire model domain is unstratified, returns None.
        Returns depth of mixed layer below ice, not below ocean surface.
    """
    zz, rho = state.zz, state.rho

    with np.errstate(invalid="ignore"):
        ix_pc = (rho - rho[0]) > sigth_crit

    if any(ix_pc):
        # iml is the index of the first occurrence of N2>n2_lim on the z grid
        iml = np.argmax(ix_pc)  # True>False; argmax gives first occurence
        # zz grid is staggered: Get last zz value where N2<=n2_lim
        mld = zz[iml - 1]
        return mld
    else:
        return None
