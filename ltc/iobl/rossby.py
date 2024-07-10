import gsw
import numpy as np
from functools import partial
from scipy.optimize import newton_krylov
from ltc.iobl.model import interface_fluxes


def shape(x):
    if isinstance(x, np.ndarray):
        return x.shape
    else:
        return (1,)


def zero_div(a, b):
    """return inf when dividing by zero"""
    return np.divide(
        a, b, out=np.inf + np.zeros(np.broadcast_shapes(shape(a), shape(b)))
    )


def rossby_similarity(
    ustar0,
    wb0,
    latitude,
    z,
    z0,
    Lambdastar,
    kappa,
    RC,
):
    """Following Section 4.2"""
    assert np.all(z <= 0)  # z positive upward

    f = np.abs(gsw.f(latitude))
    sign = 1 if latitude > 0 else -1

    delta = np.sqrt(sign * 1j / Lambdastar)
    xi_sl = -Lambdastar / kappa

    L0 = zero_div(ustar0**3, wb0)  # obukhov
    mustar = ustar0 / f / L0
    etastar = np.sqrt(1 / (1 + Lambdastar * ustar0 / (kappa * RC * f * L0)))  # Eq 4.26

    a = kappa * (1 - etastar) / etastar / Lambdastar
    # Eq 4.28
    L_scale = etastar * ustar0 / f
    U_scale = ustar0 / etastar
    stress_scale = ustar0**2
    vel_scale = ustar0 / etastar

    xi = z / L_scale
    xi0 = -z0 / L_scale

    # TODO make this stand out more clearly or send warning or similar
    xi = np.where(xi > xi0, xi0, xi)  # must not integrate higher than z0

    # non-dimensional velocities
    UE = -1j * delta * np.exp(delta * xi_sl)
    U = np.where(
        xi < xi_sl,
        -1j * delta * np.exp(delta * xi_sl),  # Ekman layer
        UE
        - etastar
        / kappa
        * (
            np.log(xi / xi_sl)
            + (delta - a) * (xi - xi_sl)
            - a * delta / 2 * (xi**2 - xi_sl**2)
        ),  # surface layer
    )

    # surface layer shear
    beta = (1 / RC + kappa / mustar / Lambdastar) * (1 - etastar)  # Eq. 4.39
    # K = kappa*(-z)*ustar0 / (1+ beta*(-z)/L0)

    # physical units
    V = U * U_scale

    return V


def rossby_similarity_inverse(
    U,
    Tff,
    Sff,
    Sice,
    hice,
    latitude,
    z,
    z0,
    w_perc,
    qcond,
    use_three_eqn,
    cht,
    chs,
    Lambdastar,
    kappa,
    RC,
    QL,
):
    """Get ustar0 based on U, Sff, Tff"""

    F = partial(
        opt_func,
        U=U,
        Tff=Tff,
        Sff=Sff,
        Sice=Sice,
        hice=hice,
        latitude=latitude,
        z=z,
        z0=z0,
        w_perc=w_perc,
        qcond=qcond,
        use_three_eqn=use_three_eqn,
        cht=cht,
        chs=chs,
        grav=gsw.grav(latitude, hice),
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
        QL=QL,
    )
    return newton_krylov(F, np.sqrt(1.5e-3) * U)


def opt_func(
    ustar0,
    U,
    Tff,
    Sff,
    Sice,
    hice,
    latitude,
    z,
    z0,
    Lambdastar,
    kappa,
    RC,
    w_perc,
    qcond,
    use_three_eqn,
    cht,
    chs,
    grav,
    QL,
):
    """Function to optimize for neutral/stable case.
    When passing into solver, leave only uturb as free parameter."""
    wb0, _, _, _, _, _ = interface_fluxes(
        uturb=ustar0,
        Tff=Tff,
        Sff=Sff,
        hice=hice,
        Sice=Sice,
        w_perc=w_perc,
        qcond=qcond,
        use_three_eqn=use_three_eqn,
        cht=cht,
        chs=chs,
        grav=grav,
        QL=QL,
    )
    Usol = rossby_similarity(
        ustar0,
        wb0,
        latitude,
        z,
        z0,
        Lambdastar=Lambdastar,
        kappa=kappa,
        RC=RC,
    )
    return np.abs(Usol) - np.abs(U)
