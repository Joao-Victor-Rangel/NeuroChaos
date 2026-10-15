COMMENT
Corrente de K+ retificadora tardia (Traub & Miles 1991; Pospischil et al. 2008):
I = gbar n^4 (v - ek). Taxas a 36 degC; Q10 = 3.
ENDCOMMENT

NEURON {
    SUFFIX kd_tm
    USEION k READ ek WRITE ik
    RANGE gbar, vt, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.005 (S/cm2)
    vt = -56.2 (mV)
    q10 = 3
    tref = 36 (degC)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    celsius (degC)
    an (/ms) bn (/ms)
    tadj
}

STATE { n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * n * n * n * n * (v - ek)
}

INITIAL {
    rates(v)
    n = an / (an + bn)
}

DERIVATIVE states {
    rates(v)
    n' = an * (1 - n) - bn * n
}

PROCEDURE rates(v (mV)) {
    LOCAL u
    tadj = q10 ^ ((celsius - tref) / 10)
    u = v - vt
    an = tadj * 0.032 * vtrap(-(u - 15), 5)
    bn = tadj * 0.5 * exp(-(u - 10) / 40)
}

FUNCTION vtrap(x (mV), y (mV)) (mV) {
    if (fabs(x / y) < 1e-6) {
        vtrap = y * (1 - x / y / 2)
    } else {
        vtrap = x / (exp(x / y) - 1)
    }
}