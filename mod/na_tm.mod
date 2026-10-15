COMMENT
Corrente de Na+ rapida (Traub & Miles 1991), na forma usada por Pospischil et al. (2008,
Biol Cybern 99:427) para neuronios corticais:  I = gbar m^3 h (v - ena).
Taxas definidas a 36 degC; fator Q10 = 3 para outras temperaturas.
VT ajusta o limiar de disparo (-56,2 mV no RS de Pospischil et al.).
ENDCOMMENT

NEURON {
    SUFFIX na_tm
    USEION na READ ena WRITE ina
    RANGE gbar, vt, ina
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.05 (S/cm2)
    vt = -56.2 (mV)
    q10 = 3
    tref = 36 (degC)
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA/cm2)
    celsius (degC)
    am (/ms) bm (/ms) ah (/ms) bh (/ms)
    tadj
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gbar * m * m * m * h * (v - ena)
}

INITIAL {
    rates(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
}

DERIVATIVE states {
    rates(v)
    m' = am * (1 - m) - bm * m
    h' = ah * (1 - h) - bh * h
}

PROCEDURE rates(v (mV)) {
    LOCAL u
    tadj = q10 ^ ((celsius - tref) / 10)
    u = v - vt
    am = tadj * 0.32 * vtrap(-(u - 13), 4)
    bm = tadj * 0.28 * vtrap(u - 40, 5)
    ah = tadj * 0.128 * exp(-(u - 17) / 18)
    bh = tadj * 4 / (1 + exp(-(u - 40) / 5))
}

FUNCTION vtrap(x (mV), y (mV)) (mV) {
    : x / (exp(x/y) - 1), com limite continuo em x = 0
    if (fabs(x / y) < 1e-6) {
        vtrap = y * (1 - x / y / 2)
    } else {
        vtrap = x / (exp(x / y) - 1)
    }
}