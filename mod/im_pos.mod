COMMENT
Corrente M (K+ lenta, nao inativante; Yamada et al. 1989, como em Pospischil et al. 2008):
I = gbar p (v - ek);  p_inf = 1/(1+exp(-(v+35)/10));
tau_p = taumax / (3.3 exp((v+35)/20) + exp(-(v+35)/20)).
Responsavel pela adaptacao da frequencia de disparo nos piramidais (RS). Q10 = 2,3 em tau.
ENDCOMMENT

NEURON {
    SUFFIX im_pos
    USEION k READ ek WRITE ik
    RANGE gbar, taumax, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 7e-5 (S/cm2)
    taumax = 1000 (ms)
    q10 = 2.3
    tref = 36 (degC)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    celsius (degC)
    pinf
    taup (ms)
}

STATE { p }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * p * (v - ek)
}

INITIAL {
    rates(v)
    p = pinf
}

DERIVATIVE states {
    rates(v)
    p' = (pinf - p) / taup
}

PROCEDURE rates(v (mV)) {
    LOCAL tadj
    tadj = q10 ^ ((celsius - tref) / 10)
    pinf = 1 / (1 + exp(-(v + 35) / 10))
    taup = taumax / (3.3 * exp((v + 35) / 20) + exp(-(v + 35) / 20)) / tadj
}