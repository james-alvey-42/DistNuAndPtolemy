module Ptolemy
using SpecialFunctions: gamma
using QuadGK: quadgk

function F(Ee::Float64)
    me = 511 * 1e6 # meV
    pe = sqrt(Ee^2 - me^2)
    Z = 2
    alpha = 1 / 137.036
    eta = Z * alpha * Ee / pe
    return 2 * π * eta / (1 - exp(- 2 * π * eta))
end

function sigmav(Ee::Float64)
    GF = 1.166 * 1e-29 # meV^-2
    m3He = 2808.391 # MeV
    m3H = 2808.921 # MeV
    me = 511 * 1e6 # meV
    pe = sqrt(Ee^2 - me^2)
    F2 = 0.9987
    GT2 = 2.788
    gA = 1.2695 # Footnote 5
    return (GF^2 / (2 * π)) * F(Ee) * (m3He/m3H) * Ee * pe * (F2 + gA^2 * GT2)
end

function Ue(order::String="normal")
    s12sq = 0.32
    if order == "normal"
        s13sq = 2.16 * 1e-2
    else
        s13sq = 2.22 * 1e-2
    end
    c12sq = 1 - s12sq
    c13sq = 1 - s13sq
    return [c12sq * c13sq, s12sq * c13sq, s13sq]
end

function fc(mi_meV::Float64, dist::String="Gaussian")
    if dist == "Gaussian"
        return 0.
    else
        return 76.5 * (mi_meV / 1000)^(2.21)
    end
end

function GammaCNB(Ee::Float64, n0::Float64, NT::Float64, order::String="normal")
    Ue_cnb = Ue(order)
    Gamma1 = NT * sigmav(Ee) * n0 * Ue_cnb[1]
    Gamma2 = NT * sigmav(Ee) * n0 * Ue_cnb[2]
    Gamma3 = NT * sigmav(Ee) * n0 * Ue_cnb[3]
    return [Gamma1, Gamma2, Gamma3]
end

function mbetasq(m1::Float64, m2::Float64, m3::Float64, order::String="normal")
    Ue_cnb = Ue(order)
    return Ue_cnb[1] * m1^2 + Ue_cnb[2] * m2^2 + Ue_cnb[3] * m3^2
end

function Eend0()
    m3He = 2808.391 * 1e9 # meV
    m3H = 2808.921 * 1e9 # meV
    me = 511 * 1e6 # meV
    return (1/(2 * m3H)) * (m3H^2 + me^2 - m3He^2)
end

function H(Ee::Float64, mi::Float64, DEEnd::Float64=0.0)
    if Eend0() + DEEnd - Ee - mi > 0.
        y = Eend0() + DEEnd- Ee - mi
    else
        y = 0.
    end
    m3He = 2808.391 * 1e9 # meV
    m3H = 2808.921 * 1e9 # meV
    me = 511 * 1e6 # meV
    return (1 - me^2/(Ee * m3H)) * sqrt(y * (y + (2 * mi * m3He)/(m3H))) * (1 - 2 * Ee / m3H + me^2 / m3H^2)^(-2) * (y + (mi/m3H)*(m3He + mi))
end

function dGbdEe(m1::Float64, m2::Float64, m3::Float64, Ee::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0)
    me = 511 * 1e6 # meV
    pe = sqrt(Ee^2 - me^2)
    pnu = pe
    Enu = sqrt(pnu^2 + mbetasq(m1, m2, m3, order))
    vnu = pnu / Enu
    sigma = sigmav(Ee) / vnu
    Ue_cnb = Ue(order)
    return sigma * NT / π^2 * (Ue_cnb[1] * H(Ee, m1, DEEnd) + Ue_cnb[2] * H(Ee, m2, DEEnd) + Ue_cnb[3] * H(Ee, m3, DEEnd))
end

function CNBfactor(mi::Float64, mlightest::Float64, Ee::Float64, delta::Float64, DEEnd::Float64=0.0)
    Eend = Eend0() - mlightest + DEEnd
    return exp(-(Ee - (Eend + mi + mlightest))^2 / (2 * delta^2/(8 * log(2))))
end

function dGtCNBdE(delta::Float64, m1::Float64, m2::Float64, m3::Float64, Ee::Float64, n0::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0)
    mlightest = min(m1, m2, m3)
    Gamma = GammaCNB(Ee, n0, NT, order)
    E = CNBfactor.([m1, m2, m3], mlightest, Ee, delta, DEEnd)
    return sqrt(8 * log(2)) / (sqrt(2 * π) * delta) * (Gamma[1] * E[1] + Gamma[2] * E[2] + Gamma[3] * E[3])
end

function dGtbdE(delta::Float64, m1::Float64, m2::Float64, m3::Float64, Ee::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0)
    prefactor = sqrt(8 * log(2)) / (sqrt(2 * π) * delta)
    function integrand(x::Float64)
        eVyr_factor = 4.794049023619834e+22
        return eVyr_factor * dGbdEe(m1, m2, m3, x, NT, order, DEEnd) * exp(-(Ee - x)^2/(2 * delta^2 / (8 * log(2))))
    end
    integral, err = quadgk(integrand, Ee - 20 * delta, Ee + 20 * delta)
    return prefactor * integral
end

function masses(mlightest::Float64, order::String="normal")
    if order == "normal"
        m1 = mlightest
        m2 = sqrt(m1^2 + 7.55 * 1e-5 * 1e6)
        m3 = sqrt(m1^2 + 2.50 * 1e-3 * 1e6)
    else
        m3 = mlightest
        m1 = sqrt(m3^2 + 2.42 * 1e-3 * 1e6)
        m2 = sqrt(m1^2 + 7.55 * 1e-5 * 1e6)
    end
    return [m1, m2, m3]
end

function N_beta(Ei::Float64, Tyrs::Float64, delta::Float64, mlightest::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0)
    m = masses(mlightest, order)
    function integrand(x::Float64)
        dGtbdE(delta, m[1], m[2], m[3], x, NT, order, DEEnd)
    end
    integral, err = quadgk(integrand, Ei - 0.5 * delta, Ei + 0.5 * delta)
    return Tyrs * 1e-3 * integral
end

function N_CNB(Ei::Float64, Tyrs::Float64, delta::Float64, mlightest::Float64, n0::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0, cDM::Float64=1.)
    m = masses(mlightest, order)
    eVyr_factor = 4.794049023619834e+22
    n0 = n0 * 7.685803257085992e-06
    function integrand(x::Float64)
        return eVyr_factor * dGtCNBdE(delta, m[1], m[2], m[3], x, n0, NT, order, DEEnd)
    end
    integral, err = quadgk(integrand, Ei - 0.5 * delta, Ei + 0.5 * delta)
    return cDM * Tyrs * 1e-3 * integral
end

function N_total(Ei::Float64, Tyrs::Float64, delta::Float64, mlightest::Float64, n0::Float64, NT::Float64, order::String="normal", DEEnd::Float64=0.0, gamma_b::Float64=1e-5, cDM::Float64=1.)
    return N_beta(Ei, Tyrs, delta, mlightest, NT, order, DEEnd) + N_CNB(Ei, Tyrs, delta, mlightest, n0, NT, order, DEEnd, cDM) + (gamma_b * 31558149.7635456) * Tyrs / (15 * 1e3 / 50)
end

end