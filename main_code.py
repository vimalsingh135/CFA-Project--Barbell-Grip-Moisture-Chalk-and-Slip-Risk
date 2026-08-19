import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def p_ws_Tetens(T_K: float) -> float:
    """Saturation vapor pressure (Pa), Magnus–Tetens form."""
    T_C = T_K - 273.15
    return 610.94 * math.exp((17.625 * T_C) / (T_C + 243.04))


def k_m_from_correlation(u, L, Dv, nu):
    """Air-side mass transfer coefficient (m/s) from Sherwood correlation."""
    Re = max(u * L / nu, 1e-12)
    Sc = max(nu / Dv, 1e-12)
    Sh = 0.664 * (Re ** 0.5) * (Sc ** (1.0 / 3.0))
    return Sh * Dv / L


@dataclass
class Params:
    # Environment / geometry
    T: float = 298.0               # K
    RH: float = 0.60               # 0-1
    u: float = 0.05                # m/s (air speed near hand)
    L: float = 0.08                # m (contact length scale)
    Dv: float = 2.5e-5             # m^2/s (H2O vapor in air)
    nu: float = 1.5e-5             # m^2/s (air kinematic viscosity)
    # Liquid film
    rho_w: float = 1000.0          # kg/m^3
    h0: float = 2e-7               # m (nearly dry start)
    # Sources/sinks
    J_sweat: float = 3e-4          # kg/m^2/s
    k_s: float = 1.0               # 1/s (chalk uptake rate)
    M_cap: float = 5e-3            # kg/m^2 (chalk capacity)
    theta0: float = 0.0            # 0-1 (initial chalk saturation)
    # Friction law
    mu_dry: float = 0.9
    mu_wet: float = 0.4
    h_c: float = 5e-6              # m
    n_fr: float = 3.0
    # Mechanics
    N: float = 400.0               # N (normal force)
    F_req: float = 200.0           # N (tangential demand)
    S_min: float = 1.0             # safety threshold
    # Integration
    dt: float = 0.05               # s
    t_max: float = 180.0           # s
    # Chalk management
    theta_chalk_limit: float = 0.8 # re-chalk trigger


# ---------- Fluxes and closures ----------
def evaporation_flux(params: Params) -> float:
    """kg/m^2/s, air-side controlled evaporation."""
    k_m = k_m_from_correlation(params.u, params.L, params.Dv, params.nu)
    Mw = 0.01801528  # kg/mol
    R = 8.314462618  # J/mol/K
    pws = p_ws_Tetens(params.T)
    c_vs = Mw * pws / (R * params.T)                 # surface vapor conc.
    c_vinf = Mw * (params.RH * pws) / (R * params.T) # ambient vapor conc.
    return k_m * max(c_vs - c_vinf, 0.0)


def chalk_flux(h: float, theta: float, params: Params) -> float:
    """kg/m^2/s, lumped chalk uptake with finite capacity."""
    return params.k_s * h * max(1.0 - theta, 0.0)


def mu_of_h(h: float, params: Params) -> float:
    """Friction coefficient as a smooth function of film thickness."""
    return params.mu_wet + (params.mu_dry - params.mu_wet) / (1.0 + (h / params.h_c) ** params.n_fr)


# ---------- Simulator ----------
def simulate(params: Params):
    J_ev_base = evaporation_flux(params)
    k_m_val = k_m_from_correlation(params.u, params.L, params.Dv, params.nu)

    t_steps = int(params.t_max / params.dt) + 1
    t = np.linspace(0.0, params.t_max, t_steps)
    h = np.zeros_like(t)
    theta = np.zeros_like(t)
    mu = np.zeros_like(t)
    S = np.zeros_like(t)

    h[0] = params.h0
    theta[0] = params.theta0
    mu[0] = mu_of_h(h[0], params)
    S[0] = (mu[0] * params.N) / max(params.F_req, 1e-12)

    failure_time = None
    rechalk_time = None

    def f_h(h_val, theta_val):
        return (params.J_sweat - J_ev_base - chalk_flux(h_val, theta_val, params)) / params.rho_w

    def f_theta(h_val, theta_val):
        return chalk_flux(h_val, theta_val, params) / max(params.M_cap, 1e-12)

    # 4th-order Runge–Kutta
    for i in range(1, t_steps):
        h_i = h[i - 1]
        th_i = theta[i - 1]
        dt = params.dt

        k1_h = f_h(h_i, th_i)
        k1_th = f_theta(h_i, th_i)

        k2_h = f_h(h_i + 0.5 * dt * k1_h, th_i + 0.5 * dt * k1_th)
        k2_th = f_theta(h_i + 0.5 * dt * k1_h, th_i + 0.5 * dt * k1_th)

        k3_h = f_h(h_i + 0.5 * dt * k2_h, th_i + 0.5 * dt * k2_th)
        k3_th = f_theta(h_i + 0.5 * dt * k2_h, th_i + 0.5 * dt * k2_th)

        k4_h = f_h(h_i + dt * k3_h, th_i + dt * k3_th)
        k4_th = f_theta(h_i + dt * k3_h, th_i + dt * k3_th)

        h[i] = max(h_i + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h), 0.0)
        theta[i] = min(max(th_i + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th), 0.0), 1.0)

        mu[i] = mu_of_h(h[i], params)
        S[i] = (mu[i] * params.N) / max(params.F_req, 1e-12)

        if failure_time is None and S[i] <= params.S_min:
            failure_time = t[i]
        if rechalk_time is None and theta[i] >= params.theta_chalk_limit:
            rechalk_time = t[i]

    if failure_time is None:
        failure_time = float('inf')
    if rechalk_time is None:
        rechalk_time = float('inf')

    return {
        "t": t, "h": h, "theta": theta, "mu": mu, "S": S,
        "failure_time": failure_time, "rechalk_time": rechalk_time,
        "k_m": k_m_val
    }


# ---------- Plot helper ----------
def save_and_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    outdir = "."
    os.makedirs(outdir, exist_ok=True)

    # Scenarios
    base = Params()
    scenarios = {
        "S1_baseline_humid_still": base,
        "S2_add_small_fan": Params(**{**asdict(base), "u": 0.5}),
        "S3_no_chalk": Params(**{**asdict(base), "M_cap": 1e-12, "theta0": 1.0}),
        "S4_partial_chalk": Params(**{**asdict(base), "theta0": 0.5}),
        "S5_high_sweater": Params(**{**asdict(base), "J_sweat": 1e-3}),
        "S6_stronger_grip": Params(**{**asdict(base), "N": 600.0}),
    }

    # Run simulations
    results = {name: simulate(p) for name, p in scenarios.items()}

    # Summary CSV
    rows = []
    for name, pars in scenarios.items():
        res = results[name]
        rows.append({
            "scenario": name,
            "RH": pars.RH,
            "u (m/s)": pars.u,
            "k_m (m/s)": res["k_m"],
            "J_sweat (kg/m2/s)": pars.J_sweat,
            "M_cap (kg/m2)": pars.M_cap,
            "theta0": pars.theta0,
            "N (N)": pars.N,
            "F_req (N)": pars.F_req,
            "failure_time_s": None if math.isinf(res["failure_time"]) else res["failure_time"],
            "rechalk_time_s": None if math.isinf(res["rechalk_time"]) else res["rechalk_time"],
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(outdir, "grip_summary.csv"), index=False)
    print("Saved grip_summary.csv")

    # Baseline time traces (S1)
    res1 = results["S1_baseline_humid_still"]
    t = res1["t"]

    fig = plt.figure()
    plt.plot(t, res1["h"])
    plt.xlabel("Time (s)")
    plt.ylabel("Film thickness h (m)")
    plt.title("Baseline (S1): Moisture film h(t)")
    save_and_show(fig, os.path.join(outdir, "S1_h_t.png"))

    fig = plt.figure()
    plt.plot(t, res1["theta"])
    plt.xlabel("Time (s)")
    plt.ylabel("Chalk saturation θ")
    plt.title("Baseline (S1): Chalk saturation θ(t)")
    save_and_show(fig, os.path.join(outdir, "S1_theta_t.png"))

    fig = plt.figure()
    plt.plot(t, res1["mu"])
    plt.xlabel("Time (s)")
    plt.ylabel("Friction coefficient μ")
    plt.title("Baseline (S1): Friction μ(t)")
    save_and_show(fig, os.path.join(outdir, "S1_mu_t.png"))

    fig = plt.figure()
    plt.plot(t, res1["S"])
    plt.axhline(y=base.S_min, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Safety margin S(t)")
    plt.title("Baseline (S1): Safety margin S(t)")
    save_and_show(fig, os.path.join(outdir, "S1_S_t.png"))

    # Re-chalk interval vs humidity (still vs fan)
    RH_values = np.linspace(0.4, 0.8, 9)
    t_fail_still = []
    t_fail_fan = []
    for RH in RH_values:
        pars_still = Params(**{**asdict(base), "RH": float(RH), "u": 0.05})
        pars_fan = Params(**{**asdict(base), "RH": float(RH), "u": 0.5})
        r_still = simulate(pars_still)
        r_fan = simulate(pars_fan)
        t_rc_still = min(r_still["failure_time"], r_still["rechalk_time"])
        t_rc_fan = min(r_fan["failure_time"], r_fan["rechalk_time"])
        t_fail_still.append(pars_still.t_max if math.isinf(t_rc_still) else t_rc_still)
        t_fail_fan.append(pars_fan.t_max if math.isinf(t_rc_fan) else t_rc_fan)

    fig = plt.figure()
    plt.plot(RH_values, t_fail_still, marker="o", label="Still air (~0.05 m/s)")
    plt.plot(RH_values, t_fail_fan, marker="s", label="Small fan (~0.5 m/s)")
    plt.xlabel("Relative Humidity (fraction)")
    plt.ylabel("Suggested re-chalk interval (s)")
    plt.title("Re-chalk interval vs humidity")
    plt.legend()
    save_and_show(fig, os.path.join(outdir, "rechalk_vs_humidity.png"))

    # Safe window vs grip force N
    N_values = np.linspace(300, 700, 9)
    t_safe_N = []
    for N in N_values:
        parsN = Params(**{**asdict(base), "N": float(N)})
        rN = simulate(parsN)
        t_rc = min(rN["failure_time"], rN["rechalk_time"])
        t_safe_N.append(parsN.t_max if math.isinf(t_rc) else t_rc)

    fig = plt.figure()
    plt.plot(N_values, t_safe_N, marker="o")
    plt.xlabel("Normal force N (N)")
    plt.ylabel("Safe window / re-chalk interval (s)")
    plt.title("Effect of grip force on safe window")
    save_and_show(fig, os.path.join(outdir, "safe_window_vs_N.png"))

    # Scenario comparison: safety margin time series
    fig = plt.figure()
    for name in ["S1_baseline_humid_still",
                 "S2_add_small_fan",
                 "S3_no_chalk",
                 "S4_partial_chalk",
                 "S5_high_sweater",
                 "S6_stronger_grip"]:
        plt.plot(results[name]["t"], results[name]["S"], label=name)
    plt.axhline(y=base.S_min, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Safety margin S(t)")
    plt.title("Scenario comparison: safety margin over time")
    plt.legend()
    save_and_show(fig, os.path.join(outdir, "scenario_safety_margins.png"))

    print("All figures saved in:", os.path.abspath(outdir))
