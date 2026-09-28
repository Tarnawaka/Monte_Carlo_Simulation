# -*- coding: utf-8 -*-
"""
=============================================================================
 COSMIC RAY LAB -- PART A
 Monte Carlo simulation of the Cosmic Ray Telescope (CRT)
=============================================================================

 This program is a Python translation and extension of the C program
 "cosmic.c" listed in Appendix A of

     E. Harpell, W. Langeveld, D. McShurley, S. Shapiro, J. Venuti,
     "The CCRT: an inexpensive cosmic ray muon detector",
     SLAC-TN-95-1 (1995).

 WHAT THE ORIGINAL C PROGRAM DOES
 --------------------------------
 It models the telescope as two identical rectangular scintillator paddles
 (width w, length l) mounted one directly above the other, separated by a
 distance d.  It then "throws" cosmic rays:

    1. pick a random point (x_top, y_top) uniformly on the TOP paddle
       ("translation invariance": every point is equally likely to be hit)
    2. pick a random direction (theta, phi) with the correct cos^2(theta)
       angular distribution
    3. follow the track in a straight line down a distance d
    4. if it lands inside the BOTTOM paddle -> HIT, otherwise -> MISS

 The ratio hits/tries is the ACCEPTANCE of the detector.  This is the
 hit/miss (acceptance-rejection) Monte Carlo method from the MC lab.

 WHAT THIS PROGRAM ADDS (the part SLAC "left as an exercise", sec. 2.3)
 ---------------------------------------------------------------------
 The lab requires the rate as a function of the ZENITH angle, so the whole
 telescope must be tilted by an angle alpha away from vertical.  Two
 estimates are produced for every angle:

   (1) SIMPLE SCALING  (what the lab handout suggests):
           N(alpha) = A * cos^2(alpha) + B
       where A is the counts obtained from the vertical (alpha = 0) run and
       B is a constant background of accidental / "fake" coincidences.

   (2) FULL TILTED-GEOMETRY MC  (more honest, and more fun):
       the detector axes are physically rotated by alpha, the cos^2(theta)
       flux is still defined with respect to the TRUE vertical, and each
       generated track is weighted by |p_hat . n_hat| (the foreshortening
       factor, eq. 2.7-2.8 of the SLAC note).  This automatically produces
       the cos^2 fall-off *and* the deviations from it near the horizon.

 Finally the program generates a set of PSEUDO-DATA (Poisson-fluctuated
 counts for a realistic 30-minute run at each angle), fits them to
 A*cos^2(theta) + B, and plots everything -- i.e. exactly the analysis you
 will perform on the real data in Part C.

 A SUBTLETY IN THE ORIGINAL C CODE  (worth a paragraph in your report)
 --------------------------------------------------------------------
 cosmic.c counts every accepted track with weight 1, having thrown
 cos(theta) ~ cos^2(theta) d(cos theta).  Its acceptance is therefore

        acc_SLAC = Int[ cos^2(theta) f_hit dOmega ] / Int[ cos^2 dOmega ]

 but it then multiplies that acceptance by the rate through the top paddle,
 1.04*w*l, which was obtained from Int[ j cos(theta) dOmega ] -- i.e. WITH
 the foreshortening factor cos(theta) of eq. (2.3).  The two halves of the
 calculation use different weights.  The flux of muons actually crossing a
 flat paddle goes as cos^3(theta) d(cos theta), not cos^2, so the correct
 acceptance is

        acc_true = Int[ cos^3(theta) f_hit dOmega ] / Int[ cos^3 dOmega ]

 Because f_hit and the extra cos(theta) both favour near-vertical tracks,
 the C code UNDERESTIMATES the coincidence rate.  For our paddles:

        acc_SLAC = 0.0178  ->  0.92 counts/min
        acc_true = 0.0234  ->  1.22 counts/min
        wl/d^2 formula     ->  1.27 counts/min      (eq. 2.5 / 4.2)

 With the correct weight the Monte Carlo and the simple analytic estimate
 agree to a few percent, as one expects for d >> w,l.  The claim in sec.
 2.2 of the note that "the approximation is not extremely accurate" is
 largely an artefact of the missing cos(theta).  The tilted simulation in
 this program uses the correct weight; the direct translation below is kept
 unchanged so that the numbers printed in the note can be reproduced.

 Run in Spyder:  just press F5.  Everything below is standard
 numpy / scipy / matplotlib.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# 1.  USER SETTINGS -- change these to match your apparatus / run plan
# =============================================================================

W_PADDLE = 5.0      # width  of the scintillator paddles            [cm]
L_PADDLE = 10.0     # length of the scintillator paddles            [cm]
D_SEP    = 36.0     # separation between the two paddles            [cm]

# Zenith angles of the telescope (0 = pointing straight up).
ANGLES_DEG = np.array([-90., -75., -50., -25., 0., 25., 50., 75., 90.])

N_EVENTS   = 500000     # cosmic rays thrown per angle
LIVE_TIME  = 30.0       # counting time per angle, minutes (one class period)
BKG_RATE   = 0.05       # accidental "fake" coincidence rate  [counts/min]
                        # -> this is the constant "B" of the fit model

# Cosmic ray flux at the zenith, from the Particle Data Group / eq. (2.2)
# of the SLAC note:  j(theta) = J0 * cos^2(theta)
J0 = 0.66               # [cm^-2 min^-1 sr^-1]

RNG_SEED = 20250826     # fixed seed -> the run is reproducible


# =============================================================================
# 2.  SAMPLING THE COSMIC RAY DIRECTION
# =============================================================================
def throw_directions(n, rng):
    """
    Generate n cosmic ray directions with the correct angular distribution.

    We need cos(theta) distributed as cos^2(theta) d(cos theta).  Using the
    INVERSE TRANSFORM method (see the C code comments):

        P(x) dx = 3 x^2 dx = d(x^3) = dR      with R uniform on [0,1)
        =>  x = cos(theta) = R^(1/3)

    The factor 3 only normalises the pdf, it does not change the sampling
    rule.  phi is thrown flat on [0, 2*pi).

    Returns the unit vector p_hat = (px, py, pz) POINTING BACK ALONG THE
    TRACK (i.e. towards where the muon came from), exactly as in eq. (2.6):

        p_hat = (sin(t)cos(p), sin(t)sin(p), cos(t))

    The muon itself travels along -p_hat, which is why the extrapolation
    below subtracts the direction tangents.
    """
    cost = rng.random(n) ** (1.0 / 3.0)          # cos(theta) ~ cos^2 distribution
    sint = np.sqrt(1.0 - cost * cost)
    phi  = rng.random(n) * 2.0 * np.pi

    px = sint * np.cos(phi)
    py = sint * np.sin(phi)
    pz = cost
    return px, py, pz


# =============================================================================
# 3.  DIRECT TRANSLATION OF cosmic.c  (vertical telescope only)
# =============================================================================
def cosmic_slac(w, l, d, n, rng, verbose=True):
    """
    Line-by-line Python version of the C routine in Appendix A of
    SLAC-TN-95-1.  Kept separate from the tilted version so that the
    numbers of the paper can be reproduced exactly as a cross-check.

    Returns (acceptance, error_on_acceptance, rate_top, rate_both).
    """
    # ---- throw the directions (C: cost = pow(drand48(), 0.333333)) ----------
    px, py, pz = throw_directions(n, rng)

    # ---- direction tangents (C: tant = sint/cost, tantx, tanty) ------------
    #      tan(theta)*sin(phi) = px/pz ,  tan(theta)*cos(phi) = py/pz
    #      (the C code swaps sin/cos between x and y; irrelevant by symmetry)
    tantx = px / pz
    tanty = py / pz

    # ---- random impact point on the TOP paddle -----------------------------
    xtop = rng.random(n) * w
    ytop = rng.random(n) * l

    # ---- extrapolate a straight line down to the BOTTOM paddle -------------
    xbot = xtop - tantx * d
    ybot = ytop - tanty * d

    # ---- see if it hits (C: two "continue" statements) ---------------------
    hit = (xbot >= 0.0) & (xbot <= w) & (ybot >= 0.0) & (ybot <= l)
    nhit = int(hit.sum())

    frac  = nhit / n                       # acceptance
    dfrac = np.sqrt(nhit) / n              # error = sqrt(N_accepted)/N_tries

    # ---- rates.  1.04 /min/cm^2 is the integrated vertical flux, eq. (2.3) -
    flux_horizontal = J0 * np.pi / 2.0     # = 1.04 cm^-2 min^-1
    rate_top  = flux_horizontal * w * l
    rate_both = rate_top * frac
    drate     = flux_horizontal * w * l * dfrac

    if verbose:
        print("--- direct translation of cosmic.c -------------------------")
        print("Width {:g} cm   Length {:g} cm   Distance {:g} cm".format(w, l, d))
        print("Number of events {:d}".format(n))
        print("Had {:d} hits out of {:d} cosmics".format(nhit, n))
        print("Accepted cosmic ray fraction:        {:.5g} +/- {:.3g}"
              .format(frac, dfrac))
        print("Cosmic ray rate through top panel:   {:.6g} /min".format(rate_top))
        print("Cosmic ray rate through both panels: {:.6g} +/- {:.4g} /min"
              .format(rate_both, drate))
        if (d > 2 * w) and (d > 2 * l):
            print("Simple estimate (wl/d^2 formula):    {:.6g} /min"
                  .format(J0 * (w * l) * (w * l) / (d * d)))
        else:
            print("Simple estimate not reliable")
        print("------------------------------------------------------------")

    return frac, dfrac, rate_top, rate_both


# =============================================================================
# 4.  THE TILTED TELESCOPE  (the "exercise" of SLAC sec. 2.3)
# =============================================================================
def cosmic_tilted(alpha_deg, w, l, d, n, rng):
    """
    Simulate the telescope tilted by alpha away from the vertical, rotating
    about the x-axis.

    DETECTOR FRAME.  The paddles stay rigidly attached to each other, so we
    define a rotated orthonormal set of axes:

        u_hat = (1,          0,          0     )   <- along the WIDTH  w
        v_hat = (0,  cos(alpha), -sin(alpha)   )   <- along the LENGTH l
        n_hat = (0,  sin(alpha),  cos(alpha)   )   <- paddle NORMAL

    For alpha = 0, n_hat = (0,0,1) = straight up, and we recover the
    original vertical geometry.  The top paddle sits at n = 0, the bottom
    one at n = -d.

    FLUX WEIGHTING.  The number of muons per minute crossing the top paddle
    with direction in dOmega is

        dN = j(theta) * |p_hat . n_hat| * dOmega * Area          (eq. 2.8)

    where theta is still measured from the TRUE vertical, and the
    foreshortening factor is now the dot product with the tilted normal
    instead of a simple cos(theta).  Since we already sample
    cos(theta) ~ cos^2(theta) d(cos theta) and phi flat,

        integral( j |p.n| dOmega ) = J0 * (2*pi/3) * < |p.n| >

    where < > is the plain average over the thrown sample.  Checking at
    alpha = 0:  J0*(2pi/3)*<cos theta> = 0.66*(2pi/3)*(3/4) = 1.04, i.e.
    exactly the flux of eq. (2.3).  So this is a WEIGHTED Monte Carlo:
    each track carries weight |p.n| instead of counting 1.

    THE FACTOR OF TWO NEAR THE HORIZON.  When the telescope is nearly
    horizontal a muon can cross the two paddles going "left to right" or
    "right to left".  Tracks with p.n > 0 enter through the top paddle;
    tracks with p.n < 0 enter through the bottom one.  Both are genuine
    coincidences and both are counted here by using |p.n| and starting the
    track on whichever paddle is hit first.  This is what doubles the
    accepted solid angle at alpha = 90 deg, as warned about in sec. 2.3.

    Returns a dictionary with the acceptance and the predicted rates.
    """
    a = np.radians(alpha_deg)

    u_hat = np.array([1.0, 0.0, 0.0])
    v_hat = np.array([0.0, np.cos(a), -np.sin(a)])
    n_hat = np.array([0.0, np.sin(a), np.cos(a)])

    px, py, pz = throw_directions(n, rng)

    # components of the track direction along the detector axes
    p_u = px * u_hat[0] + py * u_hat[1] + pz * u_hat[2]
    p_v = px * v_hat[0] + py * v_hat[1] + pz * v_hat[2]
    p_n = px * n_hat[0] + py * n_hat[1] + pz * n_hat[2]

    absn = np.abs(p_n)
    good = absn > 1.0e-12          # tracks parallel to the paddles never cross

    # random entry point on the first paddle, in detector coordinates
    u0 = rng.random(n) * w
    v0 = rng.random(n) * l

    # path length needed to go from one paddle to the other
    t = np.where(good, d / np.where(good, absn, 1.0), np.inf)

    # extrapolate to the far paddle (same algebra as the C code, but in the
    # rotated frame)
    u1 = u0 - t * p_u
    v1 = v0 - t * p_v

    hit = good & (u1 >= 0.0) & (u1 <= w) & (v1 >= 0.0) & (v1 <= l)

    # ---- weighted estimators ------------------------------------------------
    norm = J0 * (2.0 * np.pi / 3.0) * w * l      # [counts/min] normalisation

    wgt_all = absn                 # weight for "crossed the first paddle"
    wgt_hit = absn * hit           # weight for "crossed BOTH paddles"

    rate_top  = norm * wgt_all.mean()
    rate_both = norm * wgt_hit.mean()
    # statistical error of the MC estimate itself (not of the fake data)
    drate     = norm * wgt_hit.std(ddof=1) / np.sqrt(n)

    acceptance = rate_both / rate_top if rate_top > 0 else 0.0

    return {"alpha": alpha_deg,
            "acceptance": acceptance,
            "rate_top": rate_top,
            "rate_both": rate_both,
            "drate_mc": drate,
            "n_hit_raw": int(hit.sum())}


# =============================================================================
# 5.  FIT MODEL
# =============================================================================
def flux_model(theta_deg, A, B):
    """
    The model the data are fitted to:

        N(theta) = A * cos^2(theta) + B

    A  = counts at the zenith (contains the flux, the acceptance and the
         live time all rolled into one number)
    B  = flat background of accidental coincidences / noise
    """
    return A * np.cos(np.radians(theta_deg)) ** 2 + B


# =============================================================================
# 6.  MAIN PROGRAM
# =============================================================================
def main():
    rng = np.random.default_rng(RNG_SEED)

    # -- 6a. sanity check: reproduce the example printed in the SLAC note ----
    print("=" * 62)
    print("CROSS-CHECK: SLAC-TN-95-1 example  (w=14, l=15.2, d=61.6 cm)")
    print("The note quotes 266 hits / 10000, acceptance 0.0266 +/- 0.00163,")
    print("5.89 +/- 0.36 counts/min through both panels.")
    print("=" * 62)
    cosmic_slac(14.0, 15.2, 61.6, 10000, rng)
    print()

    # -- 6b. our detector, vertical -----------------------------------------
    print("=" * 62)
    print("OUR TELESCOPE  (w = {:g} cm, l = {:g} cm, d = {:g} cm), vertical"
          .format(W_PADDLE, L_PADDLE, D_SEP))
    print("=" * 62)
    frac0, _, _, rate0_slac = cosmic_slac(W_PADDLE, L_PADDLE, D_SEP,
                                          N_EVENTS, rng)
    vert = cosmic_tilted(0.0, W_PADDLE, L_PADDLE, D_SEP, N_EVENTS, rng)
    print("Same geometry, but with the correct cos^3 projection weight:")
    print("   acceptance {:.5f}  ->  {:.4f} counts/min"
          .format(vert["acceptance"], vert["rate_both"]))
    print("   (cosmic.c gives {:.5f} -> {:.4f} counts/min; see the header "
          "of this file)".format(frac0, rate0_slac))
    print()

    # -- 6c. full angular scan ----------------------------------------------
    print("=" * 78)
    print("ANGULAR SCAN -- tilted-geometry Monte Carlo, {:d} events per angle"
          .format(N_EVENTS))
    print("=" * 78)
    print("{:>7s} {:>12s} {:>13s} {:>13s} {:>11s} {:>10s}"
          .format("angle", "acceptance", "rate_top", "rate_both",
                  "expected", "simulated"))
    print("{:>7s} {:>12s} {:>13s} {:>13s} {:>11s} {:>10s}"
          .format("[deg]", "", "[/min]", "[/min]",
                  "counts", "counts"))
    print("-" * 78)

    results   = []
    counts    = np.zeros(len(ANGLES_DEG), dtype=int)
    expected  = np.zeros(len(ANGLES_DEG))
    rate_mc   = np.zeros(len(ANGLES_DEG))

    for i, ang in enumerate(ANGLES_DEG):
        r = cosmic_tilted(ang, W_PADDLE, L_PADDLE, D_SEP, N_EVENTS, rng)
        results.append(r)
        rate_mc[i] = r["rate_both"]

        # ---- turn the predicted rate into PSEUDO-DATA ----------------------
        # mean number of counts in one run = (signal + background) * live time
        mu = (r["rate_both"] + BKG_RATE) * LIVE_TIME
        expected[i] = mu
        # a real counting experiment fluctuates Poisson-wise about that mean
        counts[i] = rng.poisson(mu)

        print("{:7.1f} {:12.5f} {:13.3f} {:13.4f} {:11.1f} {:10d}"
              .format(ang, r["acceptance"], r["rate_top"], r["rate_both"],
                      mu, counts[i]))
    print("-" * 78)
    print("(live time = {:g} min per angle, background = {:g} counts/min)"
          .format(LIVE_TIME, BKG_RATE))
    print()

    # -- 6d. the "simple scaling" prescription from the lab handout ----------
    i0 = int(np.argmin(np.abs(ANGLES_DEG)))          # index of the 0 deg point
    A_simple = rate_mc[i0] * LIVE_TIME               # counts at zenith
    B_simple = BKG_RATE * LIVE_TIME
    print("Simple prescription  N(theta) = A cos^2(theta) + B  with")
    print("   A = (rate at 0 deg) x T = {:.2f} counts".format(A_simple))
    print("   B = (background)    x T = {:.2f} counts".format(B_simple))
    print()

    # -- 6e. fit the pseudo-data --------------------------------------------
    sigma = np.sqrt(np.where(counts > 0, counts, 1.0))   # Poisson errors
    popt, pcov = curve_fit(flux_model, ANGLES_DEG, counts,
                           p0=[A_simple, B_simple], sigma=sigma,
                           absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    chi2 = np.sum(((counts - flux_model(ANGLES_DEG, *popt)) / sigma) ** 2)
    ndf  = len(ANGLES_DEG) - 2

    print("=" * 62)
    print("FIT OF THE SIMULATED DATA TO  N = A cos^2(theta) + B")
    print("=" * 62)
    print("   A = {:8.2f} +/- {:.2f} counts   (input {:.2f})"
          .format(popt[0], perr[0], A_simple))
    print("   B = {:8.2f} +/- {:.2f} counts   (input {:.2f})"
          .format(popt[1], perr[1], B_simple))
    print("   chi2/ndf = {:.2f}/{:d} = {:.2f}".format(chi2, ndf, chi2 / ndf))
    print("=" * 62)

    # -- 6f. save the pseudo-data so it can be re-analysed later -------------
    header = ("angle_deg,live_time_min,counts,sigma_counts,"
              "rate_per_min,sigma_rate,mc_rate_per_min")
    table = np.column_stack([ANGLES_DEG,
                             np.full_like(ANGLES_DEG, LIVE_TIME),
                             counts,
                             sigma,
                             counts / LIVE_TIME,
                             sigma / LIVE_TIME,
                             rate_mc])
    np.savetxt("PartA_simulated_data.csv", table, delimiter=",",
               header=header, comments="", fmt="%.6g")
    print("\nPseudo-data written to PartA_simulated_data.csv")

    # =========================================================================
    # 7.  PLOTS
    # =========================================================================
    theta_fine = np.linspace(-90.0, 90.0, 721)

    fig, ax = plt.subplots(2, 1, figsize=(8.5, 9.0),
                           gridspec_kw={"height_ratios": [2, 1]})

    # ---- top panel: counts vs angle ----------------------------------------
    ax[0].errorbar(ANGLES_DEG, counts, yerr=sigma, fmt="o", color="k",
                   capsize=3, label="simulated data (Poisson)")
    ax[0].plot(theta_fine, flux_model(theta_fine, *popt), "-", color="crimson",
               label=r"fit  $A\cos^2\theta+B$" +
                     "\n" + r"$A$={:.1f}$\pm${:.1f}, $B$={:.1f}$\pm${:.1f}"
                     .format(popt[0], perr[0], popt[1], perr[1]))
    ax[0].plot(theta_fine, flux_model(theta_fine, A_simple, B_simple), "--",
               color="royalblue",
               label=r"simple scaling of the $\theta=0$ run")
    ax[0].plot(ANGLES_DEG, expected, "s", mfc="none", color="darkgreen",
               label="full tilted-geometry MC prediction")
    ax[0].axhline(B_simple, ls=":", color="grey", lw=1)
    ax[0].set_xlabel(r"zenith angle  $\theta$  [deg]")
    ax[0].set_ylabel("counts in {:g} min".format(LIVE_TIME))
    ax[0].set_title("Part A: simulated cosmic ray flux vs. zenith angle\n"
                    "w = {:g} cm, l = {:g} cm, d = {:g} cm"
                    .format(W_PADDLE, L_PADDLE, D_SEP))
    ax[0].set_xticks(ANGLES_DEG)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # ---- bottom panel: residuals -------------------------------------------
    resid = (counts - flux_model(ANGLES_DEG, *popt)) / sigma
    ax[1].axhline(0.0, color="crimson")
    ax[1].errorbar(ANGLES_DEG, resid, yerr=1.0, fmt="o", color="k", capsize=3)
    ax[1].set_xlabel(r"zenith angle  $\theta$  [deg]")
    ax[1].set_ylabel(r"(data $-$ fit)/$\sigma$")
    ax[1].set_xticks(ANGLES_DEG)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("PartA_flux_vs_angle.png", dpi=150)

    # ---- a second figure: acceptance vs angle -------------------------------
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
    acc = np.array([r["acceptance"] for r in results])
    ax2.plot(ANGLES_DEG, acc, "o-", color="darkorange")
    ax2.set_xlabel(r"zenith angle  $\theta$  [deg]")
    ax2.set_ylabel("geometric acceptance")
    ax2.set_title("Fraction of muons crossing the top paddle that also\n"
                  "cross the bottom one")
    ax2.set_xticks(ANGLES_DEG)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("PartA_acceptance.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
