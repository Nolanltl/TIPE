import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import os

# A faire
# conservation moment cinetique
# Runge lens
# Comparer avec Odeint
# plus de shema (euler en cas de divergence : spiral )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures_bis")
os.makedirs(FIG_DIR, exist_ok=True)

parameters = {
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "figure.titlesize": 20,
    "figure.figsize": (8, 6),
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "legend.fontsize": 14,
}
plt.rcParams.update(parameters)

couleur = {"Verlet": "green", "RK4": "orange", "Euler": "blue", "Odeint": "red", "Analytique": "gray"}

# Variables globales
G = 1.0  # constante gravitationnelle (unités arbitraires)


# ============================================================
# Partie 1 : 2-corps
# ============================================================


def vitesses_circulaires(r_01, r_02, m1, m2, G=1.0, sens=+1):
    """
    Calcule les vitesses initiales pour une orbite circulaire de deux corps.

    param :
        r_01: position initiale du corps 1 (array-like)
        r_02: position initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        G: constante gravitationnelle (float, défaut=1.0)
        sens: +1 pour sens trigonométrique, -1 pour sens horaire (int)

    return:
        v_01, v_02 les vitesses initiales des deux corps (arrays numpy)
    """

    r_01 = np.asarray(r_01, float)
    r_02 = np.asarray(r_02, float)

    r_rel = r_02 - r_01
    r = np.linalg.norm(r_rel)
    if r == 0:
        raise ValueError("Les positions ne doivent pas coïncider.")

    u_r = r_rel / r

    if sens == +1:  # sens trigo
        u_th = np.array([-u_r[1], u_r[0]])
    else:  # sens horaire
        u_th = np.array([u_r[1], -u_r[0]])

    M = m1 + m2
    v_rel_norm = np.sqrt(G * M / r)
    v_rel = v_rel_norm * u_th

    v_01 = -(m2 / M) * v_rel
    v_02 = (m1 / M) * v_rel

    return v_01, v_02


def position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G=1.0):
    """
    Calcule la position des deux corps à l'instant t en utilisant la solution analytique (orbite circulaire)
    param :
        r_01: position initiale du corps 1 (array-like)
        r_02: position initiale du corps 2 (array-like)
        v_01: vitesse initiale du corps 1 (array-like)
        v_02: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        t: temps (array-like)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1, r2: positions des deux corps à l'instant t (arrays numpy)
        omega: vitesse angulaire de l'orbite (float)"""

    r_01 = np.asarray(r_01, float)
    r_02 = np.asarray(r_02, float)
    v_01 = np.asarray(v_01, float)
    v_02 = np.asarray(v_02, float)

    M = m1 + m2
    R0 = (m1 * r_01 + m2 * r_02) / M
    V0 = (m1 * v_01 + m2 * v_02) / M

    r0 = r_01 - r_02
    r = np.linalg.norm(r0)

    omega = np.sqrt(G * M / r**3)

    t = np.asarray(t)

    coswt = np.cos(omega * t)
    sinwt = np.sin(omega * t)

    u_r = r0 / r
    u_th = np.array([-u_r[1], u_r[0]])

    r_rel = r * (coswt[:, None] * u_r + sinwt[:, None] * u_th)

    R = R0 + t[:, None] * V0

    r1 = R + (m2 / M) * r_rel
    r2 = R - (m1 / M) * r_rel

    return r1, r2, omega


def vitesse_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G=1.0):
    """
    Calcule la vitesse des deux corps à l'instant t via la solution analytique
    (dérivée temporelle de position_analytique — orbite circulaire).

    param :
        r_01, r_02 : positions initiales des corps 1 et 2 (array-like)
        v_01, v_02 : vitesses initiales des corps 1 et 2 (array-like)
        m1, m2     : masses des deux corps (float)
        t          : temps (array-like)
        G          : constante gravitationnelle (float, défaut=1.0)
    return :
        v1, v2 : vitesses des deux corps à chaque instant t (arrays Nx2)
    """
    r_01 = np.asarray(r_01, float)
    r_02 = np.asarray(r_02, float)
    v_01 = np.asarray(v_01, float)
    v_02 = np.asarray(v_02, float)
    t = np.asarray(t)

    M = m1 + m2
    V0 = (m1 * v_01 + m2 * v_02) / M  # vitesse du centre de masse (constante)

    r0 = r_01 - r_02
    r = np.linalg.norm(r0)

    omega = np.sqrt(G * M / r**3)  # pulsation de l'orbite circulaire

    u_r = r0 / r
    u_th = np.array([-u_r[1], u_r[0]])  # vecteur tangentiel (90° dans le sens trigo)

    # dérivée de r_rel(t) = r*(cos(ωt) u_r + sin(ωt) u_th)
    # v_rel(t) = r*ω*(-sin(ωt) u_r + cos(ωt) u_th)
    sinwt = np.sin(omega * t)
    coswt = np.cos(omega * t)

    v_rel = r * omega * (-sinwt[:, None] * u_r + coswt[:, None] * u_th)

    # vitesses individuelles = vitesse du CDM + contribution du mouvement relatif
    v1 = V0 + (m2 / M) * v_rel
    v2 = V0 - (m1 / M) * v_rel

    return v1, v2


def acceleration(r1, r2, m1, m2, G=1.0):
    """
    Calcule l'accélération gravitationnelle entre deux corps.
    param :
        r1: position du corps 1 (array-like)
        r2: position du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        a1, a2: accélérations des deux corps (arrays numpy)
    """
    r = r2 - r1
    dist3 = np.linalg.norm(r) ** 3
    a1 = G * m2 * r / dist3
    a2 = -G * m1 * r / dist3
    return a1, a2


def euler_explicite(r1_0, r2_0, v1_0, v2_0, m1, m2, t, dt, G=1.0):
    """
    Intègre les équations du mouvement de deux corps en utilisant la méthode d'Euler explicite.
    param :
        r1_0: position initiale du corps 1 (array-like)
        r2_0: position initiale du corps 2 (array-like)
        v1_0: vitesse initiale du corps 1 (array-like)
        v2_0: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        t: temps (array-like)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1, r2: positions des deux corps à chaque instant t (arrays numpy)
        v1, v2: vitesses des deux corps à chaque instant t (arrays numpy)
    """
    N = len(t)
    r1 = np.zeros((N, 2))
    r2 = np.zeros((N, 2))
    v1 = np.zeros((N, 2))
    v2 = np.zeros((N, 2))
    r1[0], r2[0], v1[0], v2[0] = r1_0, r2_0, v1_0, v2_0

    for i in range(1, N):
        a1, a2 = acceleration(r1[i - 1], r2[i - 1], m1, m2, G)
        r1[i] = r1[i - 1] + v1[i - 1] * dt
        r2[i] = r2[i - 1] + v2[i - 1] * dt
        v1[i] = v1[i - 1] + a1 * dt
        v2[i] = v2[i - 1] + a2 * dt
    return r1, r2, v1, v2


def rk4_step(r1, v1, r2, v2, m1, m2, dt, G=1.0, eps=0.0):
    """
    Effectue un pas d'intégration en utilisant la méthode de Runge-Kutta d'ordre 4 (RK4) pour deux corps en interaction gravitationnelle.
    param :
        r1: position actuelle du corps 1 (array-like)
        v1: vitesse actuelle du corps 1 (array-like)
        r2: position actuelle du corps 2 (array-like)
        v2: vitesse actuelle du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1_new, v1_new, r2_new, v2_new: nouvelles positions et vitesses des deux corps après le pas de temps dt (arrays numpy)
    """
    # conversions
    r1 = np.asarray(r1, float)
    v1 = np.asarray(v1, float)
    r2 = np.asarray(r2, float)
    v2 = np.asarray(v2, float)

    def deriv(state):
        r1, v1, r2, v2 = state
        a1, a2 = acceleration(r1, r2, m1, m2, G)
        return (v1, a1, v2, a2)

    s0 = (r1, v1, r2, v2)
    k1 = deriv(s0)
    k2 = deriv(tuple(s + 0.5 * dt * k for s, k in zip(s0, k1)))
    k3 = deriv(tuple(s + 0.5 * dt * k for s, k in zip(s0, k2)))
    k4 = deriv(tuple(s + dt * k for s, k in zip(s0, k3)))

    r1_new = r1 + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    v1_new = v1 + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    r2_new = r2 + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    v2_new = v2 + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0

    # conserve l’ordre (r1, v1, r2, v2)
    return r1_new, v1_new, r2_new, v2_new


def rk4_integrate(r1_0, r2_0, v1_0, v2_0, m1, m2, t, dt, G=1.0):
    """
    Intègre les équations du mouvement de deux corps en utilisant la méthode de Runge-Kutta d'ordre 4 (RK4).
    param :
        r1_0: position initiale du corps 1 (array-like)
        r2_0: position initiale du corps 2 (array-like)
        v1_0: vitesse initiale du corps 1 (array-like)
        v2_0: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        t: temps (array-like)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1, r2: positions des deux corps à chaque instant t (arrays numpy)
        v1, v2: vitesses des deux corps à chaque instant t (arrays numpy)
    """
    N = len(t)
    r1 = np.zeros((N, 2))
    r2 = np.zeros((N, 2))
    v1 = np.zeros((N, 2))
    v2 = np.zeros((N, 2))

    r1[0] = np.asarray(r1_0, float)
    r2[0] = np.asarray(r2_0, float)
    v1[0] = np.asarray(v1_0, float)
    v2[0] = np.asarray(v2_0, float)

    for i in range(1, N):
        r1[i], v1[i], r2[i], v2[i] = rk4_step(r1[i - 1], v1[i - 1], r2[i - 1], v2[i - 1], m1, m2, dt, G)
    return r1, r2, v1, v2


def verlet_step(r1, v1, r2, v2, m1, m2, dt, G=1.0):
    """
    Effectue un pas d'intégration en utilisant la méthode de Verlet pour deux corps en interaction gravitationnelle.
    param :
        r1: position actuelle du corps 1 (array-like)
        v1: vitesse actuelle du corps 1 (array-like)
        r2: position actuelle du corps 2 (array-like)
        v2: vitesse actuelle du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1_new, v1_new, r2_new, v2_new: nouvelles positions et vitesses des deux corps après le pas de temps dt (arrays numpy)
    """
    a1, a2 = acceleration(r1, r2, m1, m2, G)

    r1_new = r1 + v1 * dt + 0.5 * a1 * dt**2
    r2_new = r2 + v2 * dt + 0.5 * a2 * dt**2

    a1_new, a2_new = acceleration(r1_new, r2_new, m1, m2, G)

    v1_new = v1 + 0.5 * (a1 + a1_new) * dt
    v2_new = v2 + 0.5 * (a2 + a2_new) * dt

    return r1_new, v1_new, r2_new, v2_new


def verlet_integrate(r1_0, r2_0, v1_0, v2_0, m1, m2, t, dt, G=1.0):
    """
    Intègre les équations du mouvement de deux corps en utilisant la méthode de Verlet.
    param :
        r1_0: position initiale du corps 1 (array-like)
        r2_0: position initiale du corps 2 (array-like)
        v1_0: vitesse initiale du corps 1 (array-like)
        v2_0: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        t: temps (array-like)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1, r2: positions des deux corps à chaque instant t (arrays numpy)
        v1, v2: vitesses des deux corps à chaque instant t (arrays numpy)
    """
    N = len(t)
    r1 = np.zeros((N, 2))
    r2 = np.zeros((N, 2))
    v1 = np.zeros((N, 2))
    v2 = np.zeros((N, 2))

    r1[0] = np.asarray(r1_0, float)
    r2[0] = np.asarray(r2_0, float)
    v1[0] = np.asarray(v1_0, float)
    v2[0] = np.asarray(v2_0, float)

    for i in range(1, N):
        r1[i], v1[i], r2[i], v2[i] = verlet_step(r1[i - 1], v1[i - 1], r2[i - 1], v2[i - 1], m1, m2, dt, G)
    return r1, r2, v1, v2


def odeint_integrate(r1_0, r2_0, v1_0, v2_0, m1, m2, t, G=1.0):
    """
    Intègre les équations du mouvement de deux corps en utilisant la fonction odeint de SciPy.
    param :
        r1_0: position initiale du corps 1 (array-like)
        r2_0: position initiale du corps 2 (array-like)
        v1_0: vitesse initiale du corps 1 (array-like)
        v2_0: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        t: temps (array-like)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        r1, r2: positions des deux corps à chaque instant t (arrays numpy)
        v1, v2: vitesses des deux corps à chaque instant t (arrays numpy)
    """

    def deriv(state, t):
        r1, v1, r2, v2 = state.reshape(4, 2)
        a1, a2 = acceleration(r1, r2, m1, m2, G)
        return np.concatenate((v1, a1, v2, a2))

    state0 = np.concatenate((r1_0, v1_0, r2_0, v2_0))
    sol = odeint(deriv, state0, t)
    r1 = sol[:, :2]
    v1 = sol[:, 2:4]
    r2 = sol[:, 4:6]
    v2 = sol[:, 6:]
    return r1, r2, v1, v2


def affiche_positions(t, r1, r2, methode, label1="Corps 1", label2="Corps 2"):
    """
    Affiche les trajectoires des deux corps.
    param :
        t: temps (array-like)
        r1: positions du corps 1 (array-like)
        r2: positions du corps 2 (array-like)
        methode : nom de la méthode utilisée (str)
        label1: label pour le corps 1 (str)
        label2: label pour le corps 2 (str)
    return:
        None
    """
    plt.figure()
    plt.plot(r1[:, 0], r1[:, 1], label=label1, color=couleur.get(methode, "tab:blue"))
    plt.plot(r2[:, 0], r2[:, 1], label=label2, color=couleur.get(methode, "tab:blue"))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectoires des deux corps")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, f"trajectoires_{methode}.png"), bbox_inches="tight")
    plt.show()


def plot_erreur(t, dt):
    """
    Trace l'erreur entre les solutions numérique et la solution analytique.
    param :
        t: temps (array-like)
        dt: pas de temps utilisé (float)
        method_name: nom de la méthode numérique utilisée (str)
    return:
        None
    """

    r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
    r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_verlet, r2_verlet, v1_verlet, v2_verlet = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_ode, r2_ode, v1_ode, v2_ode = odeint_integrate(r_01, r_02, v_01, v_02, m1, m2, t, G)

    err_eul = np.linalg.norm(r1_eul - r1_ana, axis=1)
    err_rk4 = np.linalg.norm(r1_rk4 - r1_ana, axis=1)
    err_verlet = np.linalg.norm(r1_verlet - r1_ana, axis=1)
    err_ode = np.linalg.norm(r1_ode - r1_ana, axis=1)

    plt.figure()
    plt.plot(t, err_eul, label="Erreur Corps 1 (Euler)", color=couleur.get("Euler", "tab:blue"))
    plt.plot(t, err_rk4, label="Erreur Corps 2 (RK4)", color=couleur.get("RK4", "tab:orange"))
    plt.plot(t, err_verlet, label="Erreur Corps 3 (Verlet)", color=couleur.get("Verlet", "tab:green"))
    plt.plot(t, err_ode, label="Erreur Corps 4 (Odeint)", color=couleur.get("Odeint", "tab:red"))

    plt.xlabel("Temps")
    plt.ylabel("Erreur (distance)")
    plt.title(f"Erreur entre les solutions numérique et analytique pour dt = {dt} et T = {t[-1]}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, f"erreur_vs_analytique_dt={dt}_T={t[-1]}.png"), bbox_inches="tight")
    plt.show()


def tracer_erreur_vs_dt(dt_min=1e-4, dt_max=1.0, nb_points=6, T=100.0):
    """
    Trace l'erreur maximale en fonction du pas de temps dt pour Euler, RK4, Verlet.
    param :
        dt_min: pas de temps minimum (float)
        dt_max: pas de temps maximum (float)
        nb_points: nombre de points à tracer (int)
        T: durée totale de la simulation (float)
    return:
        dt_list: liste des pas de temps testés (array numpy)
        erreur_euler: erreurs maximales pour Euler (array numpy)
        erreur_rk4: erreurs maximales pour RK4 (array numpy)
        erreur_verlet: erreurs maximales pour Verlet (array numpy)

    """

    dt_list = np.logspace(np.log10(dt_min), np.log10(dt_max), nb_points)

    erreur_euler = []
    erreur_rk4 = []
    erreur_verlet = []
    erreur_odeint = []

    for d in dt_list:
        t = np.arange(0.0, T, d)

        # Solution analytique aux mêmes instants
        r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)

        # Euler
        r1_eul, r2_eul, *_ = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, d, G)
        err_eul = np.max(np.linalg.norm(r1_eul - r1_ana, axis=1))
        erreur_euler.append(err_eul)

        # RK4
        r1_rk4, r2_rk4, *_ = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, d, G)
        err_rk4 = np.max(np.linalg.norm(r1_rk4 - r1_ana, axis=1))
        erreur_rk4.append(err_rk4)

        # Verlet
        r1_ver, r2_ver, *_ = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, d, G)
        err_ver = np.max(np.linalg.norm(r1_ver - r1_ana, axis=1))
        erreur_verlet.append(err_ver)

        # Odeint
        r1_ode, r2_ode, *_ = odeint_integrate(r_01, r_02, v_01, v_02, m1, m2, t, G)
        err_ode = np.max(np.linalg.norm(r1_ode - r1_ana, axis=1))
        erreur_odeint.append(err_ode)

    # Tracé
    plt.figure(figsize=(8, 6))
    plt.loglog(dt_list, erreur_euler, marker="o", label="Euler")
    plt.loglog(dt_list, erreur_rk4, marker="o", label="RK4")
    plt.loglog(dt_list, erreur_verlet, marker="o", label="Verlet")
    plt.loglog(dt_list, erreur_odeint, marker="o", label="Odeint")
    plt.xlabel("Pas de temps dt")
    plt.ylabel("Erreur maximale")
    plt.title("Erreur maximale en fonction du pas de temps")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, f"erreur_vs_dt_T={T}.png"), bbox_inches="tight")
    plt.show()

    return dt_list, np.array(erreur_euler), np.array(erreur_rk4), np.array(erreur_verlet), np.array(erreur_odeint)


def energie_mecanique(r1, r2, v1, v2, m1, m2, G=1.0):
    """
    Calcule l'énergie mécanique totale du système à deux corps
    à chaque instant.

    param :
        r1, r2 : positions des deux corps (Nx2)
        v1, v2 : vitesses des deux corps (Nx2)
        m1, m2 : masses
        G : constante gravitationnelle

    return :
        E : énergie mécanique totale (array de taille N)
    """
    # énergie cinétique
    Ec1 = 0.5 * m1 * np.sum(v1**2, axis=1)
    Ec2 = 0.5 * m2 * np.sum(v2**2, axis=1)

    # distance entre les deux corps
    r12 = np.linalg.norm(r2 - r1, axis=1)

    # énergie potentielle gravitationnelle
    Ep = -G * m1 * m2 / r12

    # énergie totale
    E = Ec1 + Ec2 + Ep
    return E


def tracer_energie_double(T, dt, m1, m2, G=1.0):
    """
    Trace la dérive de l'énergie mécanique pour les trois méthodes d'intégration.
    param :
        t: temps (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        G: constante gravitationnelle (float, défaut=1.0)
    """

    def dE_rel(r1, r2, v1, v2):
        E = energie_mecanique(r1, r2, v1, v2, m1, m2, G)
        E0 = E[0]
        return (E - E0) / abs(E0)

    t = np.arange(0.0, T, dt)
    r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
    r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_ver, r2_ver, v1_ver, v2_ver = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_ode, r2_ode, v1_ode, v2_ode = odeint_integrate(r_01, r_02, v_01, v_02, m1, m2, t, G)

    y_eul = dE_rel(r1_eul, r2_eul, v1_eul, v2_eul)
    y_rk4 = dE_rel(r1_rk4, r2_rk4, v1_rk4, v2_rk4)
    y_ver = dE_rel(r1_ver, r2_ver, v1_ver, v2_ver)
    y_ode = dE_rel(r1_ode, r2_ode, v1_ode, v2_ode)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, y_eul, label="Euler", color=couleur.get("Euler", "tab:blue"))
    ax1.plot(t, y_rk4, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax1.plot(t, y_ver, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax1.plot(t, y_ode, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax1.set_title("Dérive de l'énergie (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(E - E_0)/|E_0|$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, y_rk4, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax2.plot(t, y_ver, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax2.plot(t, y_ode, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax2.set_title("Zoom : RK4 vs Verlet vs Odeint")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.savefig(os.path.join(FIG_DIR, f"conservation_energie_T={T}_dt={dt}.png"), bbox_inches="tight")
    plt.show()


def drift_energie_vs_dt(dt_list, T, r_01, r_02, v_01, v_02, m1, m2, G=1.0, methode_nam=""):
    """
    Trace la dérive de l'énergie mécanique en fonction du pas de temps dt.
    param :
        dt_list: liste des pas de temps à tester (array-like)
        T: durée totale de la simulation (float)
        r_01: position initiale du corps 1 (array-like)
        r_02: position initiale du corps 2 (array-like)
        v_01: vitesse initiale du corps 1 (array-like)
        v_02: vitesse initiale du corps 2 (array-like)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        G: constante gravitationnelle (float, défaut=1.0)
    return:
        drift_rk4, drift_ver: dérives maximales de l'énergie pour RK4 et Verlet (arrays numpy)
    """
    drift_rk4 = []
    drift_ver = []
    drift_ode = []

    for dt in dt_list:
        t = np.arange(0, T, dt)

        r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
        r1_ver, r2_ver, v1_ver, v2_ver = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
        r1_ode, r2_ode, v1_ode, v2_ode = odeint_integrate(r_01, r_02, v_01, v_02, m1, m2, t, G)

        E_rk4 = energie_mecanique(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2, G)
        E_ver = energie_mecanique(r1_ver, r2_ver, v1_ver, v2_ver, m1, m2, G)
        E_ode = energie_mecanique(r1_ode, r2_ode, v1_ode, v2_ode, m1, m2, G)

        y_rk4 = (E_rk4 - E_rk4[0]) / abs(E_rk4[0])
        y_ver = (E_ver - E_ver[0]) / abs(E_ver[0])
        y_ode = (E_ode - E_ode[0]) / abs(E_ode[0])

        drift_rk4.append(np.max(np.abs(y_rk4)))
        drift_ver.append(np.max(np.abs(y_ver)))
        drift_ode.append(np.max(np.abs(y_ode)))

    plt.figure(figsize=(8, 6))
    plt.loglog(dt_list, drift_rk4, marker="o", label="RK4", color=couleur.get("RK4", "tab:orange"))
    plt.loglog(dt_list, drift_ver, marker="o", label="Verlet", color=couleur.get("Verlet", "tab:green"))
    plt.loglog(dt_list, drift_ode, marker="o", label="Odeint", color=couleur.get("Odeint", "tab:red"))
    plt.xlabel("dt")
    plt.ylabel(r"max |(E - E0)/|E0||")
    plt.title("Dérive max d'énergie en fonction de dt")
    plt.savefig(os.path.join(FIG_DIR, "drift_energie_vs_dt.png"), bbox_inches="tight")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    return np.array(drift_rk4), np.array(drift_ver), np.array(drift_ode)


def moment_cinetique(r1, r2, v1, v2, m1, m2):
    """
    Calcule le moment cinetique total du systeme a deux corps
    en 2D, a chaque instant.

    param :
        r1, r2 : positions des deux corps (Nx2)
        v1, v2 : vitesses des deux corps (Nx2)
        m1, m2 : masses

    return :
        Lz : moment cinetique total selon l'axe z (array de taille N)
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)

    L1 = m1 * (r1[:, 0] * v1[:, 1] - r1[:, 1] * v1[:, 0])
    L2 = m2 * (r2[:, 0] * v2[:, 1] - r2[:, 1] * v2[:, 0])
    return L1 + L2


def tracer_moment_cinetique_double(t, L_eul, L_rk4, L_ver, L_ana,L_ode):
    """
    Trace le moment cinetique pour les différentes méthodes d'intégration.
    param :
        t: temps (array-like)
        L_eul: moment cinetique total pour Euler (array-like)
        L_rk4: moment cinetique total pour RK4 (array-like)
        L_ver: moment cinetique total pour Verlet (array-like)
        L_ana: moment cinetique total pour la solution analytique (array-like)
    return:
        None
    """
    plt.figure(figsize=(8, 6))
    denom = abs(L_ana[0]) if abs(L_ana[0]) > 0 else 1.0
    L_eul_norm = np.abs(L_eul - L_ana) / denom
    L_rk4_norm = np.abs(L_rk4 - L_ana) / denom
    L_ver_norm = np.abs(L_ver - L_ana) / denom
    L_ode_norm = np.abs(L_ode - L_ana) / denom

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, L_eul_norm, label="Euler", color=couleur.get("Euler", "tab:blue"))
    ax1.plot(t, L_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax1.plot(t, L_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax1.plot(t, L_ode_norm, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax1.set_title("Dérive du moment cinétique (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(L - L_0)/|L_0|$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, L_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax2.plot(t, L_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax2.plot(t, L_ode_norm, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax2.set_title("Zoom : RK4 vs Verlet")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    Tfinal = t[-1] if hasattr(t, "__len__") and len(t) > 0 else 0
    plt.savefig(os.path.join(FIG_DIR, f"conservation_moment_T={Tfinal}.png"), bbox_inches="tight")
    plt.show()


def runge_lenz_vecteur(r1, r2, v1, v2, m1, m2, G=1.0):
    """
    Calcule le vecteur de Runge-Lenz pour le problème à deux corps en 2D.
    param :
        r1, r2 : positions (Nx2)
        v1, v2 : vitesses  (Nx2)
        m1, m2 : masses (float)
        G      : constante gravitationnelle (float, défaut=1.0)
    return :
        A : vecteur de Runge-Lenz à chaque instant (Nx2)
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)

    mu = G * (m1 + m2)
    r12 = r2 - r1  # (N, 2)
    v12 = v2 - v1  # (N, 2)

    # Moment cinétique spécifique de la trajectoire relative (scalaire Lz)
    Lz = r12[:, 0] * v12[:, 1] - r12[:, 1] * v12[:, 0]  # (N,)

    # v12 × (Lz ẑ) projeté en 2D : (vy·Lz, -vx·Lz)
    vxL = np.stack([v12[:, 1] * Lz, -v12[:, 0] * Lz], axis=1)  # (N, 2)

    # Norme de r12 à chaque instant, keepdims pour la division
    r12_norm = np.linalg.norm(r12, axis=1, keepdims=True)  # (N, 1)

    A = vxL / mu - r12 / r12_norm  # (N, 2)
    return A


def tracer_runge_lenz(t, A_eul, A_rk4, A_ver, A_ana, A_ode):
    """
    Trace le vecteur de Runge-Lenz pour les différentes méthodes d'intégration.
    param :
        t: temps (array-like)
        A_eul: vecteur de Runge-Lenz pour Euler (array-like)
        A_rk4: vecteur de Runge-Lenz pour RK4 (array-like)
        A_ver: vecteur de Runge-Lenz pour Verlet (array-like)
        A_ana: vecteur de Runge-Lenz pour la solution analytique (array-like)
        A_ode: vecteur de Runge-Lenz pour Odeint (array-like)
    return:
        None
    """
    plt.figure(figsize=(8, 6))
    A_eul_norm = np.linalg.norm(A_eul - A_ana, axis=1) / (np.linalg.norm(A_ana, axis=1) + 1e-12)
    A_rk4_norm = np.linalg.norm(A_rk4 - A_ana, axis=1) / (np.linalg.norm(A_ana, axis=1) + 1e-12)
    A_ver_norm = np.linalg.norm(A_ver - A_ana, axis=1) / (np.linalg.norm(A_ana, axis=1) + 1e-12)
    A_ode_norm = np.linalg.norm(A_ode - A_ana, axis=1) / (np.linalg.norm(A_ana, axis=1) + 1e-12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, A_eul_norm, label="Euler", color=couleur.get("Euler", "tab:blue"))
    ax1.plot(t, A_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax1.plot(t, A_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax1.plot(t, A_ode_norm, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax1.set_title("Dérive du vecteur de Runge-Lenz (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"||A - A_0||/||A_0||")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, A_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax2.plot(t, A_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax2.plot(t, A_ode_norm, label="Odeint", color=couleur.get("Odeint", "tab:red"))
    ax2.set_title("Zoom : RK4 vs Verlet")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    Tfinal = t[-1] if hasattr(t, "__len__") and len(t) > 0 else 0
    plt.savefig(os.path.join(FIG_DIR, f"conservation_runge_lenz_T={Tfinal}.png"), bbox_inches="tight")
    plt.show()



# ============================================================
# Testes 2-corps
# ============================================================

m1 = 1
m2 = 2
G = 1.0
r_01 = (1.0, 0.0)
r_02 = (-1.0, 0.0)
v_01, v_02 = vitesses_circulaires(r_01, r_02, m1, m2, G, sens=+1)

dt = 0.1
t = np.arange(0, 4000, dt)

# =============================================================
# calcule des positions et vitesses avec les différentes méthodes
# =============================================================

r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
v1_ana, v2_ana = vitesse_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
r1_verlet, r2_verlet, v1_verlet, v2_verlet = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
r1_ode, r2_ode, v1_ode, v2_ode = odeint_integrate(r_01, r_02, v_01, v_02, m1, m2, t, G)

# =============================================================
# affichage des trajectoires
# =============================================================

# affiche_positions(t, r1_ana, r2_ana,"analitique", label1=f"Corps 1 (Analytique)(m = {m1})", label2=f"Corps 2 (Analytique)(m = {m2})")
# affiche_positions(t, r1_eul, r2_eul,"euler", label1=f'Corps 1 (Euler)(m = {m1})',label2=f"Corps 2 (Euler)(m = {m2})")
# affiche_positions(t, r1_rk4, r2_rk4, "rk4", label1=f"Corps 1 (RK4)(m = {m1})", label2=f"Corps 2 (RK4)(m = {m2})")
# affiche_positions(t, r1_verlet, r2_verlet,"verlet", label1="Corps 1 (Verlet)(m = {m1})",label2="Corps 2 (Verlet)(m = {m2})")
# affiche_positions(t, r1_ode, r2_ode,"odeint", label1=f'Corps 1 (ODEINT)(m = {m1})',label2=f"Corps 2 (ODEINT)(m = {m2})")


# =============================================================
# affichage de l'erreur
# =============================================================

# plot_erreur(t, dt)
# tracer_erreur_vs_dt(dt_min=1e-5, dt_max=1, nb_points=11, T=50)


# =============================================================
# etude de l'energie mécanique
# =============================================================

tracer_energie_double(1000,dt, m1, m2, G)

# dt_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
# drift_energie_vs_dt(dt_list, T=300, r_01=r_01, r_02=r_02, v_01=v_01, v_02=v_02, m1=m1, m2=m2, G=G)


# =============================================================
# etude du moment cinetique
# =============================================================

# L_eul = moment_cinetique(r1_eul, r2_eul, v1_eul, v2_eul, m1, m2)
# L_rk4 = moment_cinetique(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2)
# L_ver = moment_cinetique(r1_verlet, r2_verlet, v1_verlet, v2_verlet, m1, m2)
# L_ana = moment_cinetique(r1_ana, r2_ana, v1_ana, v2_ana, m1, m2)
# L_ode = moment_cinetique(r1_ode, r2_ode, v1_ode, v2_ode, m1, m2)
# tracer_moment_cinetique_double(t, L_eul, L_rk4, L_ver, L_ana, L_ode)

# =============================================================
# etude du vecteur de Runge-Lenz
# =============================================================

# A_eul = runge_lenz_vecteur(r1_eul, r2_eul, v1_eul, v2_eul, m1, m2, G)
# A_rk4 = runge_lenz_vecteur(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2, G)
# A_ver = runge_lenz_vecteur(r1_verlet, r2_verlet, v1_verlet, v2_verlet, m1, m2, G)
# A_ana = runge_lenz_vecteur(r1_ana, r2_ana, v1_ana, v2_ana, m1, m2, G)
# A_ode = runge_lenz_vecteur(r1_ode, r2_ode, v1_ode, v2_ode, m1, m2, G)
# tracer_runge_lenz(t, A_eul, A_rk4, A_ver, A_ana, A_ode)
