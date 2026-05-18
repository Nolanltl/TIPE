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
FIG_DIR = os.path.join(BASE_DIR, "figures")
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

couleur = {"Verlet ": "green", "RK4": "orange", "Euler": "blue", "Analytique": "gray"}

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
    t    = np.asarray(t)

    M  = m1 + m2
    V0 = (m1 * v_01 + m2 * v_02) / M   # vitesse du centre de masse (constante)

    r0 = r_01 - r_02
    r  = np.linalg.norm(r0)

    omega = np.sqrt(G * M / r**3)       # pulsation de l'orbite circulaire

    u_r  = r0 / r
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
    err_eul = np.linalg.norm(r1_eul - r1_ana, axis=1)
    err_rk4 = np.linalg.norm(r1_rk4 - r1_ana, axis=1)
    err_verlet = np.linalg.norm(r1_verlet - r1_ana, axis=1)

    plt.figure()
    plt.plot(t, err_eul, label=f"Erreur Corps 1 (Euler)", color=couleur.get("Euler", "tab:blue"))
    plt.plot(t, err_rk4, label=f"Erreur Corps 2 (RK4)", color=couleur.get("RK4", "tab:orange"))
    plt.plot(t, err_verlet, label=f"Erreur Corps 3 (Verlet)", color=couleur.get("Verlet", "tab:green"))

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

    # Tracé
    plt.figure(figsize=(8, 6))
    plt.loglog(dt_list, erreur_euler, marker="o", label="Euler")
    plt.loglog(dt_list, erreur_rk4, marker="o", label="RK4")
    plt.loglog(dt_list, erreur_verlet, marker="o", label="Verlet")
    plt.xlabel("Pas de temps dt")
    plt.ylabel("Erreur maximale")
    plt.title("Erreur maximale en fonction du pas de temps")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, f"erreur_vs_dt_{dt}_T={T}.png"), bbox_inches="tight")
    plt.show()

    return dt_list, np.array(erreur_euler), np.array(erreur_rk4), np.array(erreur_verlet)


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

    r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
    r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
    r1_ver, r2_ver, v1_ver, v2_ver = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)

    y_eul = dE_rel(r1_eul, r2_eul, v1_eul, v2_eul)
    y_rk4 = dE_rel(r1_rk4, r2_rk4, v1_rk4, v2_rk4)
    y_ver = dE_rel(r1_ver, r2_ver, v1_ver, v2_ver)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, y_eul, label="Euler", color=couleur.get("Euler", "tab:blue"))
    ax1.plot(t, y_rk4, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax1.plot(t, y_ver, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax1.set_title("Dérive de l'énergie (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(E - E_0)/|E_0|$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, y_rk4, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax2.plot(t, y_ver, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax2.set_title("Zoom : RK4 vs Verlet")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.savefig(os.path.join(FIG_DIR, f"conservation_energie_T={T[-1]}_dt={dt}.png"), bbox_inches="tight")
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

    for dt in dt_list:
        t = np.arange(0, T, dt)

        r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
        r1_ver, r2_ver, v1_ver, v2_ver = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)

        E_rk4 = energie_mecanique(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2, G)
        E_ver = energie_mecanique(r1_ver, r2_ver, v1_ver, v2_ver, m1, m2, G)

        y_rk4 = (E_rk4 - E_rk4[0]) / abs(E_rk4[0])
        y_ver = (E_ver - E_ver[0]) / abs(E_ver[0])

        drift_rk4.append(np.max(np.abs(y_rk4)))
        drift_ver.append(np.max(np.abs(y_ver)))

    plt.figure(figsize=(8, 6))
    plt.loglog(dt_list, drift_rk4, marker="o", label="RK4", color=couleur.get("RK4", "tab:orange"))
    plt.loglog(dt_list, drift_ver, marker="o", label="Verlet", color=couleur.get("Verlet", "tab:green"))
    plt.xlabel("dt")
    plt.ylabel(r"max |(E - E0)/|E0||")
    plt.title("Dérive max d'énergie en fonction de dt")
    plt.savefig(os.path.join(FIG_DIR, "drift_energie_vs_dt.png"), bbox_inches="tight")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    return np.array(drift_rk4), np.array(drift_ver)


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


def tracer_moment_cinetique_double(t, L_eul, L_rk4, L_ver, L_ana):
    """
    Trace le moment cinetique total en fonction du temps pour RK4 et Verlet.
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, L_eul_norm, label="Euler", color=couleur.get("Euler", "tab:blue"))
    ax1.plot(t, L_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax1.plot(t, L_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax1.set_title("Dérive du moment cinétique (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(L - L_0)/|L_0|$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, L_rk4_norm, label="RK4", color=couleur.get("RK4", "tab:orange"))
    ax2.plot(t, L_ver_norm, label="Verlet", color=couleur.get("Verlet", "tab:green"))
    ax2.set_title("Zoom : RK4 vs Verlet")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    Tfinal = t[-1] if hasattr(t, "__len__") and len(t) > 0 else 0
    plt.savefig(os.path.join(FIG_DIR, f"conservation_moment_T={Tfinal}.png"), bbox_inches="tight")
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

# =============================================================
# affichage des trajectoires
# =============================================================

# affiche_positions(t, r1_ana, r2_ana, label1=f"Corps 1 (Analytique)(m = {m1})", label2=f"Corps 2 (Analytique)(m = {m2})")
# affiche_positions(t, r1_eul, r2_eul,label1=f'Corps 1 (Euler)(m = {m1})',label2=f"Corps 2 (Euler)(m = {m2})")
# affiche_positions(t, r1_rk4, r2_rk4, label1=f"Corps 1 (RK4)(m = {m1})", label2=f"Corps 2 (RK4)(m = {m2})")
# affiche_positions(t, r1_verlet, r2_verlet,label1="Corps 1 (Verlet)(m = {m1})",label2="Corps 2 (Verlet)(m = {m2})")


# =============================================================
# etude de l'energie mécanique
# =============================================================

# tracer_energie_double(t,dt, m1, m2, G)

# dt_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
# drift_energie_vs_dt(dt_list, T=300, r_01=r_01, r_02=r_02, v_01=v_01, v_02=v_02, m1=m1, m2=m2, G=G)


# =============================================================
# affichage de l'erreur
# =============================================================


# plot_erreur(t, dt)
# tracer_erreur_vs_dt(dt_min=1e-1, dt_max=100.0, nb_points=40, T=500.0)

# =============================================================
# etude du moment cinetique 
# =============================================================
L_eul = moment_cinetique(r1_eul, r2_eul, v1_eul, v2_eul, m1, m2)
L_rk4 = moment_cinetique(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2)
L_ver = moment_cinetique(r1_verlet, r2_verlet, v1_verlet, v2_verlet, m1, m2)
L_ana = moment_cinetique(r1_ana, r2_ana, v1_ana, v2_ana, m1, m2)
tracer_moment_cinetique_double(t, L_eul, L_rk4, L_ver, L_ana)

# ============================================================
# Partie 2 : N-corps
# ============================================================

# ============================================================
# 1) Accélération N-corps (2D)
# ============================================================


def accelerations_nbody(R, m, G=1.0, eps=1e-12):
    """
    Calcule les accélérations gravitationnelles pour N corps en 2D.
    param :
        R : positions des N corps (array de taille (N,2))
        m : masses des N corps (array de taille (N,))
        G : constante gravitationnelle (float, défaut=1.0)
        eps : régularisation pour éviter les singularités (float, défaut=1e-12)
    return :
        A : accélérations des N corps (array de taille (N,2))
    """
    R = np.asarray(R, float)
    m = np.asarray(m, float)
    N = R.shape[0]

    A = np.zeros_like(R)

    for i in range(N):
        ai = np.zeros(2)
        for j in range(N):
            if i == j:
                continue
            rij = R[j] - R[i]
            dist2 = rij[0] * rij[0] + rij[1] * rij[1] + eps
            dist3 = dist2 * np.sqrt(dist2)
            ai += G * m[j] * rij / dist3
        A[i] = ai

    return A


# ============================================================
#  Intégrateurs N-corps : RK4 et Velocity-Verlet
# ============================================================


def rk4_step_nbody(R, V, m, dt, G=1.0, eps=1e-12):
    """
    Effectue un pas d'intégration en utilisant la méthode de Runge-Kutta d'ordre 4 (RK4) pour N corps en interaction gravitationnelle.
    param :
        R: positions actuelles des N corps (array de taille (N,2))
        V: vitesses actuelles des N corps (array de taille (N,2))
        m: masses des N corps (array de taille (N,))
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
        eps: régularisation pour éviter les singularités (float, défaut=1e-12)

    return:
        Rn, Vn: nouvelles positions et vitesses des N corps après le pas de temps dt (arrays de taille (N,2))

    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    m = np.asarray(m, float)

    def deriv(Rx, Vx):
        Ax = accelerations_nbody(Rx, m, G=G, eps=eps)
        return Vx, Ax

    k1_R, k1_V = deriv(R, V)
    k2_R, k2_V = deriv(R + 0.5 * dt * k1_R, V + 0.5 * dt * k1_V)
    k3_R, k3_V = deriv(R + 0.5 * dt * k2_R, V + 0.5 * dt * k2_V)
    k4_R, k4_V = deriv(R + dt * k3_R, V + dt * k3_V)

    Rn = R + (dt / 6.0) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
    Vn = V + (dt / 6.0) * (k1_V + 2 * k2_V + 2 * k3_V + k4_V)
    return Rn, Vn


def rk4_integrate_nbody(R0, V0, m, t, dt, G=1.0, eps=1e-12):
    """
    Intègre les équations du mouvement de N corps en utilisant la méthode de Runge-Kutta d'ordre 4 (RK4).
    param :
        R0: positions initiales des N corps (array de taille (N,2))
        V0: vitesses initiales des N corps (array de taille (N,2))
        m: masses des N corps (array de taille (N,))
        t: temps (array-like)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
        eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    return:
        Rs: positions des N corps à chaque instant t (array de taille (T,N,2))
        Vs: vitesses des N corps à chaque instant t (array de taille (T
    """
    t = np.asarray(t, float)
    Tn = len(t)
    R0 = np.asarray(R0, float)
    V0 = np.asarray(V0, float)
    m = np.asarray(m, float)

    N = R0.shape[0]
    Rs = np.zeros((Tn, N, 2))
    Vs = np.zeros((Tn, N, 2))
    Rs[0] = R0
    Vs[0] = V0

    for k in range(1, Tn):
        Rs[k], Vs[k] = rk4_step_nbody(Rs[k - 1], Vs[k - 1], m, dt, G=G, eps=eps)

    return Rs, Vs


def verlet_step_nbody(R, V, m, dt, G=1.0, eps=1e-12):
    """
    Effectue un pas d'intégration en utilisant la méthode de Verlet pour N corps en interaction gravitationnelle.
    param :
        R: positions actuelles des N corps (array de taille (N,2))
        V: vitesses actuelles des N corps (array de taille (N,2))
        m: masses des N corps (array de taille (N,))
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
        eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    return:
        Rn, Vn: nouvelles positions et vitesses des N corps après le pas de temps dt (arrays de taille (N,2))
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    m = np.asarray(m, float)

    A = accelerations_nbody(R, m, G=G, eps=eps)
    Rn = R + V * dt + 0.5 * A * (dt**2)
    An = accelerations_nbody(Rn, m, G=G, eps=eps)
    Vn = V + 0.5 * (A + An) * dt
    return Rn, Vn


def verlet_integrate_nbody(R0, V0, m, t, dt, G=1.0, eps=1e-12):
    """
    Intègre les équations du mouvement de N corps en utilisant la méthode de Verlet.
    param :
        R0: positions initiales des N corps (array de taille (N,2))
        V0: vitesses initiales des N corps (array de taille (N,2))
        m: masses des N corps (array de taille (N,))
        t: temps (array-like)
        dt: pas de temps (float)
        G: constante gravitationnelle (float, défaut=1.0)
        eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    return:
        Rs: positions des N corps à chaque instant t (array de taille (T,N,2))
        Vs: vitesses des N corps à chaque instant t (array de taille (T,N,2))
    """
    t = np.asarray(t, float)
    Tn = len(t)
    R0 = np.asarray(R0, float)
    V0 = np.asarray(V0, float)
    m = np.asarray(m, float)

    N = R0.shape[0]
    Rs = np.zeros((Tn, N, 2))
    Vs = np.zeros((Tn, N, 2))
    Rs[0] = R0
    Vs[0] = V0

    for k in range(1, Tn):
        Rs[k], Vs[k] = verlet_step_nbody(Rs[k - 1], Vs[k - 1], m, dt, G=G, eps=eps)

    return Rs, Vs


# ============================================================
# Invariants
# ============================================================


def energie_nbody(Rs, Vs, m, G=1.0, eps=1e-12):
    """
    Calcule l'énergie mécanique totale du système à N corps à chaque instant.
    param :
        Rs: positions des N corps à chaque instant t (array de taille (T,N,2))
        Vs: vitesses des N corps à chaque instant t (array de taille (T ,N,2))
        m: masses des N corps (array de taille (N,))
        G: constante gravitationnelle (float, défaut=1.0)
        eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    return:
        E: énergie mécanique totale à chaque instant t (array de taille (T,))
    """
    Rs = np.asarray(Rs, float)
    Vs = np.asarray(Vs, float)
    m = np.asarray(m, float)

    Tn, N, _ = Rs.shape

    # cinétique
    Ec = 0.5 * np.sum(m[None, :] * np.sum(Vs**2, axis=2), axis=1)

    # potentielle
    Ep = np.zeros(Tn)
    for i in range(N):
        for j in range(i + 1, N):
            rij = np.linalg.norm(Rs[:, j, :] - Rs[:, i, :], axis=1) + eps
            Ep += -G * m[i] * m[j] / rij

    return Ec + Ep


# ============================================================
# à faire seulment si fait en 2 corps : vérifier la conservation du moment cinétique (doit rester constant pour une orbite circulaire)
# ============================================================


def moment_cinetique_nbody(Rs, Vs, m):
    """
    Moment cinétique total (scalaire Lz) en 2D : sum m (x vy - y vx)
    return array de taille T
    """
    Rs = np.asarray(Rs, float)
    Vs = np.asarray(Vs, float)
    m = np.asarray(m, float)

    x = Rs[:, :, 0]
    y = Rs[:, :, 1]
    vx = Vs[:, :, 0]
    vy = Vs[:, :, 1]

    Lz = np.sum(m[None, :] * (x * vy - y * vx), axis=1)
    return Lz


def barycentre_nbody(Rs, Vs, m):
    """
    Renvoie Rcm(t) et Vcm(t).
    """
    Rs = np.asarray(Rs, float)
    Vs = np.asarray(Vs, float)
    m = np.asarray(m, float)
    M = np.sum(m)

    Rcm = np.sum(Rs * m[None, :, None], axis=1) / M
    Vcm = np.sum(Vs * m[None, :, None], axis=1) / M
    return Rcm, Vcm


# ============================================================
# Expérience chaos : sensibilité aux conditions initiales
# ============================================================


def distance_trajectoires(RsA, RsB):
    """
    Distance entre deux simulations (T,N,2) -> array T
    (on prend la norme globale sur tous les corps)
    """
    D = RsA - RsB
    return np.sqrt(np.sum(D**2, axis=(1, 2)))


# ============================================================
# Plots
# ============================================================


def sensibilite_CI(R0, V0, m, epsilons, T=20.0, dt=0.005, G=1.0, eps=1e-12, corps_perturbe=0, composante=0):
    """
    Trace la divergence entre une trajectoire de référence et des trajectoires
    légèrement perturbées sur les conditions initiales (sensibilité au chaos).

    param :
        R0             : positions initiales de référence (array (N,2))
        V0             : vitesses initiales de référence (array (N,2))
        m              : masses des N corps (array (N,))
        epsilons       : liste des perturbations à tester (list of float)
                        ex : [1e-3, 1e-5, 1e-7, 1e-9]
        T              : durée totale de la simulation (float, défaut=20.0)
        dt             : pas de temps (float, défaut=0.005)
        G              : constante gravitationnelle (float, défaut=1.0)
        eps            : régularisation singularités (float, défaut=1e-12)
        corps_perturbe : indice du corps dont on perturbe la position (int, défaut=0)
        composante     : composante perturbée : 0=x, 1=y (int, défaut=0)

    return :
        t         : tableau de temps (array de taille (Tn,))
        distances : dict {epsilon: array de distances} pour chaque epsilon
    """
    t = np.arange(0.0, T, dt)

    # ── Simulation de référence ──────────────────────────────────────────────
    Rs_ref, _ = verlet_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)

    distances = {}

    # ── Simulations perturbées ───────────────────────────────────────────────
    for epsilon in epsilons:
        R0_pert = R0.copy()
        R0_pert[corps_perturbe, composante] += epsilon

        # recentre le barycentre après perturbation
        R0_pert, V0_pert = normalize_com(R0_pert, V0.copy(), m)

        Rs_pert, _ = verlet_integrate_nbody(R0_pert, V0_pert, m, t, dt, G=G, eps=eps)
        distances[epsilon] = distance_trajectoires(Rs_ref, Rs_pert)

    # ── Palette de couleurs ──────────────────────────────────────────────────
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(epsilons) - 1)) for i in range(len(epsilons))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for (epsilon, dist), col in zip(distances.items(), colors):
        exp = int(np.round(np.log10(epsilon)))
        label = rf"$\varepsilon = 10^{{{exp}}}$"
        ax1.plot(t, dist, label=label, color=col)
        ax2.semilogy(t, dist + 1e-16, label=label, color=col)

    # ── Panneau gauche : échelle linéaire ────────────────────────────────────
    ax1.set_title("Divergence des trajectoires")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$\|\Delta \mathbf{r}(t)\|$")
    ax1.grid(True)
    ax1.legend()
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # ── Panneau droit : échelle logarithmique ────────────────────────────────
    ax2.set_title("Divergence des trajectoires (log)")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel(r"$\log\|\Delta \mathbf{r}(t)\|$")
    ax2.grid(True, which="both")
    ax2.legend()

    comp_str = "x" if composante == 0 else "y"
    plt.suptitle(f"Sensibilité aux conditions initiales — corps {corps_perturbe + 1} " f"(perturbation sur {comp_str})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"sensibilite_CI_corps{corps_perturbe + 1}_{comp_str}_T={T}_dt={dt}.png"), bbox_inches="tight")
    plt.show()

    return t, distances


def plot_trajectoires_3corps(Rs, Vs=None, title="Trajectoires 3 corps"):
    """
    Affiche les trajectoires des 3 corps à partir de Rs (T,N,2).
    Affiche un point sur la position finale de chaque corps,
    et le vecteur vitesse final si Vs est fourni.

    param :
        Rs    : positions des N corps à chaque instant t (array de taille (T,N,2))
        Vs    : vitesses des N corps à chaque instant t (array de taille (T,N,2))
                si None, les vecteurs vitesse ne sont pas affichés (défaut=None)
        title : titre du graphique (str)

    return :
        None
    """
    # palette de couleurs — une par corps
    colors = ["tab:blue", "tab:orange", "tab:green"]

    N = Rs.shape[1]

    plt.figure(figsize=(6, 6))

    for i in range(N):
        col = colors[i % len(colors)]

        # ── Trajectoire ──────────────────────────────────────────────────────
        plt.plot(Rs[:, i, 0], Rs[:, i, 1], label=f"Corps {i + 1}", color=col)

        # ── Point final ──────────────────────────────────────────────────────
        xf, yf = Rs[-1, i, 0], Rs[-1, i, 1]
        plt.plot(xf, yf, marker="o", markersize=10, color=col, markeredgecolor="black", markeredgewidth=1.2, zorder=5)

        # ── Vecteur vitesse final ─────────────────────────────────────────────
        if Vs is not None:
            vx, vy = Vs[-1, i, 0], Vs[-1, i, 1]
            plt.annotate(
                "",
                xy=(xf + vx * 0.3, yf + vy * 0.3),  # pointe de la flèche
                xytext=(xf, yf),  # base = position finale
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=2.0,
                    mutation_scale=15,
                ),
                zorder=6,
            )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, f"{title.replace(' ', '_')}.png"), bbox_inches="tight")
    plt.show()


def plot_invariants(t, E, L, title="Invariants"):

    E0 = E[0]
    L0 = L[0]

    # --- Energie ---
    dE = (E - E0) / abs(E0) if E0 != 0 else (E - E0)

    # --- Moment cinétique ---
    if abs(L0) < 1e-8:
        dL = L - L0  # dérive absolue
        ylabelL = r"$(L - L_0)$"
    else:
        dL = (L - L0) / abs(L0)
        ylabelL = r"$(L - L_0)/|L_0|$"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # --- Energie ---
    ax1.plot(t, dE)
    ax1.set_title("Dérive énergie")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(E - E_0)/|E_0|$")
    ax1.grid(True)
    ax1.ticklabel_format(style="plain", axis="y", useOffset=False)

    # --- Moment cinétique ---
    ax2.plot(t, dL)
    ax2.set_title("Dérive moment cinétique")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel(ylabelL)
    ax2.grid(True)
    ax2.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.suptitle(title)
    plt.savefig(os.path.join(FIG_DIR, f"{title.replace(' ', '_')}.png"), bbox_inches="tight")
    plt.show()


def normalize_com(R0, V0, m):
    """Recentre pour avoir Rcm=0 et Vcm=0 au départ (utile pour la figure-eight)
    param :
    R0: positions initiales des N corps (array de taille (N,2))
    V0: vitesses initiales des N corps (array de taille (N,2))
    m: masses des N corps (array de taille (N,))
    return:
    R0n, V0n: positions et vitesses normalisées (array de taille (N,2))
    """
    V0 = np.asarray(V0, float)
    m = np.asarray(m, float)
    M = np.sum(m)

    Rcm = np.sum(R0 * m[:, None], axis=0) / M
    Vcm = np.sum(V0 * m[:, None], axis=0) / M

    return R0 - Rcm, V0 - Vcm


def scenario_figure_eight(G=1.0):
    """
    Orbite périodique en '8' (choreography) pour 3 masses égales.
    param :
    G: constante gravitationnelle (float, défaut=1.0)
    return: (masses, R0, V0)
    """
    m = np.array([1.0, 1.0, 1.0], float)

    R0 = np.array([[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0.0, 0.0]], float)

    V0 = np.array([[0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146]], float)
    R0, V0 = normalize_com(R0, V0, m)
    return m, R0, V0


def scenario_lagrange(d=2.0, G=1.0):
    """
    Orbite périodique en triangle équilatéral de Lagrange (1772).
    3 masses égales aux sommets d'un triangle équilatéral en rotation uniforme
    autour du barycentre commun.

    param :
        d : côté du triangle équilatéral (float, défaut=2.0)
        G : constante gravitationnelle (float, défaut=1.0)

    return : (masses, R0, V0)
        masses : array (3,)
        R0     : positions initiales (array (3,2))
        V0     : vitesses initiales (array (3,2))

    """
    m = np.array([1.0, 1.0, 1.0], float)

    r = d / np.sqrt(3.0)
    omega = np.sqrt(G * np.sum(m) / d**3) * d / r

    omega = np.sqrt(3.0 * G * 1.0 / d**3)
    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])

    R0 = np.array([[r * np.cos(a), r * np.sin(a)] for a in angles], float)

    V0 = np.array([[-r * omega * np.sin(a), r * omega * np.cos(a)] for a in angles], float)

    R0, V0 = normalize_com(R0, V0, m)

    return m, R0, V0


def scenario_euler_collinear(d=2.0, G=1.0):
    """
    Orbite périodique colinéaire d'Euler (1767).
    3 masses égales alignées sur l'axe x, en rotation autour du barycentre.
    Configuration symétrique : m1 et m3 aux extrémités, m2 au centre.

    param :
        d : distance entre corps adjacents (float, défaut=2.0)
            => m1 à -d, m2 à 0, m3 à +d
        G : constante gravitationnelle (float, défaut=1.0)

    return : (masses, R0, V0)
        masses : array (3,)
        R0     : positions initiales (array (3,2))
        V0     : vitesses initiales (array (3,2))

    Note : cette solution est instable — une perturbation même infinitésimale
            brise l'alignement. Elle illustre parfaitement la sensibilité aux CI.
    """
    m = np.array([1.0, 1.0, 1.0], float)

    R0 = np.array([[-d, 0.0], [0.0, 0.0], [d, 0.0]], float)

    omega = np.sqrt(5.0 * G * 1.0 / (4.0 * d**3))

    V0 = np.array([[0.0, -omega * d], [0.0, 0.0], [0.0, omega * d]], float)

    R0, V0 = normalize_com(R0, V0, m)

    return m, R0, V0


def demo_figure_eight(dt=0.005, T=50.0, G=1.0, eps=1e-12, method="verlet"):
    """
    test de la figure-eight : intégration + invariants + plot
    param :
    dt: pas de temps (float)
    T: durée totale de la simulation (float)
    G: constante gravitationnelle (float, défaut=1.0)
    eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    method: méthode d'intégration ("rk4" ou "verlet")
    return:
    t, m, R0, V0, Rs, Vs, E, L
    - t: temps (array de taille (Tn,))
    - m: masses des N corps (array de taille (N,))
    - R0: positions initiales des N corps (array de taille (N,2))
    - V0: vitesses initiales des N corps (array de taille (N,2))
    - Rs: positions des N corps à chaque instant t (array de taille (Tn,N,2))
    - Vs: vitesses des N corps à chaque instant t (array de taille (Tn,N,2))
    - E: énergie mécanique totale à chaque instant t (array de taille (Tn,))
    - L: moment cinétique total à chaque instant t (array de taille (Tn,))
    """
    m, R0, V0 = scenario_figure_eight(G=G)
    t = np.arange(0.0, T, dt)

    if method == "rk4":
        Rs, Vs = rk4_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)
        name = "RK4"
    else:
        Rs, Vs = verlet_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)
        name = "Verlet"

    # invariants
    E = energie_nbody(Rs, Vs, m, G=G, eps=eps)
    L = moment_cinetique_nbody(Rs, Vs, m)

    plot_trajectoires_3corps(Rs, title=f"Figure en huit (3 corps){name}")
    # plot_invariants(t, E, L, title=f"Invariants — Figure-eight — {name}")

    return t, m, R0, V0, Rs, Vs, E, L


def divergence_to_reference(Rs_ref, Rs_test):
    """
    Mesure de divergence : distance max à la trajectoire de référence sur [0,T]
    param :
    Rs_ref: positions de référence (array de taille (Tn,N,2))
    Rs_test: positions à tester (array de taille (Tn,N,2))
    return:
    divergence: distance maximale entre Rs_test et Rs_ref sur tous les instants t (float)
    - calculée comme max_t ||Rs_test(t) - Rs_ref(t)||, où ||.|| est la norme globale sur tous les corps (sqrt(sum_i ||r_i_test - r_i_ref||^2)))

    """
    return np.max(np.linalg.norm(Rs_test - Rs_ref, axis=(1, 2)))


def ftle_from_divergence(D, D0, T):
    """
    FTLE (finite-time Lyapunov exponent) :
    lambda_T = (1/T) * ln(D/D0)
    """
    return (1.0 / T) * np.log(D / D0)


def divergence_map_figure_eight(
    dx_min=-0.02,
    dx_max=0.02,
    dy_min=-0.02,
    dy_max=0.02,
    n=41,
    dt=0.005,
    T=6.3259,  # IMPORTANT : période ~ 6.3259
    G=1.0,
    eps=1e-12,
    body_index=0,  # corps perturbé
    method="verlet",
):
    """
    Carte de divergence autour de la figure-eight :
    Z(dx,dy) = log10( max_t ||R_test(t) - R_ref(t)|| )
    param :
    dx_min, dx_max: intervalle de perturbation en x (float)
    dy_min, dy_max: intervalle de perturbation en y (float)
    n: nombre de points dans chaque direction (int)
    dt: pas de temps pour l'intégration (float)
    T: durée totale de la simulation (float)
    G: constante gravitationnelle (float, défaut=1.0)
    eps: régularisation pour éviter les singularités (float, défaut=1e-12)
    body_index: index du corps perturbé (int, 0, 1 ou 2)
    method: méthode d'intégration ("rk4" ou "verlet")
    return:
    dxs, dys, Z
     - dxs: valeurs de perturbation en x (array de taille (n,))
     - dys: valeurs de perturbation en y (array de taille (n,))
     - Z: carte de divergence (array de taille (n,n)), où Z[j,i] correspond à la perturbation (dxs[i], dys[j])
    """

    m, R0_ref, V0_ref = scenario_figure_eight(G=G)
    t = np.arange(0.0, T, dt)

    # --- trajectoire de référence (dx=dy=0) ---
    if method == "rk4":
        Rs_ref, Vs_ref = rk4_integrate_nbody(R0_ref, V0_ref, m, t, dt, G=G, eps=eps)
    else:
        Rs_ref, Vs_ref = verlet_integrate_nbody(R0_ref, V0_ref, m, t, dt, G=G, eps=eps)

    dxs = np.linspace(dx_min, dx_max, n)
    dys = np.linspace(dy_min, dy_max, n)

    Z = np.zeros((n, n), float)

    for i, dx in enumerate(dxs):
        for j, dy in enumerate(dys):
            R0 = R0_ref.copy()
            V0 = V0_ref.copy()

            # perturbation sur un corps
            R0[body_index, 0] += dx
            R0[body_index, 1] += dy

            # recentre barycentre (translation + vitesse globale)
            R0, V0 = normalize_com(R0, V0, m)

            # intégration
            if method == "rk4":
                Rs, Vs = rk4_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)
            else:
                Rs, Vs = verlet_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)

            # divergence à la référence
            err = divergence_to_reference(Rs_ref, Rs)

            # log pour une carte lisible
            Z[j, i] = np.log10(max(err, 1e-12))

    # --- plot ---
    plt.figure(figsize=(7, 6))
    plt.imshow(Z, origin="lower", extent=[dx_min, dx_max, dy_min, dy_max], aspect="auto", vmin=-6, vmax=-1)
    plt.colorbar(label=r"$\log_{10}(D)$ (divergence à la trajectoire)")
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$\Delta y$")
    plt.title(f"Carte de divergence autour de la figure-eight ({method}, T={T}, dt={dt})")
    plt.savefig(os.path.join(FIG_DIR, f"divergence_map_figure_eight_{method}_dt_{dt}_T_{T}_dx_min_{dx_min}.png"), bbox_inches="tight")
    plt.show()

    return dxs, dys, Z


def lyapunov_map_figure_eight(
    dx_min=-0.02,
    dx_max=0.02,
    dy_min=-0.02,
    dy_max=0.02,
    n=51,
    dt=0.005,
    T=6.3259 * 2,
    G=1.0,
    eps=1e-12,
    body_index=0,
    method="verlet",
    use_max_over_time=True,  # True = D = max_t ||R-Rref|| ; False = D(T) seulement
    clip_lambda=None,  # ex: (0, 2) pour limiter l'échelle des couleurs
):
    """
    Carte FTLE (Lyapunov local sur temps fini) autour de la figure-eight.
    On calcule lambda_T(dx,dy) = (1/T) ln( D / D0 )
    avec D0 = sqrt(dx^2 + dy^2), D = séparation (finale ou max sur le temps).

    - use_max_over_time=True : plus robuste (comme ta carte divergence)
    - use_max_over_time=False: plus "Lyapunov pur" (à temps T)
    """

    m, R0_ref, V0_ref = scenario_figure_eight(G=G)
    t = np.arange(0.0, T, dt)

    # Trajectoire de référence
    if method == "rk4":
        Rs_ref, Vs_ref = rk4_integrate_nbody(R0_ref, V0_ref, m, t, dt, G=G, eps=eps)
    else:
        Rs_ref, Vs_ref = verlet_integrate_nbody(R0_ref, V0_ref, m, t, dt, G=G, eps=eps)

    dxs = np.linspace(dx_min, dx_max, n)
    dys = np.linspace(dy_min, dy_max, n)
    LAM = np.full((n, n), np.nan, float)

    for i, dx in enumerate(dxs):
        for j, dy in enumerate(dys):
            D0 = np.hypot(dx, dy)
            if D0 == 0.0:
                LAM[j, i] = 0.0
                continue

            R0 = R0_ref.copy()
            V0 = V0_ref.copy()

            # Perturbation
            R0[body_index, 0] += dx
            R0[body_index, 1] += dy

            # Recentre COM pour comparer proprement
            R0, V0 = normalize_com(R0, V0, m)

            # Intègre
            if method == "rk4":
                Rs, Vs = rk4_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)
            else:
                Rs, Vs = verlet_integrate_nbody(R0, V0, m, t, dt, G=G, eps=eps)

            # Séparation
            if use_max_over_time:
                D = np.max(np.linalg.norm(Rs - Rs_ref, axis=(1, 2)))
            else:
                D = np.linalg.norm(Rs[-1] - Rs_ref[-1])

            # Evite log(0) si super proche
            D = max(D, 1e-16)

            lam = ftle_from_divergence(D, D0, T)
            LAM[j, i] = lam

    # Option : clip de l'échelle (juste pour l'affichage)
    Lplot = LAM.copy()
    if clip_lambda is not None:
        lo, hi = clip_lambda
        Lplot = np.clip(Lplot, lo, hi)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.imshow(Lplot, origin="lower", extent=[dx_min, dx_max, dy_min, dy_max], aspect="auto")
    plt.colorbar(label=r"$\lambda_T$ (FTLE)  [$1/\mathrm{time}$]")
    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$\Delta y$")
    mode = "max_t" if use_max_over_time else "final"
    plt.title(f"Carte Lyapunov locale (FTLE, {method}, mode={mode}, T={T}, dt={dt})")
    plt.savefig(os.path.join(FIG_DIR, f"lyapunov_map_figure_eight_{method}_mode_{mode}_dt_{dt}_T_{T}_dx_min_{dx_min}.png"), bbox_inches="tight")
    plt.show()

    return dxs, dys, LAM


# divergence_map_figure_eight(
#     dx_min=-0.02, dx_max=0.02,
#     dy_min=-0.02, dy_max=0.02,
#     n=81,
#     T=6.3259*2,
#     dt=0.005,
#     method="verlet"
# )
# demo_figure_eight(dt=0.005, T=100.0, method="verlet")

# m, R0_col, V0_col = scenario_euler_collinear()
# m, R0_huit, V0_huit = scenario_figure_eight()
# m, R0_lag, V0_lag = scenario_lagrange()

# plot_trajectoires_3corps(verlet_integrate_nbody(R0_col, V0_col, m, np.arange(0.0, 20.0, 0.005), dt=0.005)[0], title="Orbite colinéaire d'Euler")
# plot_trajectoires_3corps(verlet_integrate_nbody(R0_huit, V0_huit, m, np.arange(0.0, 20.0, 0.005), dt=0.005)[0], title="Orbite en huit")
# plot_trajectoires_3corps(verlet_integrate_nbody(R0_lag, V0_lag, m, np.arange(0.0, 20.0, 0.005), dt=0.005)[0], title="Orbite de Lagrange")

# lyapunov_map_figure_eight(
#     dx_min=-0.02, dx_max=0.02,
#     dy_min=-0.02, dy_max=0.02,
#     n=61,
#     dt=0.005,
#     T=6.3259*2,
#     method="verlet",
#     use_max_over_time=True,
#     clip_lambda=(0.0, 2.0)
# )


# m, R0, V0 = scenario_figure_eight()

# t, distances = sensibilite_CI(
#     R0, V0, m,
#     epsilons       = [1e-3, 1e-5, 1e-7, 1e-9],
#     T              = 25.0,
#     dt             = 0.005,
#     corps_perturbe = 0,
#     composante     = 0   # perturbation sur x du corps 1
# )
