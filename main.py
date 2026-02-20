
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Variables globales
G = 1.0  # constante gravitationnelle (unités arbitraires)
parameters = {"axes.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 20}
plt.rcParams.update(parameters)

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


def affiche_positions(t, r1, r2, label1="Corps 1", label2="Corps 2"):
    """
    Affiche les trajectoires des deux corps.
    param :
        t: temps (array-like)
        r1: positions du corps 1 (array-like)
        r2: positions du corps 2 (array-like)
        label1: label pour le corps 1 (str)
        label2: label pour le corps 2 (str)
    return:
        None
    """
    plt.figure()
    plt.plot(r1[:, 0], r1[:, 1], label=label1)
    plt.plot(r2[:, 0], r2[:, 1], label=label2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectoires des deux corps")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_erreur(t, r_1_ana, r2_ana, r1_num, r2_num, dt, method_name=""):
    """
    Trace l'erreur entre la solution numérique et la solution analytique.
    param :
        t: temps (array-like)
        r_1_ana: positions analytiques du corps 1 (array-like)
        r2_ana: positions analytiques du corps 2 (array-like)
        r1_num: positions numériques du corps 1 (array-like)
        r2_num: positions numériques du corps 2 (array-like)
        dt: pas de temps utilisé (float)
        method_name: nom de la méthode numérique utilisée (str)
    return:
        None
    """
    err1 = np.linalg.norm(r1_num - r_1_ana, axis=-1)
    err2 = np.linalg.norm(r2_num - r2_ana, axis=-1)

    plt.figure()
    plt.plot(t, err1, label=f"Erreur Corps 1 ({method_name})")
    plt.plot(t, err2, label=f"Erreur Corps 2 ({method_name})")

    plt.xlabel("Temps")
    plt.ylabel("Erreur (distance)")
    plt.title(f"Erreur entre solutions numérique et analytique avec dt = {dt}")
    plt.legend()
    plt.grid(True)
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
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "erreur_vs_dt.png"))
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


def tracer_energie_double(t, r1_eul, r2_eul, v1_eul, v2_eul, r1_rk4, r2_rk4, v1_rk4, v2_rk4, r1_ver, r2_ver, v1_ver, v2_ver, m1, m2, G=1.0):
    """
    Trace la dérive de l'énergie mécanique pour les trois méthodes d'intégration.
    param :
        t: temps (array-like)
        r1_eul, r2_eul, v1_eul, v2_eul: positions et vitesses pour Euler (arrays numpy)
        r1_rk4, r2_rk4, v1_rk4, v2_rk4: positions et vitesses pour RK4 (arrays numpy)
        r1_ver, r2_ver, v1_ver, v2_ver: positions et vitesses pour Verlet (arrays numpy)
        m1: masse du corps 1 (float)
        m2: masse du corps 2 (float)
        G: constante gravitationnelle (float, défaut=1.0)
    """
    def dE_rel(r1, r2, v1, v2):
        E = energie_mecanique(r1, r2, v1, v2, m1, m2, G)
        E0 = E[0]
        return (E - E0) / abs(E0)

    y_eul = dE_rel(r1_eul, r2_eul, v1_eul, v2_eul)
    y_rk4 = dE_rel(r1_rk4, r2_rk4, v1_rk4, v2_rk4)
    y_ver = dE_rel(r1_ver, r2_ver, v1_ver, v2_ver)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t, y_eul, label="Euler")
    ax1.plot(t, y_rk4, label="RK4")
    ax1.plot(t, y_ver, label="Verlet")
    ax1.set_title("Dérive de l'énergie (toutes méthodes)")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(E - E_0)/|E_0|$")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, y_rk4, label="RK4")
    ax2.plot(t, y_ver, label="Verlet")
    ax2.set_title("Zoom : RK4 vs Verlet")
    ax2.set_xlabel("Temps")
    ax2.grid(True)
    ax2.legend()

    for ax in (ax1, ax2):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "conservation_energie.png"))
    plt.show()


def drift_energie_vs_dt(dt_list, T, r_01, r_02, v_01, v_02, m1, m2, G=1.0):
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
    plt.loglog(dt_list, drift_rk4, marker="o", label="RK4")
    plt.loglog(dt_list, drift_ver, marker="o", label="Verlet")
    plt.xlabel("dt")
    plt.ylabel(r"max |(E - E0)/|E0||")
    plt.title("Dérive max d'énergie en fonction de dt")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    return np.array(drift_rk4), np.array(drift_ver)


# --- Test ---
# Variable

m1 = 1
m2 = 2
G = 1.0
r_01 = (1.0, 0.0)
r_02 = (-1.0, 0.0)
v_01, v_02 = vitesses_circulaires(r_01, r_02, m1, m2, G, sens=+1)

dt = 0.1
t = np.arange(0, 10000, dt)

#================================================================================
# calcule des positions et vitesses avec les différentes méthodes
# r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
# r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
# r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
# r1_verlet, r2_verlet, v1_verlet, v2_verlet = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)

#================================================================================
# affichage des trajectoires
# affiche_positions(t, r1_ana, r2_ana, label1=f"Corps 1 (Analytique)(m = {m1})", label2=f"Corps 2 (Analytique)(m = {m2})")
# affiche_positions(t, r1_eul, r2_eul,label1=f'Corps 1 (Euler)(m = {m1})',label2=f"Corps 2 (Euler)(m = {m2})")
# affiche_positions(t, r1_rk4, r2_rk4, label1=f"Corps 1 (RK4)(m = {m1})", label2=f"Corps 2 (RK4)(m = {m2})")
# affiche_positions(t, r1_verlet, r2_verlet,label1="Corps 1 (Verlet)(m = {m1})",label2="Corps 2 (Verlet)(m = {m2})")

#================================================================================
# etude de l'energie mécanique
# tracer_energie_double(
#     t,
#     r1_eul, r2_eul, v1_eul, v2_eul,
#     r1_rk4, r2_rk4, v1_rk4, v2_rk4,
#     r1_verlet, r2_verlet, v1_verlet, v2_verlet,
#     m1, m2, G
# )

# dt_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
# drift_energie_vs_dt(dt_list, T=300, r_01=r_01, r_02=r_02, v_01=v_01, v_02=v_02, m1=m1, m2=m2, G=G)


#================================================================================
# affichage de l'erreur

# plot_erreur(t, r1_ana, r2_ana, r1_eul, r2_eul, dt, method_name="Euler")
# plot_erreur(t, r1_ana, r2_ana, r1_rk4, r2_rk4, dt, method_name="RK4")
# plot_erreur(t, r1_ana, r2_ana, r1_verlet, r2_verlet, dt, method_name="Verlet")

# tracer_erreur_vs_dt(dt_min=1e-5, dt_max=1.0, nb_points=10, T=10.0)

# ============================================================
# Partie 2 : N-corps
# ============================================================

# ============================================================
# 1) Accélération N-corps (2D)
# ============================================================

def accelerations_nbody(R, m, G=1.0, eps=1e-12):
    """
    Calcule les accélérations gravitationnelles pour N corps en 2D.

    R : (N,2) positions
    m : (N,) masses
    return A : (N,2) accélérations
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
            dist2 = rij[0]*rij[0] + rij[1]*rij[1] + eps
            dist3 = dist2 * np.sqrt(dist2)
            ai += G * m[j] * rij / dist3
        A[i] = ai

    return A


# ============================================================
# 2) Intégrateurs N-corps : RK4 et Velocity-Verlet
# ============================================================

def rk4_step_nbody(R, V, m, dt, G=1.0, eps=1e-12):
    """
    Un pas RK4 pour N corps.
    Etat : (R, V) avec R,V en (N,2)
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    m = np.asarray(m, float)

    def deriv(Rx, Vx):
        Ax = accelerations_nbody(Rx, m, G=G, eps=eps)
        return Vx, Ax

    k1_R, k1_V = deriv(R, V)
    k2_R, k2_V = deriv(R + 0.5*dt*k1_R, V + 0.5*dt*k1_V)
    k3_R, k3_V = deriv(R + 0.5*dt*k2_R, V + 0.5*dt*k2_V)
    k4_R, k4_V = deriv(R + dt*k3_R,     V + dt*k3_V)

    Rn = R + (dt/6.0)*(k1_R + 2*k2_R + 2*k3_R + k4_R)
    Vn = V + (dt/6.0)*(k1_V + 2*k2_V + 2*k3_V + k4_V)
    return Rn, Vn


def rk4_integrate_nbody(R0, V0, m, t, dt, G=1.0, eps=1e-12):
    """
    Intégration RK4 sur la grille de temps t.
    return Rs : (T,N,2), Vs : (T,N,2)
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
        Rs[k], Vs[k] = rk4_step_nbody(Rs[k-1], Vs[k-1], m, dt, G=G, eps=eps)

    return Rs, Vs


def verlet_step_nbody(R, V, m, dt, G=1.0, eps=1e-12):
    """
    Un pas Velocity-Verlet pour N corps.
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    m = np.asarray(m, float)

    A = accelerations_nbody(R, m, G=G, eps=eps)
    Rn = R + V*dt + 0.5*A*(dt**2)
    An = accelerations_nbody(Rn, m, G=G, eps=eps)
    Vn = V + 0.5*(A + An)*dt
    return Rn, Vn


def verlet_integrate_nbody(R0, V0, m, t, dt, G=1.0, eps=1e-12):
    """
    Intégration Velocity-Verlet sur la grille de temps t.
    return Rs : (T,N,2), Vs : (T,N,2)
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
        Rs[k], Vs[k] = verlet_step_nbody(Rs[k-1], Vs[k-1], m, dt, G=G, eps=eps)

    return Rs, Vs


# ============================================================
# 3) Invariants / diagnostics N-corps
# ============================================================

def energie_nbody(Rs, Vs, m, G=1.0, eps=1e-12):
    """
    Energie totale du système (array de taille T).
    Rs: (T,N,2), Vs: (T,N,2)
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
        for j in range(i+1, N):
            rij = np.linalg.norm(Rs[:, j, :] - Rs[:, i, :], axis=1) + eps
            Ep += -G * m[i] * m[j] / rij

    return Ec + Ep


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

    Lz = np.sum(m[None, :] * (x*vy - y*vx), axis=1)
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
# 4) Scénario 3 corps simple : "restricted 3-body" (planète test)
# ============================================================

def scenario_restricted_3body(m1=1.0, m2=2.0, m3=1e-3, G=1.0):
    """
    Deux masses principales en orbite circulaire (comme ton 2-corps),
    + un 3e corps très léger (planète test).
    Retourne (masses, R0, V0)
    """
    m = np.array([m1, m2, m3], float)

    # on reprend l'idée de ton 2-corps : positions opposées
    r1_0 = np.array([ 1.0, 0.0])
    r2_0 = np.array([-1.0, 0.0])

    # vitesses circulaires pour les deux gros corps (copie du principe 2-corps)
    M = m1 + m2
    r_rel = r2_0 - r1_0
    r = np.linalg.norm(r_rel)
    u_r = r_rel / r
    u_th = np.array([-u_r[1], u_r[0]])  # sens trigo
    v_rel_norm = np.sqrt(G * M / r)
    v_rel = v_rel_norm * u_th
    v1_0 = -(m2 / M) * v_rel
    v2_0 =  (m1 / M) * v_rel

    # 3e corps : un peu plus loin, vitesse initiale approximative
    r3_0 = np.array([0.0, 2.5])
    # vitesse "à peu près orbitale" autour du barycentre (approx)
    v3_0 = np.array([ -0.7, 0.0])

    R0 = np.stack([r1_0, r2_0, r3_0], axis=0)
    V0 = np.stack([v1_0, v2_0, v3_0], axis=0)

    return m, R0, V0


# ============================================================
# 5) Expérience chaos : sensibilité aux conditions initiales
# ============================================================

def distance_trajectoires(RsA, RsB):
    """
    Distance entre deux simulations (T,N,2) -> array T
    (on prend la norme globale sur tous les corps)
    """
    D = RsA - RsB
    return np.sqrt(np.sum(D**2, axis=(1, 2)))


# ============================================================
# 6) Plots utiles
# ============================================================

def plot_trajectoires_3corps(Rs, title="Trajectoires 3 corps"):
    """
    Trace les trajectoires (x,y) des 3 corps.
    Rs : (T,3,2)
    """
    plt.figure(figsize=(6, 6))
    for i in range(Rs.shape[1]):
        plt.plot(Rs[:, i, 0], Rs[:, i, 1], label=f"Corps {i+1}")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_invariants(t, E, L, title="Invariants"):
    """
    Trace (E-E0)/|E0| et (L-L0)/|L0|
    """
    E0 = E[0]
    L0 = L[0]
    dE = (E - E0) / abs(E0) if E0 != 0 else (E - E0)
    dL = (L - L0) / abs(L0) if L0 != 0 else (L - L0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(t, dE)
    ax1.set_title("Dérive énergie")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel(r"$(E-E_0)/|E_0|$")
    ax1.grid(True)
    ax1.ticklabel_format(style="plain", axis="y", useOffset=False)

    ax2.plot(t, dL)
    ax2.set_title("Dérive moment cinétique")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel(r"$(L-L_0)/|L_0|$")
    ax2.grid(True)
    ax2.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) DEMO 3 CORPS (à mettre en bas, à la place de tes tests)
# ============================================================

if __name__ == "__main__":
    G = 1.0
    m, R0, V0 = scenario_restricted_3body(m1=1.0, m2=2.0, m3=1e-3, G=G)

    dt = 0.01
    T = 300.0
    t = np.arange(0.0, T, dt)

    # --- Simulation RK4 ---
    Rs_rk4, Vs_rk4 = rk4_integrate_nbody(R0, V0, m, t, dt, G=G)
    E_rk4 = energie_nbody(Rs_rk4, Vs_rk4, m, G=G)
    L_rk4 = moment_cinetique_nbody(Rs_rk4, Vs_rk4, m)

    # --- Simulation Verlet ---
    Rs_ver, Vs_ver = verlet_integrate_nbody(R0, V0, m, t, dt, G=G)
    E_ver = energie_nbody(Rs_ver, Vs_ver, m, G=G)
    L_ver = moment_cinetique_nbody(Rs_ver, Vs_ver, m)

    # Trajectoires
    plot_trajectoires_3corps(Rs_rk4, "3 corps (RK4)")
    plot_trajectoires_3corps(Rs_ver, "3 corps (Verlet)")

    # Invariants
    plot_invariants(t, E_rk4, L_rk4, "Invariants 3 corps (RK4)")
    plot_invariants(t, E_ver, L_ver, "Invariants 3 corps (Verlet)")

    # --- Sensibilité aux CI (chaos) : petite perturbation sur le 3e corps ---
    R0b = R0.copy()
    R0b[2, 0] += 1e-6  # perturbation minuscule

    Rs_ver_b, Vs_ver_b = verlet_integrate_nbody(R0b, V0, m, t, dt, G=G)
    d = distance_trajectoires(Rs_ver, Rs_ver_b)

    plt.figure()
    plt.plot(t, d)
    plt.xlabel("Temps")
    plt.ylabel("Distance entre deux trajectoires")
    plt.title("Sensibilité aux conditions initiales (Verlet)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()