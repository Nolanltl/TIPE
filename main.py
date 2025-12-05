import numpy as np
import math as mt
import matplotlib.pyplot as plt
#test ajout github

# Variables globales
G = 1.0  # constante gravitationnelle (unités arbitraires)
parameters = {"axes.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 20}
plt.rcParams.update(parameters)


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

def plot_erreur(t,r_1_ana,r2_ana,r1_num,r2_num,dt,method_name=""):
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

# def plot_erreur(
#     t,
#     r1_eul,
#     r1_rk4,
#     r1_verlet,
#     r1_ana,
# ):

#     err1_eul = np.linalg.norm(r1_eul - r1_ana, axis=-1)
#     err1_rk4 = np.linalg.norm(r1_rk4 - r1_ana, axis=-1)
#     err1_verlet = np.linalg.norm(r1_verlet - r1_ana, axis=-1)

#     plt.figure()
#     plt.plot(t, err1_eul, label="Erreur Corps 1 (Euler)")
#     plt.plot(t, err1_rk4, label="Erreur Corps 1 (RK4)")
#     plt.plot(t, err1_verlet, label="Erreur Corps 1 (Verlet)")

#     plt.xlabel("Temps")
#     plt.ylabel("Erreur (distance)")
#     plt.title("Erreur entre solutions numérique et analytique")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def erreur_dt(dt1, dtf):
    dt = np.linspace(dt1, dtf, num=5)
    for d in dt:
        pass
    pass


# --- Test ---
# Variable

m1 = 1
m2 = 2
G = 1.0
r_01 = (1.0, 0.0)
r_02 = (-1.0, 0.0)
v_01, v_02 = vitesses_circulaires(r_01, r_02, m1, m2, G, sens=+1)

dt = 0.01
t = np.arange(0, 100, dt)

r1_ana, r2_ana, omega = position_analytique(r_01, r_02, v_01, v_02, m1, m2, t, G)
# r1_eul, r2_eul, v1_eul, v2_eul = euler_explicite(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
r1_rk4, r2_rk4, v1_rk4, v2_rk4 = rk4_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)
# r1_verlet, r2_verlet, v1_verlet, v2_verlet = verlet_integrate(r_01, r_02, v_01, v_02, m1, m2, t, dt, G)


# affiche_positions(t, r1_ana, r2_ana, label1=f"Corps 1 (Analytique)(m = {m1})", label2=f"Corps 2 (Analytique)(m = {m2})")
# affiche_positions(t, r1_eul, r2_eul,label1=f'Corps 1 (Euler)(m = {m1})',label2=f"Corps 2 (Euler)(m = {m2})")
# affiche_positions(t, r1_rk4, r2_rk4,label1=f"Corps 1 (RK4)(m = {m1})",label2=f"Corps 2 (RK4)(m = {m2})")
# affiche_positions(t, r1_verlet, r2_verlet,label1="Corps 1 (Verlet)(m = {m1})",label2="Corps 2 (Verlet)(m = {m2})")



# plot_erreur(t, r1_ana, r2_ana, r1_eul, r2_eul, dt, method_name="Euler")
plot_erreur(t, r1_ana, r2_ana, r1_rk4, r2_rk4, dt, method_name="RK4")
# plot_erreur(t, r1_ana, r2_ana, r1_verlet, r2_verlet, dt, method_name="Verlet")