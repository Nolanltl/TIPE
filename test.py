import main

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
# L_eul = moment_cinetique(r1_eul, r2_eul, v1_eul, v2_eul, m1, m2)
# L_rk4 = moment_cinetique(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2)
# L_ver = moment_cinetique(r1_verlet, r2_verlet, v1_verlet, v2_verlet, m1, m2)
# L_ana = moment_cinetique(r1_ana, r2_ana, v1_ana, v2_ana, m1, m2)
# tracer_moment_cinetique_double(t, L_eul, L_rk4, L_ver, L_ana)

# =============================================================
# etude du vecteur de Runge-Lenz
# =============================================================

A_eul = runge_lenz_vecteur(r1_eul, r2_eul, v1_eul, v2_eul, m1, m2, G)
A_rk4 = runge_lenz_vecteur(r1_rk4, r2_rk4, v1_rk4, v2_rk4, m1, m2, G)
A_ver = runge_lenz_vecteur(r1_verlet, r2_verlet, v1_verlet, v2_verlet, m1, m2, G)
A_ana = runge_lenz_vecteur(r1_ana, r2_ana, v1_ana, v2_ana, m1, m2, G)
tracer_runge_lenz(t, A_eul, A_rk4, A_ver, A_ana)


