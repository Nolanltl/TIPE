from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

G = 1.0  # constante gravitationnelle normalisée
M1 = 1.0  # masse planète 1
M2 = 0.8  # masse planète 2
SEPARATION = 2.0  # distance initiale entre les deux planètes
ALLOWED_METHODS = ("euler", "verlet", "rk4")


@dataclass
class CircularOrbitState:
    m1: float
    m2: float
    separation: float
    phase: float
    omega: float
    r1: float
    r2: float
    pos1: np.ndarray
    pos2: np.ndarray
    vel1: np.ndarray
    vel2: np.ndarray
    G_value: float = G


def build_circular_orbit_state(
    m1: float = M1,
    m2: float = M2,
    separation: float = SEPARATION,
    phase: float = 0.0,
    G_value: float = G,
) -> CircularOrbitState:
    if separation <= 0:
        raise ValueError("La distance de séparation doit être strictement positive.")
    total_mass = m1 + m2
    if total_mass <= 0:
        raise ValueError("La somme des masses doit être strictement positive.")
    r1 = m2 / total_mass * separation
    r2 = m1 / total_mass * separation
    omega = np.sqrt(G_value * total_mass / separation**3)
    cos_phi = np.cos(phase)
    sin_phi = np.sin(phase)
    pos2 = np.array([r2 * cos_phi, r2 * sin_phi], dtype=float)
    pos1 = -np.array([r1 * cos_phi, r1 * sin_phi], dtype=float)
    vel2 = omega * r2 * np.array([-sin_phi, cos_phi], dtype=float)
    vel1 = omega * r1 * np.array([sin_phi, -cos_phi], dtype=float)
    return CircularOrbitState(
        m1=m1,
        m2=m2,
        separation=separation,
        phase=phase,
        omega=omega,
        r1=r1,
        r2=r2,
        pos1=pos1,
        pos2=pos2,
        vel1=vel1,
        vel2=vel2,
        G_value=G_value,
    )


DEFAULT_ORBIT = build_circular_orbit_state()


@dataclass
class Trajectory:
    times: np.ndarray
    pos1: np.ndarray
    pos2: np.ndarray


def _allocate_trajectory(total_time: float, dt: float) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Préparer le stockage des trajectoires et la grille temporelle.

    Returns
    -------
    steps:
        Nombre d'itérations discrètes.
    times:
        Instants échantillonnés (seconds).
    pos1, pos2:
        Tableaux préalloués pour les positions (shape = [steps + 1, 2]).
    """
    if dt <= 0:
        raise ValueError("Le pas de temps dt doit être strictement positif.")
    if total_time <= 0:
        raise ValueError("La durée totale doit être strictement positive.")

    steps = max(1, int(round(total_time / dt)))
    actual_total_time = steps * dt
    times = np.linspace(0.0, actual_total_time, steps + 1)
    pos1 = np.zeros((steps + 1, 2))
    pos2 = np.zeros((steps + 1, 2))
    return steps, times, pos1, pos2


def _initial_conditions(
    orbit: CircularOrbitState | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retourner des copies des conditions initiales pour éviter les mutations."""
    orbit = orbit or DEFAULT_ORBIT
    return (
        orbit.pos1.copy(),
        orbit.pos2.copy(),
        orbit.vel1.copy(),
        orbit.vel2.copy(),
    )


def _normalise_method_list(methods: Iterable[str]) -> Tuple[str, ...]:
    """Nettoyer et valider une liste de méthodes d'intégration."""
    if methods is None:
        raise ValueError("Une liste de méthodes doit être fournie.")
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in methods:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in ALLOWED_METHODS:
            raise ValueError(f"Méthode inconnue: {raw}")
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        raise ValueError("Aucune méthode valide n'a été fournie.")
    return tuple(ordered)


def gravitational_acceleration(
    pos1: np.ndarray,
    pos2: np.ndarray,
    m1: float,
    m2: float,
    G_value: float = G,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accélération gravitationnelle appliquée à chaque planète."""
    offset = pos2 - pos1
    dist_sq = float(np.dot(offset, offset))
    dist_sq = max(dist_sq, 1e-12)
    dist = dist_sq**0.5
    factor = G_value / (dist_sq * dist)  # correspond à G / |r|^3
    a1 = m2 * factor * offset
    a2 = -m1 * factor * offset
    return a1, a2


def analytical_positions(
    times: np.ndarray, orbit: CircularOrbitState | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positions exactes des deux planètes pour une orbite circulaire coplanaire.

    Parameters
    ----------
    times:
        Tableau d'instants (secondes).
    orbit:
        Paramètres décrivant l'orbite circulaire.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Positions de la planète 1 et 2 (shape = [n, 2]).
    """
    orbit = orbit or DEFAULT_ORBIT
    t = np.atleast_1d(np.asarray(times, dtype=float))
    angles = orbit.phase + orbit.omega * t
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    pos1 = np.stack((-orbit.r1 * cos_theta, -orbit.r1 * sin_theta), axis=-1)
    pos2 = np.stack((orbit.r2 * cos_theta, orbit.r2 * sin_theta), axis=-1)
    return pos1, pos2


def euler_integration(
    total_time: float, dt: float, orbit: CircularOrbitState | None = None
) -> Trajectory:
    """
    Intégration explicite d'Euler pour le problème à deux corps en 2D.

    Parameters
    ----------
    total_time:
        Durée totale désirée (secondes). Le dernier échantillon correspond à
        steps * dt, potentiellement légèrement différent selon l'arrondi.
    dt:
        Pas de temps (secondes).
    """
    orbit = orbit or DEFAULT_ORBIT
    steps, times, pos1, pos2 = _allocate_trajectory(total_time, dt)
    p1, p2, v1, v2 = _initial_conditions(orbit)

    pos1[0] = p1
    pos2[0] = p2

    for i in range(steps):
        a1, a2 = gravitational_acceleration(
            p1, p2, orbit.m1, orbit.m2, orbit.G_value
        )
        p1 = p1 + v1 * dt
        p2 = p2 + v2 * dt
        v1 = v1 + a1 * dt
        v2 = v2 + a2 * dt
        pos1[i + 1] = p1
        pos2[i + 1] = p2

    return Trajectory(times=times, pos1=pos1, pos2=pos2)


def velocity_verlet_integration(
    total_time: float, dt: float, orbit: CircularOrbitState | None = None
) -> Trajectory:
    """Schéma de Verlet vitesse (position explicite) pour le problème à deux corps."""
    orbit = orbit or DEFAULT_ORBIT
    steps, times, pos1, pos2 = _allocate_trajectory(total_time, dt)
    p1, p2, v1, v2 = _initial_conditions(orbit)
    a1, a2 = gravitational_acceleration(
        p1, p2, orbit.m1, orbit.m2, orbit.G_value
    )

    pos1[0] = p1
    pos2[0] = p2

    dt_sq = dt * dt

    for i in range(steps):
        p1 = p1 + v1 * dt + 0.5 * a1 * dt_sq
        p2 = p2 + v2 * dt + 0.5 * a2 * dt_sq
        a1_new, a2_new = gravitational_acceleration(
            p1, p2, orbit.m1, orbit.m2, orbit.G_value
        )
        v1 = v1 + 0.5 * (a1 + a1_new) * dt
        v2 = v2 + 0.5 * (a2 + a2_new) * dt
        a1, a2 = a1_new, a2_new
        pos1[i + 1] = p1
        pos2[i + 1] = p2

    return Trajectory(times=times, pos1=pos1, pos2=pos2)


def rk4_integration(
    total_time: float, dt: float, orbit: CircularOrbitState | None = None
) -> Trajectory:
    """
    Intégration de Runge-Kutta d'ordre 4 sur les équations du mouvement.

    Le vecteur d'état comprend positions et vitesses des deux planètes.
    """
    orbit = orbit or DEFAULT_ORBIT
    steps, times, pos1, pos2 = _allocate_trajectory(total_time, dt)
    p1, p2, v1, v2 = _initial_conditions(orbit)

    pos1[0] = p1
    pos2[0] = p2

    for i in range(steps):
        a1, a2 = gravitational_acceleration(
            p1, p2, orbit.m1, orbit.m2, orbit.G_value
        )
        k1_p1 = v1
        k1_p2 = v2
        k1_v1 = a1
        k1_v2 = a2

        p1_k2 = p1 + 0.5 * dt * k1_p1
        p2_k2 = p2 + 0.5 * dt * k1_p2
        v1_k2 = v1 + 0.5 * dt * k1_v1
        v2_k2 = v2 + 0.5 * dt * k1_v2
        a1_k2, a2_k2 = gravitational_acceleration(
            p1_k2, p2_k2, orbit.m1, orbit.m2, orbit.G_value
        )
        k2_p1 = v1_k2
        k2_p2 = v2_k2
        k2_v1 = a1_k2
        k2_v2 = a2_k2

        p1_k3 = p1 + 0.5 * dt * k2_p1
        p2_k3 = p2 + 0.5 * dt * k2_p2
        v1_k3 = v1 + 0.5 * dt * k2_v1
        v2_k3 = v2 + 0.5 * dt * k2_v2
        a1_k3, a2_k3 = gravitational_acceleration(
            p1_k3, p2_k3, orbit.m1, orbit.m2, orbit.G_value
        )
        k3_p1 = v1_k3
        k3_p2 = v2_k3
        k3_v1 = a1_k3
        k3_v2 = a2_k3

        p1_k4 = p1 + dt * k3_p1
        p2_k4 = p2 + dt * k3_p2
        v1_k4 = v1 + dt * k3_v1
        v2_k4 = v2 + dt * k3_v2
        a1_k4, a2_k4 = gravitational_acceleration(
            p1_k4, p2_k4, orbit.m1, orbit.m2, orbit.G_value
        )
        k4_p1 = v1_k4
        k4_p2 = v2_k4
        k4_v1 = a1_k4
        k4_v2 = a2_k4

        p1 = p1 + dt / 6.0 * (k1_p1 + 2.0 * k2_p1 + 2.0 * k3_p1 + k4_p1)
        p2 = p2 + dt / 6.0 * (k1_p2 + 2.0 * k2_p2 + 2.0 * k3_p2 + k4_p2)
        v1 = v1 + dt / 6.0 * (k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1)
        v2 = v2 + dt / 6.0 * (k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2)

        pos1[i + 1] = p1
        pos2[i + 1] = p2

    return Trajectory(times=times, pos1=pos1, pos2=pos2)


def integrate_two_body(
    method: str,
    total_time: float,
    dt: float,
    orbit: CircularOrbitState | None = None,
) -> Trajectory:
    """
    Intégrer le système à deux corps en utilisant la méthode spécifiée.

    Parameters
    ----------
    method:
        Nom de la méthode : 'euler', 'verlet', 'rk4'.
    """
    key = method.strip().lower()
    if key == "euler":
        return euler_integration(total_time, dt, orbit=orbit)
    if key in {"verlet", "velocity_verlet"}:
        return velocity_verlet_integration(total_time, dt, orbit=orbit)
    if key in {"rk4", "runge-kutta4", "runge_kutta4"}:
        return rk4_integration(total_time, dt, orbit=orbit)
    raise ValueError(f"Méthode d'intégration inconnue : {method}")


def compare_method_to_analytic(
    method: str,
    total_time: float,
    dt: float,
    orbit: CircularOrbitState | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compare une méthode d'intégration à la solution analytique circulaire.

    Returns
    -------
    Dict[str, np.ndarray]
        times: instants communs
        numerical_pos1 / numerical_pos2: intégration numérique
        analytic_pos1 / analytic_pos2: solution exacte
        error1 / error2: norme de l'erreur à chaque pas
    """
    orbit = orbit or DEFAULT_ORBIT
    traj = integrate_two_body(method, total_time, dt, orbit=orbit)
    analytic_pos1, analytic_pos2 = analytical_positions(traj.times, orbit=orbit)

    error1 = np.linalg.norm(traj.pos1 - analytic_pos1, axis=1)
    error2 = np.linalg.norm(traj.pos2 - analytic_pos2, axis=1)

    return {
        "method": method,
        "times": traj.times,
        "numerical_pos1": traj.pos1,
        "numerical_pos2": traj.pos2,
        "analytic_pos1": analytic_pos1,
        "analytic_pos2": analytic_pos2,
        "error1": error1,
        "error2": error2,
    }


def compare_euler_to_analytic(
    total_time: float, dt: float, orbit: CircularOrbitState | None = None
) -> Dict[str, np.ndarray]:
    """
    Compare la trajectoire obtenue par Euler à la solution analytique circulaire.

    Returns
    -------
    Dict[str, np.ndarray]
        times: instants communs
        numerical_pos1 / numerical_pos2: intégration numérique
        analytic_pos1 / analytic_pos2: solution exacte
        error1 / error2: norme de l'erreur à chaque pas
    """
    return compare_method_to_analytic("euler", total_time, dt, orbit=orbit)


def max_position_error(
    total_time: float,
    dt: float,
    method: str = "euler",
    orbit: CircularOrbitState | None = None,
) -> float:
    """Erreur maximale parmi les deux planètes pour (total_time, dt, méthode)."""
    results = compare_method_to_analytic(method, total_time, dt, orbit=orbit)
    return float(max(results["error1"].max(), results["error2"].max()))


def measure_runtime_seconds(
    method: str,
    total_time: float,
    dt: float,
    repeats: int = 1,
    orbit: CircularOrbitState | None = None,
) -> float:
    """Mesurer le temps moyen d'exécution d'une méthode."""
    if repeats <= 0:
        raise ValueError("Le nombre de répétitions doit être positif.")
    start = time.perf_counter()
    for _ in range(repeats):
        integrate_two_body(method, total_time, dt, orbit=orbit)
    duration = time.perf_counter() - start
    return duration / repeats


def plot_error_vs_parameters(
    dt_values: Iterable[float] | None = None,
    total_times: Iterable[float] | None = None,
    reference_total_time: float = 20.0,
    reference_dt: float = 0.01,
    methods: Iterable[str] | None = None,
    orbit: CircularOrbitState | None = None,
) -> None:
    """
    Trace l'erreur maximale en faisant varier dt et/ou la durée totale.

    - Si dt_values est fourni, l'erreur est tracée en fonction de dt (total_time fixe).
    - Si total_times est fourni, l'erreur est tracée en fonction de total_time (dt fixe).
    - Plusieurs méthodes peuvent être tracées simultanément.
    """
    if dt_values is None and total_times is None:
        raise ValueError("Fournir au moins une liste de valeurs pour dt ou total_time.")

    method_list = _normalise_method_list(methods or ALLOWED_METHODS)
    orbit = orbit or DEFAULT_ORBIT

    def _label(name: str) -> str:
        return name.replace("_", " ").title()

    rows = 1 if total_times is None or dt_values is None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4.5 * rows), squeeze=False)

    row_idx = 0

    if dt_values is not None:
        ax = axes[row_idx][0]
        dt_values = np.asarray(list(dt_values), dtype=float)
        if np.any(dt_values <= 0):
            raise ValueError("Les valeurs de dt doivent être strictement positives.")
        for method in method_list:
            errors = [
                max_position_error(
                    reference_total_time, dt, method=method, orbit=orbit
                )
                for dt in dt_values
            ]
            ax.plot(dt_values, errors, marker="o", label=_label(method))
        if dt_values.size > 1 and dt_values.max() / dt_values.min() > 20.0:
            ax.set_xscale("log")
        ax.set_xlabel("Pas de temps dt")
        ax.set_ylabel("Erreur maximale")
        ax.set_title(f"Erreur max vs dt (T = {reference_total_time}s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        row_idx += 1

    if total_times is not None:
        ax = axes[row_idx][0]
        total_times = np.asarray(list(total_times), dtype=float)
        if np.any(total_times <= 0):
            raise ValueError("Les durées totales doivent être strictement positives.")
        for method in method_list:
            errors = [
                max_position_error(T, reference_dt, method=method, orbit=orbit)
                for T in total_times
            ]
            ax.plot(total_times, errors, marker="o", label=_label(method))
        ax.set_xlabel("Durée totale T")
        ax.set_ylabel("Erreur maximale")
        ax.set_title(f"Erreur max vs T (dt = {reference_dt}s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plt.show()


def animate_positions(
    traj: Trajectory,
    orbit: CircularOrbitState | None = None,
    interval_ms: int = 50,
    trail_length: int = 0,
) -> None:
    """
    Anime les positions des deux planètes à partir d'une `Trajectory`.

    Parameters
    ----------
    traj:
        Objet `Trajectory` contenant `times`, `pos1`, `pos2`.
    orbit:
        Optionnel, utilisé pour ajuster l'échelle si nécessaire.
    interval_ms:
        Millisecondes entre les images.
    trail_length:
        Longueur de la traînée (nombre d'images précédentes à afficher). 0 = pas de traînée.
    """
    times = traj.times
    p1 = traj.pos1
    p2 = traj.pos2

    # calculer l'étendue pour cadrer l'animation
    stacked = np.vstack((p1, p2)) if p1.size and p2.size else np.zeros((1, 2))
    max_abs = float(np.max(np.abs(stacked))) if stacked.size else 1.0
    margin = max_abs * 0.15 + 1e-6

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max_abs - margin, max_abs + margin)
    ax.set_ylim(-max_abs - margin, max_abs + margin)
    ax.grid(True, alpha=0.25)

    scat1 = ax.plot([], [], "o", color="C0", markersize=8, label="Planète 1")[0]
    scat2 = ax.plot([], [], "o", color="C1", markersize=8, label="Planète 2")[0]
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    trail1_lines: list = []
    trail2_lines: list = []

    def init():
        scat1.set_data([], [])
        scat2.set_data([], [])
        time_text.set_text("")
        for ln in trail1_lines + trail2_lines:
            ln.set_data([], [])
        return [scat1, scat2, time_text] + trail1_lines + trail2_lines

    def update(frame: int):
        x1, y1 = p1[frame, 0], p1[frame, 1]
        x2, y2 = p2[frame, 0], p2[frame, 1]
        scat1.set_data(x1, y1)
        scat2.set_data(x2, y2)
        time_text.set_text(f"t = {times[frame]:.3f} s")

        # traînées
        if trail_length > 0:
            start = max(0, frame - trail_length)
            trail1_x = p1[start : frame + 1, 0]
            trail1_y = p1[start : frame + 1, 1]
            trail2_x = p2[start : frame + 1, 0]
            trail2_y = p2[start : frame + 1, 1]
            # effacer et redessiner une seule ligne par traînée
            if not trail1_lines:
                l1, = ax.plot(trail1_x, trail1_y, color="C0", alpha=0.6)
                l2, = ax.plot(trail2_x, trail2_y, color="C1", alpha=0.6)
                trail1_lines.append(l1)
                trail2_lines.append(l2)
            else:
                trail1_lines[0].set_data(trail1_x, trail1_y)
                trail2_lines[0].set_data(trail2_x, trail2_y)

        return [scat1, scat2, time_text] + trail1_lines + trail2_lines

    ani = FuncAnimation(fig, update, frames=len(times), init_func=init, interval=interval_ms, blit=False)
    ax.legend()
    plt.show()


def save_positions_plot(
    traj: Trajectory,
    analytic_pos1: np.ndarray,
    analytic_pos2: np.ndarray,
    method_name: str,
    filename: str,
) -> None:
    """
    Sauvegarde une image (PNG) montrant les trajectoires analytiques et numériques.

    - `analytic_pos2` est tracée en trait pointillé noir.
    - `traj.pos1` et `traj.pos2` sont tracées avec des couleurs différentes.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    # empiler pour calculer l'échelle
    all_pts = np.vstack((analytic_pos1, analytic_pos2, traj.pos1, traj.pos2))
    max_abs = float(np.max(np.abs(all_pts))) if all_pts.size else 1.0
    margin = max_abs * 0.15 + 1e-6
    ax.set_xlim(-max_abs - margin, max_abs + margin)
    ax.set_ylim(-max_abs - margin, max_abs + margin)

    ax.plot(analytic_pos2[:, 0], analytic_pos2[:, 1], "--", color="black", lw=2, label="Analytique (planète 2)")
    ax.plot(traj.pos1[:, 0], traj.pos1[:, 1], color="C0", lw=1, label=f"{method_name} planète 1")
    ax.plot(traj.pos2[:, 0], traj.pos2[:, 1], color="C1", lw=1, label=f"{method_name} planète 2")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectoires — méthode: {method_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_time_series_errors(
    total_time: float,
    dt: float,
    methods: Iterable[str] | None = None,
    orbit: CircularOrbitState | None = None,
) -> None:
    """Trace l'évolution temporelle de l'erreur maximale pour chaque méthode."""
    method_list = _normalise_method_list(methods or ALLOWED_METHODS)
    orbit = orbit or DEFAULT_ORBIT

    def _label(name: str) -> str:
        return name.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in method_list:
        results = compare_method_to_analytic(method, total_time, dt, orbit=orbit)
        max_error = np.maximum(results["error1"], results["error2"])
        ax.plot(results["times"], max_error, label=_label(method))

    ax.set_xlabel("Temps")
    ax.set_ylabel("Erreur maximale")
    ax.set_title(f"Erreur temporelle (dt = {dt}, T ≈ {total_time})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_verlet_rk4_parameter_zoom(
    reference_total_time: float,
    dt_values: Iterable[float] | None = None,
    total_times: Iterable[float] | None = None,
    orbit: CircularOrbitState | None = None,
) -> None:
    """
    Graphique supplémentaire focalisé sur Verlet et RK4.

    Les valeurs fournies permettent de zoomer sur des régions d'intérêt
    différentes de celles utilisées pour les courbes globales.
    """
    methods = ("verlet", "rk4")
    if dt_values is None and total_times is None:
        dt_values = np.logspace(-2.5, -1, 10)
    dt_list = list(dt_values) if dt_values is not None else None
    total_list = list(total_times) if total_times is not None else None
    reference_dt = dt_list[0] if dt_list else 0.01
    plot_error_vs_parameters(
        dt_values=dt_list,
        total_times=total_list,
        reference_total_time=reference_total_time,
        reference_dt=reference_dt,
        methods=methods,
        orbit=orbit,
    )


def plot_runtime_vs_dt(
    total_time: float,
    dt_values: Iterable[float],
    methods: Iterable[str] | None = None,
    repeats: int = 3,
    orbit: CircularOrbitState | None = None,
) -> None:
    """Tracer le temps d'exécution moyen des méthodes en fonction de dt."""
    method_list = _normalise_method_list(methods or ALLOWED_METHODS)
    orbit = orbit or DEFAULT_ORBIT
    dt_values = np.asarray(list(dt_values), dtype=float)
    if dt_values.size == 0:
        raise ValueError("Fournir au moins une valeur de dt pour les mesures.")
    if np.any(dt_values <= 0):
        raise ValueError("Les valeurs de dt doivent être strictement positives.")
    if repeats <= 0:
        raise ValueError("Le nombre de répétitions doit être positif.")

    def _label(name: str) -> str:
        return name.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in method_list:
        timings = [
            measure_runtime_seconds(method, total_time, dt, repeats, orbit=orbit)
            for dt in dt_values
        ]
        ax.plot(dt_values, timings, marker="o", label=_label(method))
    if dt_values.size > 1 and dt_values.max() / dt_values.min() > 20.0:
        ax.set_xscale("log")
    ax.set_xlabel("Pas de temps dt")
    ax.set_ylabel("Temps moyen par intégration (s)")
    ax.set_title(
        f"Temps d'exécution moyen ({repeats} répétition(s) par point, T = {total_time}s)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def launch_interactive_dashboard(
    total_time: float,
    dt: float,
    orbit: CircularOrbitState | None = None,
    methods: Iterable[str] | None = None,
) -> None:
    """
    Interface interactive pour explorer l'influence des conditions initiales.

    Des curseurs permettent d'ajuster masses, séparation, phase initiale et pas de
    temps. Les graphiques se mettent à jour instantanément : trajectoires à gauche,
    erreurs temporelles à droite.
    """
    base_orbit = orbit or DEFAULT_ORBIT
    method_list = _normalise_method_list(methods or ALLOWED_METHODS)

    fig, (ax_orbit, ax_error) = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.32)

    analytic_line, = ax_orbit.plot([], [], "--", color="black", lw=2, label="Analytique planète 2")
    numerical_lines: dict[str, any] = {}
    error_lines: dict[str, any] = {}

    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.grid(True, alpha=0.3)

    ax_error.set_xlabel("Temps")
    ax_error.set_ylabel("Erreur maximale")
    ax_error.set_title("Erreur au cours du temps")
    ax_error.grid(True, alpha=0.3)

    error_text = ax_error.text(
        0.02,
        0.95,
        "",
        transform=ax_error.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    slider_ax_m1 = fig.add_axes([0.1, 0.25, 0.35, 0.03])
    slider_ax_m2 = fig.add_axes([0.55, 0.25, 0.35, 0.03])
    slider_ax_sep = fig.add_axes([0.1, 0.20, 0.35, 0.03])
    slider_ax_phase = fig.add_axes([0.55, 0.20, 0.35, 0.03])
    slider_ax_dt = fig.add_axes([0.1, 0.15, 0.35, 0.03])
    reset_ax = fig.add_axes([0.55, 0.15, 0.15, 0.04])

    slider_m1 = Slider(slider_ax_m1, "m1", valmin=0.1, valmax=5.0, valinit=base_orbit.m1)
    slider_m2 = Slider(slider_ax_m2, "m2", valmin=0.1, valmax=5.0, valinit=base_orbit.m2)
    slider_sep = Slider(
        slider_ax_sep, "Séparation", valmin=0.5, valmax=6.0, valinit=base_orbit.separation
    )
    slider_phase = Slider(
        slider_ax_phase,
        "Phase (°)",
        valmin=0.0,
        valmax=360.0,
        valinit=np.degrees(base_orbit.phase),
    )
    slider_dt = Slider(
        slider_ax_dt,
        "dt",
        valmin=1e-3,
        valmax=0.1,
        valinit=max(dt, 1e-3),
    )

    reset_button = Button(reset_ax, "Réinitialiser")

    def _format_error_summary(results_by_method: dict[str, Dict[str, np.ndarray]]) -> str:
        lines = []
        for method in method_list:
            data = results_by_method[method]
            max_err = float(np.maximum(data["error1"], data["error2"]).max())
            lines.append(f"{method}: {max_err:.2e}")
        return "\n".join(lines)

    def _compute_results() -> tuple[CircularOrbitState, dict[str, Dict[str, np.ndarray]]]:
        new_orbit = build_circular_orbit_state(
            m1=slider_m1.val,
            m2=slider_m2.val,
            separation=slider_sep.val,
            phase=np.deg2rad(slider_phase.val),
            G_value=base_orbit.G_value,
        )
        current_dt = max(slider_dt.val, 1e-6)
        results = {
            method: compare_method_to_analytic(method, total_time, current_dt, orbit=new_orbit)
            for method in method_list
        }
        return new_orbit, results

    def _update(_: float | None = None) -> None:
        new_orbit, results = _compute_results()
        reference = results[method_list[0]]
        analytic_pos1, analytic_pos2 = analytical_positions(reference["times"], orbit=new_orbit)

        analytic_line.set_data(analytic_pos2[:, 0], analytic_pos2[:, 1])

        for method, data in results.items():
            numerical_lines[method].set_data(
                data["numerical_pos2"][:, 0], data["numerical_pos2"][:, 1]
            )
            error_lines[method].set_data(
                data["times"], np.maximum(data["error1"], data["error2"])
            )

        all_points = [analytic_pos1, analytic_pos2]
        for data in results.values():
            all_points.extend((data["numerical_pos1"], data["numerical_pos2"]))
        stacked = np.vstack(all_points)
        radius = np.max(np.abs(stacked)) * 1.05 if stacked.size else 1.0
        ax_orbit.set_xlim(-radius, radius)
        ax_orbit.set_ylim(-radius, radius)

        ax_error.relim()
        ax_error.autoscale()

        error_text.set_text(_format_error_summary(results))
        fig.canvas.draw_idle()

    initial_orbit, initial_results = _compute_results()
    reference_init = initial_results[method_list[0]]
    analytic_init_1, analytic_init_2 = analytical_positions(
        reference_init["times"], orbit=initial_orbit
    )
    analytic_line.set_data(analytic_init_2[:, 0], analytic_init_2[:, 1])

    for method, data in initial_results.items():
        (orbit_line,) = ax_orbit.plot(
            data["numerical_pos2"][:, 0],
            data["numerical_pos2"][:, 1],
            label=f"{method} planète 2",
        )
        numerical_lines[method] = orbit_line

        (error_line,) = ax_error.plot(
            data["times"],
            np.maximum(data["error1"], data["error2"]),
            label=method,
        )
        error_lines[method] = error_line

    _update()
    ax_orbit.legend(loc="upper right")
    ax_error.legend(loc="upper right")

    def _on_reset(_: object) -> None:
        slider_m1.reset()
        slider_m2.reset()
        slider_sep.reset()
        slider_phase.reset()
        slider_dt.reset()

    for slider in (slider_m1, slider_m2, slider_sep, slider_phase, slider_dt):
        slider.on_changed(_update)

    reset_button.on_clicked(_on_reset)

    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Comparer différentes méthodes d'intégration (Euler, Verlet, RK4) "
            "pour un système à deux corps en orbite circulaire."
        )
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=20.0,
        help="Durée totale intégrée pour les comparaisons (défaut: 20.0).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Pas de temps utilisé pour les comparaisons ponctuelles (défaut: 0.02).",
    )
    parser.add_argument(
        "--m1",
        type=float,
        default=M1,
        help="Masse de la planète 1 (défaut: 1.0).",
    )
    parser.add_argument(
        "--m2",
        type=float,
        default=M2,
        help="Masse de la planète 2 (défaut: 0.8).",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=SEPARATION,
        help="Distance initiale entre les deux planètes (défaut: 2.0).",
    )
    parser.add_argument(
        "--phase-deg",
        type=float,
        default=0.0,
        help="Phase initiale en degrés (0° correspond à l'axe +x pour la planète 2).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(ALLOWED_METHODS),
        help="Méthodes à inclure (choix: euler, verlet, rk4).",
    )
    parser.add_argument(
        "--dt-values",
        type=float,
        nargs="+",
        default=None,
        help="Liste de valeurs de dt pour les courbes d'erreur (sinon logspace par défaut).",
    )
    parser.add_argument(
        "--total-time-values",
        type=float,
        nargs="+",
        default=None,
        help="Liste de durées totales pour les courbes d'erreur (sinon linspace par défaut).",
    )
    parser.add_argument(
        "--verlet-rk4-dt-values",
        type=float,
        nargs="+",
        default=None,
        help="Valeurs de dt dédiées au graphique supplémentaire (Verlet vs RK4).",
    )
    parser.add_argument(
        "--runtime-dt-values",
        type=float,
        nargs="+",
        default=None,
        help="Valeurs de dt pour le graphique des temps d'exécution.",
    )
    parser.add_argument(
        "--runtime-methods",
        type=str,
        nargs="+",
        default=None,
        help="Sous-ensemble de méthodes pour le graphique des temps d'exécution.",
    )
    parser.add_argument(
        "--runtime-repeats",
        type=int,
        default=2,
        help="Nombre de répétitions pour la moyenne des temps d'exécution (défaut: 2).",
    )
    parser.add_argument(
        "--skip-time-series",
        action="store_true",
        help="Ne pas afficher l'évolution temporelle de l'erreur.",
    )
    parser.add_argument(
        "--skip-parameter-plot",
        action="store_true",
        help="Ne pas afficher les courbes d'erreur en fonction de dt / T.",
    )
    parser.add_argument(
        "--skip-verlet-rk4-plot",
        action="store_true",
        help="Ne pas afficher le graphique supplémentaire (Verlet vs RK4).",
    )
    parser.add_argument(
        "--skip-runtime-plot",
        action="store_true",
        help="Ne pas afficher le graphique des temps d'exécution.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Ne pas afficher le résumé des erreurs maximales en console.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ouvrir le tableau de bord interactif après les graphiques statiques.",
    )
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="N'afficher que le tableau de bord interactif (ignore les graphiques statiques).",
    )
    parser.add_argument(
        "--save-positions",
        action="store_true",
        help="Sauvegarder des images des trajectoires des planètes pour chaque méthode (PNG).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    methods = _normalise_method_list(args.methods)
    runtime_methods = (
        _normalise_method_list(args.runtime_methods)
        if args.runtime_methods is not None
        else methods
    )

    dt_values_default = (
        args.dt_values if args.dt_values is not None else np.logspace(-2, -0.5, 10)
    )
    total_time_values_default = (
        args.total_time_values
        if args.total_time_values is not None
        else np.linspace(5.0, 40.0, 10)
    )
    runtime_dt_values_default = (
        args.runtime_dt_values
        if args.runtime_dt_values is not None
        else dt_values_default
    )

    phase_rad = np.deg2rad(args.phase_deg)
    orbit = build_circular_orbit_state(
        m1=args.m1,
        m2=args.m2,
        separation=args.separation,
        phase=phase_rad,
        G_value=G,
    )

    interactive_requested = args.interactive or args.interactive_only

    if args.interactive_only:
        launch_interactive_dashboard(args.total_time, args.dt, orbit=orbit, methods=methods)
        return

    if args.save_positions:
        # Générer et sauvegarder des images des trajectoires pour chaque méthode.
        for method in methods:
            traj = integrate_two_body(method, args.total_time, args.dt, orbit=orbit)
            analytic_p1, analytic_p2 = analytical_positions(traj.times, orbit=orbit)
            filename = f"trajectoire_{method}.png"
            save_positions_plot(traj, analytic_p1, analytic_p2, method, filename)
            print(f"Sauvegardé: {filename}")

    if not args.no_summary:
        for method in methods:
            results = compare_method_to_analytic(
                method, args.total_time, args.dt, orbit=orbit
            )
            max_err = float(np.maximum(results["error1"], results["error2"]).max())
            print(f"[{method}] erreur max : {max_err:.3e}")

    if not args.skip_time_series:
        plot_time_series_errors(
            args.total_time, args.dt, methods=methods, orbit=orbit
        )

    if not args.skip_parameter_plot:
        plot_error_vs_parameters(
            dt_values=dt_values_default,
            total_times=total_time_values_default,
            reference_total_time=args.total_time,
            reference_dt=args.dt,
            methods=methods,
            orbit=orbit,
        )

    if not args.skip_verlet_rk4_plot:
        plot_verlet_rk4_parameter_zoom(
            reference_total_time=args.total_time,
            dt_values=args.verlet_rk4_dt_values,
            orbit=orbit,
        )

    if not args.skip_runtime_plot:
        plot_runtime_vs_dt(
            args.total_time,
            dt_values=runtime_dt_values_default,
            methods=runtime_methods,
            repeats=args.runtime_repeats,
            orbit=orbit,
        )

    if interactive_requested:
        launch_interactive_dashboard(
            args.total_time, args.dt, orbit=orbit, methods=methods
        )


if __name__ == "__main__":
    main()
