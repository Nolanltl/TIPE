#!/usr/bin/env python3
"""
Interactive two-body simulation in 3D.

The script opens a Matplotlib window that renders the trajectories of two bodies
in mutual gravitation. Sliders and text boxes let you tweak masses, initial
positions, and velocities without restarting the program. A dedicated time
slider controls playback, letting you scrub the animation forward or backward.
Use the play and reverse buttons to automate playback.

Units are arbitrary and normalised so that the gravitational constant G is set
to 1.0. The default configuration starts in the centre-of-mass frame, but you
are free to provide any initial conditions; they are automatically recentred to
avoid drift when possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

G = 1.0  # Normalised gravitational constant
DEFAULT_TOTAL_TIME = 40.0
DEFAULT_DT = 0.02


@dataclass
class SimulationParams:
    m1: float
    m2: float
    pos1: np.ndarray
    pos2: np.ndarray
    vel1: np.ndarray
    vel2: np.ndarray


def vector_to_string(vec: np.ndarray) -> str:
    """Return a compact string representation of a 3D vector."""
    return f"{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}"


def parse_vector(text: str, fallback: np.ndarray) -> np.ndarray:
    """Parse a user-supplied vector string, falling back on invalid input."""
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not numbers:
        return fallback
    values = [float(num) for num in numbers[:3]]
    if len(values) < 3:
        values += list(fallback[len(values):])
    return np.array(values[:3])


def to_center_of_mass_frame(
    pos1: np.ndarray,
    pos2: np.ndarray,
    vel1: np.ndarray,
    vel2: np.ndarray,
    m1: float,
    m2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return positions and velocities expressed in the centre-of-mass frame."""
    total_mass = m1 + m2
    com_pos = (m1 * pos1 + m2 * pos2) / total_mass
    com_vel = (m1 * vel1 + m2 * vel2) / total_mass
    return (
        pos1 - com_pos,
        pos2 - com_pos,
        vel1 - com_vel,
        vel2 - com_vel,
    )


def acceleration(
    pos1: np.ndarray, pos2: np.ndarray, m1: float, m2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gravitational acceleration on each body."""
    offset = pos2 - pos1
    dist = np.linalg.norm(offset)
    dist = max(dist, 1e-6)  # Avoid division by zero
    factor = G / dist**3
    a1 = m2 * factor * offset
    a2 = -m1 * factor * offset
    return a1, a2


def integrate_orbit(
    pos1: np.ndarray,
    pos2: np.ndarray,
    vel1: np.ndarray,
    vel2: np.ndarray,
    m1: float,
    m2: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the motion for a given number of steps using velocity Verlet."""
    positions1 = np.zeros((steps + 1, 3))
    positions2 = np.zeros((steps + 1, 3))

    p1 = pos1.copy()
    p2 = pos2.copy()
    v1 = vel1.copy()
    v2 = vel2.copy()
    a1, a2 = acceleration(p1, p2, m1, m2)

    positions1[0] = p1
    positions2[0] = p2

    dt_sq = dt * dt

    for i in range(1, steps + 1):
        p1 = p1 + v1 * dt + 0.5 * a1 * dt_sq
        p2 = p2 + v2 * dt + 0.5 * a2 * dt_sq
        a1_new, a2_new = acceleration(p1, p2, m1, m2)
        v1 = v1 + 0.5 * (a1 + a1_new) * dt
        v2 = v2 + 0.5 * (a2 + a2_new) * dt
        a1, a2 = a1_new, a2_new
        positions1[i] = p1
        positions2[i] = p2

    return positions1, positions2


def simulate_system(
    params: SimulationParams,
    total_time: float = DEFAULT_TOTAL_TIME,
    dt: float = DEFAULT_DT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times and positions for both bodies across the simulation window."""
    steps = max(1, int(round(total_time / dt)))
    total_time = steps * dt

    pos1_cm, pos2_cm, vel1_cm, vel2_cm = to_center_of_mass_frame(
        params.pos1, params.pos2, params.vel1, params.vel2, params.m1, params.m2
    )

    forward1, forward2 = integrate_orbit(
        pos1_cm, pos2_cm, vel1_cm, vel2_cm, params.m1, params.m2, dt, steps
    )
    backward1, backward2 = integrate_orbit(
        pos1_cm, pos2_cm, -vel1_cm, -vel2_cm, params.m1, params.m2, dt, steps
    )

    times_forward = np.linspace(0.0, total_time, steps + 1)
    times_negative = -times_forward[1:][::-1]
    times = np.concatenate((times_negative, times_forward))

    positions1 = np.vstack((backward1[1:][::-1], forward1))
    positions2 = np.vstack((backward2[1:][::-1], forward2))

    return times, positions1, positions2


class TwoBodyApp:
    """Matplotlib-based UI that ties the simulation and interactive controls."""

    def __init__(self) -> None:
        defaults = SimulationParams(
            m1=1.2,
            m2=0.9,
            pos1=np.array([-0.7, 0.0, 0.0]),
            pos2=np.array([0.7, 0.0, 0.0]),
            vel1=np.array([0.0, -0.35, 0.25]),
            vel2=np.array([0.0, 0.45, -0.2]),
        )

        self.params = defaults
        self.total_time = DEFAULT_TOTAL_TIME
        self.dt = DEFAULT_DT

        self.times = np.array([0.0])
        self.pos1 = np.zeros((1, 3))
        self.pos2 = np.zeros((1, 3))
        self.current_index = 0
        self.time_step = self.dt

        self.is_playing = False
        self.play_direction = 1
        self.updating_controls = False

        self._build_figure()
        self._connect_widgets()
        self._recompute()
        self._start_timer()

    # --------------------------------------------------------------------- UI
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(12, 7))
        self.ax3d = self.fig.add_axes([0.05, 0.25, 0.6, 0.7], projection="3d")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("Simulation gravitationnelle à deux corps")

        self.time_slider_ax = self.fig.add_axes([0.05, 0.16, 0.6, 0.03])
        self.time_slider = Slider(
            self.time_slider_ax,
            "Temps",
            valmin=-self.total_time,
            valmax=self.total_time,
            valinit=0.0,
            valstep=self.dt,
        )

        self.speed_slider_ax = self.fig.add_axes([0.05, 0.08, 0.6, 0.03])
        self.speed_slider = Slider(
            self.speed_slider_ax,
            "Vitesse lecture",
            valmin=0.1,
            valmax=10.0,
            valinit=1.0,
        )

        panel_x = 0.72
        panel_w = 0.23

        self.mass1_ax = self.fig.add_axes([panel_x, 0.82, panel_w, 0.04])
        self.mass1_slider = Slider(
            self.mass1_ax,
            "Masse 1",
            valmin=0.1,
            valmax=5.0,
            valinit=self.params.m1,
        )

        self.mass2_ax = self.fig.add_axes([panel_x, 0.76, panel_w, 0.04])
        self.mass2_slider = Slider(
            self.mass2_ax,
            "Masse 2",
            valmin=0.1,
            valmax=5.0,
            valinit=self.params.m2,
        )

        textbox_h = 0.05
        spacing = 0.06
        start_y = 0.68

        self.pos1_ax = self.fig.add_axes([panel_x, start_y, panel_w, textbox_h])
        self.pos1_box = TextBox(
            self.pos1_ax, "Pos 1 (x,y,z)", initial=vector_to_string(self.params.pos1)
        )

        self.pos2_ax = self.fig.add_axes([panel_x, start_y - spacing, panel_w, textbox_h])
        self.pos2_box = TextBox(
            self.pos2_ax, "Pos 2 (x,y,z)", initial=vector_to_string(self.params.pos2)
        )

        self.vel1_ax = self.fig.add_axes([panel_x, start_y - 2 * spacing, panel_w, textbox_h])
        self.vel1_box = TextBox(
            self.vel1_ax, "Vit 1 (x,y,z)", initial=vector_to_string(self.params.vel1)
        )

        self.vel2_ax = self.fig.add_axes([panel_x, start_y - 3 * spacing, panel_w, textbox_h])
        self.vel2_box = TextBox(
            self.vel2_ax, "Vit 2 (x,y,z)", initial=vector_to_string(self.params.vel2)
        )

        self.play_ax = self.fig.add_axes([panel_x, 0.23, 0.11, 0.06])
        self.play_button = Button(self.play_ax, "Lecture")

        self.reverse_ax = self.fig.add_axes([panel_x + 0.12, 0.23, 0.11, 0.06])
        self.reverse_button = Button(self.reverse_ax, "Forward")

        self.reset_ax = self.fig.add_axes([panel_x, 0.15, 0.23, 0.05])
        self.reset_button = Button(self.reset_ax, "Réinitialiser temps")

        # Plot elements
        (self.orbit1_line,) = self.ax3d.plot([], [], [], color="tab:blue", lw=1.2, alpha=0.7)
        (self.orbit2_line,) = self.ax3d.plot([], [], [], color="tab:orange", lw=1.2, alpha=0.7)
        (self.body1_marker,) = self.ax3d.plot(
            [], [], [], marker="o", markersize=10, color="tab:blue"
        )
        (self.body2_marker,) = self.ax3d.plot(
            [], [], [], marker="o", markersize=10, color="tab:orange"
        )
        (self.com_marker,) = self.ax3d.plot(
            [0.0], [0.0], [0.0], marker="x", markersize=6, color="gray"
        )

        self.time_text = self.ax3d.text2D(0.02, 0.95, "", transform=self.ax3d.transAxes)

    def _connect_widgets(self) -> None:
        self.time_slider.on_changed(self._on_time_change)
        self.speed_slider.on_changed(self._on_speed_change)
        self.mass1_slider.on_changed(lambda val: self._update_mass(1, val))
        self.mass2_slider.on_changed(lambda val: self._update_mass(2, val))

        self.pos1_box.on_submit(lambda text: self._update_vector("pos1", text))
        self.pos2_box.on_submit(lambda text: self._update_vector("pos2", text))
        self.vel1_box.on_submit(lambda text: self._update_vector("vel1", text))
        self.vel2_box.on_submit(lambda text: self._update_vector("vel2", text))

        self.play_button.on_clicked(self._toggle_playback)
        self.reverse_button.on_clicked(self._toggle_direction)
        self.reset_button.on_clicked(self._reset_time)

    # -------------------------------------------------------------- Callbacks
    def _update_mass(self, index: int, value: float) -> None:
        if self.updating_controls:
            return
        if index == 1:
            self.params.m1 = max(0.1, float(value))
        else:
            self.params.m2 = max(0.1, float(value))
        self._recompute()

    def _update_vector(self, key: str, text: str) -> None:
        if self.updating_controls:
            return
        vector = parse_vector(text, getattr(self.params, key))
        setattr(self.params, key, vector)
        self._recompute()

    def _toggle_playback(self, _event) -> None:
        self.is_playing = not self.is_playing
        self.play_button.label.set_text("Pause" if self.is_playing else "Lecture")

    def _toggle_direction(self, _event) -> None:
        self.play_direction *= -1
        label = "Reverse" if self.play_direction < 0 else "Forward"
        self.reverse_button.label.set_text(label)

    def _reset_time(self, _event) -> None:
        self.is_playing = False
        self.play_button.label.set_text("Lecture")
        self.time_slider.set_val(0.0)

    def _on_time_change(self, value: float) -> None:
        if self.updating_controls or len(self.times) == 0:
            return
        idx = np.searchsorted(self.times, value)
        if idx >= len(self.times):
            idx = len(self.times) - 1
        elif idx > 0:
            if abs(value - self.times[idx - 1]) < abs(self.times[idx] - value):
                idx -= 1
        if idx != self.current_index:
            self.current_index = idx
            self._update_plot()
        self.time_text.set_text(f"t = {self.times[self.current_index]:.2f}")

    def _on_speed_change(self, _value: float) -> None:
        # No immediate action needed; the new value is consumed by the timer callback.
        pass

    # ------------------------------------------------------- Simulation cycle
    def _recompute(self) -> None:
        self.updating_controls = True
        try:
            times, pos1, pos2 = simulate_system(self.params, self.total_time, self.dt)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Erreur pendant la simulation: {exc}")
            return
        finally:
            self.updating_controls = False

        self.times = times
        self.pos1 = pos1
        self.pos2 = pos2
        self.time_step = self.times[1] - self.times[0] if len(self.times) > 1 else self.dt
        self.current_index = np.searchsorted(self.times, self.time_slider.val)
        self.current_index = np.clip(self.current_index, 0, len(self.times) - 1)

        self.orbit1_line.set_data_3d(self.pos1[:, 0], self.pos1[:, 1], self.pos1[:, 2])
        self.orbit2_line.set_data_3d(self.pos2[:, 0], self.pos2[:, 1], self.pos2[:, 2])

        self._set_axes_limits()
        self._update_plot()

        self.fig.canvas.draw_idle()

    def _update_plot(self) -> None:
        if len(self.pos1) == 0:
            return

        p1 = self.pos1[self.current_index]
        p2 = self.pos2[self.current_index]

        self.body1_marker.set_data_3d([p1[0]], [p1[1]], [p1[2]])
        self.body2_marker.set_data_3d([p2[0]], [p2[1]], [p2[2]])

        self.time_text.set_text(f"t = {self.times[self.current_index]:.2f}")

    def _set_axes_limits(self) -> None:
        all_points = np.vstack((self.pos1, self.pos2))
        maxima = np.max(all_points, axis=0)
        minima = np.min(all_points, axis=0)
        center = 0.5 * (maxima + minima)
        radius = np.max(maxima - minima) * 0.6
        if not np.isfinite(radius) or radius < 1e-3:
            radius = 1.0

        self.ax3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax3d.set_zlim(center[2] - radius, center[2] + radius)
        try:
            self.ax3d.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass  # Matplotlib < 3.3 fallback

    # ------------------------------------------------------------ Timer loop
    def _start_timer(self) -> None:
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self._advance_time)
        self.timer.start()

    def _advance_time(self) -> None:
        if not self.is_playing or len(self.times) == 0:
            return
        speed = self.speed_slider.val
        step = self.play_direction * self.time_step * max(speed, 0.1)
        new_time = self.time_slider.val + step

        if new_time > self.time_slider.valmax:
            new_time = self.time_slider.valmax
            self.is_playing = False
            self.play_button.label.set_text("Lecture")
        elif new_time < self.time_slider.valmin:
            new_time = self.time_slider.valmin
            self.is_playing = False
            self.play_button.label.set_text("Lecture")

        self.time_slider.set_val(new_time)


def main() -> None:
    TwoBodyApp()
    plt.show()


if __name__ == "__main__":
    main()
