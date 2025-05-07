# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 03:45:15 2025

@author: emrek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## GWO (Grey Wolf Optimizer) tabanlı Kamikaze Drone Sürüsü

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def move(self, alpha_pos, beta_pos, delta_pos, a, step_size=2.0):
        if not self.alive:
            return

        r1, r2 = np.random.rand(2), np.random.rand(2)

        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        D_alpha = np.abs(C1 * alpha_pos - self.position)
        X1 = alpha_pos - A1 * D_alpha

        r1, r2 = np.random.rand(2), np.random.rand(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = np.abs(C2 * beta_pos - self.position)
        X2 = beta_pos - A2 * D_beta

        r1, r2 = np.random.rand(2), np.random.rand(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = np.abs(C3 * delta_pos - self.position)
        X3 = delta_pos - A3 * D_delta

        new_pos = (X1 + X2 + X3) / 3

        # Adımı sınırlayalım
        direction = new_pos - self.position
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size

        self.position += direction
        self.path.append(self.position.copy())

    def check_impact(self, impact_radius=2):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")
            return True
        return False

# Başlangıç ayarları
num_drones = 30
target_position = np.array([100, 100])

# Drone sürüsü oluştur
drones = [Drone(position=np.random.rand(2) * 50, target=target_position, drone_id=i) for i in range(num_drones)]

# Simülasyon parametreleri
max_steps = 1000

# Animasyon için veri
fig, ax = plt.subplots(figsize=(8,8))

def update(frame):
    ax.clear()

    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    alive_positions = np.array([drone.position for drone in alive_drones])
    distances = np.linalg.norm(alive_positions - target_position, axis=1)
    sorted_indices = np.argsort(distances)

    if len(sorted_indices) < 3:
        leaders = [alive_positions[i] for i in sorted_indices] + [alive_positions[sorted_indices[0]]] * (3 - len(sorted_indices))
    else:
        leaders = [alive_positions[sorted_indices[0]],
                   alive_positions[sorted_indices[1]],
                   alive_positions[sorted_indices[2]]]

    alpha_pos, beta_pos, delta_pos = leaders

    a = 2 - frame * (2 / max_steps)  # a değeri zamanla lineer azalıyor (saldırı artıyor)

    for drone in alive_drones:
        drone.move(alpha_pos, beta_pos, delta_pos, a)
        drone.check_impact()

    # Çizim
    for drone in drones:
        path = np.array(drone.path)
        if drone.alive:
            ax.plot(path[:,0], path[:,1], label=f"Drone {drone.id} (Alive)")
            ax.scatter(drone.position[0], drone.position[1], marker='o')
        else:
            ax.plot(path[:,0], path[:,1], '--', label=f"Drone {drone.id} (Destroyed)")
            ax.scatter(drone.position[0], drone.position[1], marker='x')

    ax.scatter(target_position[0], target_position[1], c='red', marker='*', s=200, label='Hedef')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_title(f'GWO Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
plt.show()
