# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 03:50:01 2025

@author: emrek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Bat Algorithm (Yarasa Algoritması) tabanlı Kamikaze Drone Sürüsü

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.frequency = 0
        self.loudness = 1  # Ses şiddeti
        self.pulse_rate = 0  # Darbe hızı
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def fitness(self):
        distance = np.linalg.norm(self.position - self.target)
        return 1 / (distance + 1e-6)

    def move(self, global_best, freq_min=0, freq_max=1):
        if not self.alive:
            return

        # Frekansı güncelle
        self.frequency = freq_min + (freq_max - freq_min) * np.random.rand()

        # Hız ve pozisyon güncellemesi
        self.velocity += (self.position - global_best) * self.frequency
        new_position = self.position + self.velocity

        # Lokal arama (küçük rasgele adımlar)
        if np.random.rand() > self.pulse_rate:
            epsilon = np.random.uniform(-0.5, 0.5, size=2)
            new_position = global_best + epsilon

        # Yeni pozisyon daha iyi mi? (fitness kıyaslaması)
        if self._is_better(new_position):
            self.position = new_position
            self.loudness *= 0.9  # Ses şiddeti azalır
            self.pulse_rate = self.pulse_rate + 0.1 * (1 - self.pulse_rate)  # Darbe oranı artar

        self.path.append(self.position.copy())

    def _is_better(self, new_position):
        new_distance = np.linalg.norm(new_position - self.target)
        current_distance = np.linalg.norm(self.position - self.target)
        return new_distance < current_distance

    def check_impact(self, impact_radius=2):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")

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

    # Global en iyi: hedefe en yakın drone
    alive_positions = np.array([drone.position for drone in alive_drones])
    distances = np.linalg.norm(alive_positions - target_position, axis=1)
    best_index = np.argmin(distances)
    global_best = alive_positions[best_index]

    for drone in alive_drones:
        drone.move(global_best)
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
    ax.set_title(f'Bat Algorithm Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
plt.show()
