# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 03:48:32 2025

@author: emrek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## ABC (Artificial Bee Colony) tabanlı Kamikaze Drone Sürüsü

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]
        self.trial_counter = 0  # Başarısız deneme sayacı

    def fitness(self):
        # Hedefe ne kadar yakınsa fitness o kadar iyi
        distance = np.linalg.norm(self.position - self.target)
        return 1 / (distance + 1e-6)

    def move_towards(self, other_drone, phi_range=0.1):
        if not self.alive:
            return

        phi = np.random.uniform(-phi_range, phi_range, size=2)
        new_position = self.position + phi * (self.position - other_drone.position)

        # Konum güncellemesi
        if np.linalg.norm(new_position - self.target) < np.linalg.norm(self.position - self.target):
            self.position = new_position
            self.trial_counter = 0
        else:
            self.trial_counter += 1

        self.path.append(self.position.copy())

    def scout(self, area_size=50):
        # Yeni rastgele pozisyon ata
        self.position = np.random.rand(2) * area_size
        self.trial_counter = 0
        self.path.append(self.position.copy())

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
max_steps = 100
limit = 5  # Deneme limiti

# Animasyon için veri
fig, ax = plt.subplots(figsize=(8,8))

def update(frame):
    ax.clear()

    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    fitnesses = np.array([drone.fitness() for drone in alive_drones])
    probs = fitnesses / fitnesses.sum()

    # İşçi arılar: mevcut çözümü iyileştir
    for drone in alive_drones:
        partner = np.random.choice(alive_drones)
        drone.move_towards(partner)

    # Gözlemci arılar: en iyi çözümleri takip et
    for drone in alive_drones:
        if np.random.rand() < probs[drones.index(drone) % len(probs)]:
            partner = np.random.choice(alive_drones)
            drone.move_towards(partner)

    # Kaşif arılar: başarısız drone'u yeniden başlat
    for drone in alive_drones:
        if drone.trial_counter > limit:
            drone.scout()

    for drone in alive_drones:
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
    ax.set_title(f'ABC Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
plt.show()
