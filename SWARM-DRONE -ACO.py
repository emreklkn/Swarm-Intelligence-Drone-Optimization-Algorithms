# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 03:43:32 2025

@author: emrek
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## ACO (Ant Colony Optimization) tabanlı Kamikaze Drone Sürüsü

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def move(self, pheromone_map, alpha=1.0, beta=5.0, q0=0.7, step_size=2.0):
        if not self.alive:
            return

        directions = np.array([
            [1,0], [-1,0], [0,1], [0,-1],   # Sağ, Sol, Yukarı, Aşağı
            [1,1], [1,-1], [-1,1], [-1,-1]  # Çaprazlar
        ])

        desirability = []
        for d in directions:
            next_pos = self.position + d * step_size
            distance_to_target = np.linalg.norm(next_pos - self.target)
            pheromone = pheromone_map.get(tuple(np.round(next_pos)), 0.1)

            # Seçim olasılığı: feromon * görünürlük
            desirability.append((pheromone ** alpha) * ((1.0 / (distance_to_target + 1e-6)) ** beta))

        desirability = np.array(desirability)
        desirability_sum = np.sum(desirability)

        # Greedy veya olasılıksal seçim (q0 kullanımı)
        if np.random.rand() < q0:
            idx = np.argmax(desirability)
        else:
            probabilities = desirability / desirability_sum
            idx = np.random.choice(len(directions), p=probabilities)

        chosen_direction = directions[idx]
        self.position += chosen_direction * step_size
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

# Feromon haritası
pheromone_map = {}

# Simülasyon parametreleri
max_steps = 100
evaporation_rate = 0.05  # Feromon buharlaşması
pheromone_deposit = 10.0 # Hedefe yaklaşıldıkça bırakılan feromon

# Animasyon için veri
fig, ax = plt.subplots(figsize=(8,8))

def update(frame):
    ax.clear()

    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    for drone in alive_drones:
        drone.move(pheromone_map)
        impact = drone.check_impact()
        if impact:
            pos_key = tuple(np.round(drone.position))
            pheromone_map[pos_key] = pheromone_map.get(pos_key, 0.1) + pheromone_deposit

    # Feromon buharlaşması
    keys_to_update = list(pheromone_map.keys())
    for key in keys_to_update:
        pheromone_map[key] *= (1.0 - evaporation_rate)
        if pheromone_map[key] < 0.01:
            del pheromone_map[key]

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
    ax.set_title(f'ACO Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()

ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)
plt.show()
