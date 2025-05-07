# -*- coding: utf-8 -*-
"""
Created on Mon May  5 01:59:35 2025

@author: emrek
"""

# -*- coding: utf-8 -*-
"""
LGPSO Drone Simülasyonu
@author: emrek
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Drone sınıfı
class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.velocity = np.zeros(2)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def update_velocity(self, global_best, step, max_steps):
        # Rastgele atalet ağırlığı
        w = 0.4 + 0.5 * np.random.rand()

        # Dinamik c1 ve c2 katsayıları (başta keşif, sonra sömürü)
        c1_start, c1_end = 2.5, 1.0
        c2_start, c2_end = 0.5, 2.5
        c1 = c1_start + (c1_end - c1_start) * (step / max_steps)
        c2 = c2_start + (c2_end - c2_start) * (step / max_steps)

        # Rastgelelik
        r1 = np.random.uniform(0.9, 1.1, 2)
        r2 = np.random.uniform(0.9, 1.1, 2)

        # Kişisel en iyi (hedef)
        personal_best = self.target

        # Yeni hız hesaplama
        new_velocity = (
            w * self.velocity +
            c1 * r1 * (personal_best - self.position) +
            c2 * r2 * (global_best - self.position)
        )

        # Hedefe yönelme kuvveti (sezgisel destek)
        to_target = self.target - self.position
        distance = np.linalg.norm(to_target)
        if distance > 1e-6:
            direction = to_target / distance
            new_velocity += direction * 0.3

        # Hız sınırı
        max_speed = 5
        speed = np.linalg.norm(new_velocity)
        if speed > max_speed:
            new_velocity = new_velocity / speed * max_speed

        # Greedy güncelleme
        new_position = self.position + new_velocity
        if np.linalg.norm(new_position - self.target) < np.linalg.norm(self.position - self.target):
            self.velocity = new_velocity

    def move(self):
        if self.alive:
            self.position += self.velocity
            self.path.append(self.position.copy())

    def check_impact(self, impact_radius=4):
        if np.linalg.norm(self.position - self.target) < impact_radius:
            self.alive = False
            print(f"Drone {self.id} hedefi imha etti!")

# Simülasyon parametreleri
num_drones = 30
max_steps = 100
target_position = np.array([100, 100])
drones = [Drone(position=np.random.rand(2) * 50, target=target_position, drone_id=i) for i in range(num_drones)]

# Görselleştirme için ayarlar
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.clear()
    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    # Global en iyi konumu belirle
    alive_positions = np.array([drone.position for drone in alive_drones])
    distances = np.linalg.norm(alive_positions - target_position, axis=1)
    global_best = alive_positions[np.argmin(distances)]

    # Drone'ları güncelle
    for drone in alive_drones:
        drone.update_velocity(global_best, step=frame, max_steps=max_steps)
        drone.move()
        drone.check_impact()

    # Çizim
    for drone in drones:
        path = np.array(drone.path)
        if drone.alive:
            ax.plot(path[:, 0], path[:, 1], label=f"Drone {drone.id} (Alive)")
            ax.scatter(drone.position[0], drone.position[1], marker='o')
        else:
            ax.plot(path[:, 0], path[:, 1], '--', label=f"Drone {drone.id} (Destroyed)")
            ax.scatter(drone.position[0], drone.position[1], marker='x')

    # Hedefi göster
    ax.scatter(target_position[0], target_position[1], c='red', marker='*', s=200, label='Hedef')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_title(f'LGPSO Drone Simülasyonu - Adım {frame}')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)

# Animasyonu başlat
ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)




plt.show()
