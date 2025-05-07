import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## İyileştirilmiş PSO ALGORİTMASI

class Drone:
    def __init__(self, position, target, drone_id):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.velocity = np.zeros(2)
        self.id = drone_id
        self.alive = True
        self.path = [self.position.copy()]

    def update_velocity(self, global_best, w=0.1, c1=2.0, c2=0.5):
        r1 = np.random.uniform(0.9, 1.1, 2)  # Daha az rastgelelik
        r2 = np.random.uniform(0.9, 1.1, 2)# r1 ve r2 rastgelelik katıyor pso da bulunan
        
        """PSO DAKİ PARAMETRELER NE İŞE YARAR ?"""
        """# Kendi hedefine ve global en iyiye odaklanarak hız güncellemesi"""
        """#c1 (cognitive coefficient - bireysel katsayı) = Kendi en iyi durumuna çekilme kuvveti == Drone'un kendi hedefi/amacı için uğraşması."""
        """#c2(social coefficient - sosyal katsayı) = Sürüdeki en iyi çözüme çekilme kuvveti = Drone'un toplu davranıp sürüdeki en iyi çözümden etkilenmesi."""
        """#w =Mevcut hızın etkisi =Droneların eski hızlarını ne kadar koruyacaklarını belirler."""
        """#global best = Tüm sürüdeki en iyi çözüm (hedefe en yakın pozisyon) ==  en yakın pozisyon)	Sürü genelinde "en iyiye" doğru toplu hareket sağlar."""
        
        
        personal_best = self.target
        pso_velocity = (w * self.velocity +
                        c1 * r1 * (personal_best - self.position) +
                        c2 * r2 * (global_best - self.position))

        # Hedefe doğru bir yönelme kuvveti ekliyoruz
        direction_to_target = self.target - self.position
        direction_to_target /= np.linalg.norm(direction_to_target)  # Normalize etmenin amacı dron hedefe giderken çok fazla zikzak çiziyor bunun nedeni ise 
        pso_velocity += direction_to_target * 0.5  # Hedefe daha çok odaklanma

        # Maksimum hız sınırı
        max_speed = 5
        speed = np.linalg.norm(pso_velocity)
        if speed > max_speed:
            pso_velocity = pso_velocity / speed * max_speed

        self.velocity = pso_velocity

    def move(self):
        if self.alive:
            self.position += self.velocity
            self.path.append(self.position.copy())

    def check_impact(self, impact_radius=4):# BURDA EĞERKİ AZALTIR İSEK PSO DAKİ HIZ GÜNCELLEME DEZAVANTAJI VE YÖNLENDİRME DEZAVANTAJINDAN ÖTÜRÜ HEDEF ÇEVRESİNDE SIKINTI OLUŞYUR
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

# Animasyon için veri
fig, ax = plt.subplots(figsize=(8,8))

def update(frame):
    ax.clear()

    alive_drones = [drone for drone in drones if drone.alive]

    if not alive_drones:
        return

    # Global en iyi: hedefe en yakın canlı drone
    alive_positions = np.array([drone.position for drone in alive_drones])
    distances = np.linalg.norm(alive_positions - target_position, axis=1)
    best_index = np.argmin(distances)
    global_best = alive_positions[best_index]

    for drone in alive_drones:
        drone.update_velocity(global_best)
        drone.move()
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
    ax.set_title(f'Kamikaze Drone Sürüsü - Adım {frame}')
    ax.legend()
    ax.grid()


ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=200, repeat=False)

# Animasyonu kaydet

plt.show()
