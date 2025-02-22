import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools

# 定数（天文単位・年）
a_earth = 1.0          # 地球の半長軸 (AU)
a_mars = 1.524         # 火星の半長軸 (AU)
T_earth = 1.0          # 地球の公転周期 (年)
T_mars = 1.88          # 火星の公転周期 (年)
omega_earth = 2 * np.pi / T_earth  # 地球の角速度 (rad/年)
omega_mars = 2 * np.pi / T_mars    # 火星の角速度 (rad/年)

# Hohmann 転移軌道パラメータ
a_transfer = (a_earth + a_mars) / 2      # 転移軌道の半長軸 (AU)
T_transfer = a_transfer**1.5             # 転移軌道の周期 (年)
t_transfer = T_transfer / 2              # 転移時間（半周期, 約0.7085年）
e = (a_mars - a_earth) / (a_earth + a_mars)  # 転移軌道の離心率 (約0.208)

# 出発条件：地球から火星へ出発する際、火星は出発時の地球から見て約44度先に位置する
phi_initial = np.pi - omega_mars * t_transfer  # 約0.775 rad

# 火星での待機時間（火星到着後の滞在期間）
t_wait = 455 / 365  # 約1.247年

# 1往復のミッション時間（アウトバウンド + 待機 + リターン）
T_total = 2 * t_transfer + t_wait  # 約2.67年

# 会合周期（地球と火星の相対位相が同じになる周期、約2.135年）
synodic_period = 2.135

# 発射時刻判定の許容幅（年）
launch_tolerance = 0.01

# 1フレームあたりの経過時間（年単位）
dt = 0.01  # 約3.65日/フレーム

# Kepler方程式をニュートン法で解く関数
def solve_kepler(M, e, tol=1e-6, max_iter=10):
    E = M  # 初期推定
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        if abs(f) < tol:
            break
        df = 1 - e * np.cos(E)
        E -= f / df
    return E

# 静的な軌道（背景描画用）
theta = np.linspace(0, 2 * np.pi, 100)
earth_orbit = [a_earth * np.cos(theta), a_earth * np.sin(theta), np.zeros_like(theta)]
mars_orbit = [a_mars * np.cos(theta), a_mars * np.sin(theta), np.zeros_like(theta)]

# 3Dプロットのセットアップ
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-a_mars - 0.5, a_mars + 0.5)
ax.set_ylim(-a_mars - 0.5, a_mars + 0.5)
ax.set_zlim(-0.1, 0.1)
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Earth ↔ Mars Continuous Round Trips')

# 背景に軌道と太陽をプロット
ax.plot(earth_orbit[0], earth_orbit[1], earth_orbit[2], 'b-', label='Earth Orbit')
ax.plot(mars_orbit[0], mars_orbit[1], mars_orbit[2], 'r-', label='Mars Orbit')
ax.scatter([0], [0], [0], color='yellow', s=100, label='Sun')

# 地球と火星の現在位置を示す点
earth_point, = ax.plot([], [], [], 'bo', ms=10, label='Earth')
mars_point, = ax.plot([], [], [], 'ro', ms=10, label='Mars')

# 複数ミッションの管理（各ミッションは辞書で管理）
missions = []  
# 次の出発予定時刻（初回は t=0）
next_departure_time = 0

# アニメーション用の時間表示（経過日数・years・打ち上げ宇宙船数・帰還数）
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def animate(frame):
    global next_departure_time

    # 現在時刻（年単位）はフレーム数に dt をかけて算出
    current_time = frame * dt

    # 地球の位置（グローバルに動く）
    theta_earth_global = omega_earth * current_time
    earth_pos_global = [a_earth * np.cos(theta_earth_global), a_earth * np.sin(theta_earth_global), 0]
    earth_point.set_data([earth_pos_global[0]], [earth_pos_global[1]])
    earth_point.set_3d_properties([earth_pos_global[2]])

    # 火星の位置（初期位相 phi_initial を加える）
    theta_mars_global = omega_mars * current_time + phi_initial
    mars_pos_global = [a_mars * np.cos(theta_mars_global), a_mars * np.sin(theta_mars_global), 0]
    mars_point.set_data([mars_pos_global[0]], [mars_pos_global[1]])
    mars_point.set_3d_properties([mars_pos_global[2]])

    # 新たなミッション開始：現在時刻と出発予定時刻との差が許容範囲内の場合のみ発射
    if abs(current_time - next_departure_time) < launch_tolerance:
        theta_earth_dep = omega_earth * next_departure_time
        theta_mars_dep = omega_mars * next_departure_time + phi_initial
        phase_diff = (theta_mars_dep - theta_earth_dep) % (2*np.pi)
        if abs(phase_diff - phi_initial) < 0.05:  # 許容範囲（ラジアン）
            # 新規ミッションを開始
            mission = {
                'departure': next_departure_time,
                'history': {'x': [], 'y': [], 'z': []},
                'artist': ax.plot([], [], [], 'go', ms=8)[0],
                'path_artist': ax.plot([], [], [], 'g--', lw=2)[0],
                'finished': False,
                'final_pos': None
            }
            missions.append(mission)
            next_departure_time += synodic_period

    # 各ミッションの更新
    for mission in missions:
        dep = mission['departure']
        if current_time < dep:
            continue  # ミッション開始前は更新しない

        tau = current_time - dep  # ミッション内経過時間

        if mission.get('finished', False):
            # リターン到着済みの場合は、点は非表示（軌跡は残す）
            pos = mission['final_pos']
            mission['artist'].set_data([], [])
            mission['artist'].set_3d_properties([])
        else:
            if tau <= t_transfer:
                # 【アウトバウンド転移】
                theta_dep = omega_earth * dep
                M = (np.pi / t_transfer) * tau
                E_val = solve_kepler(M, e)
                x0 = a_transfer * (np.cos(E_val) - e)
                y0 = a_transfer * np.sqrt(1 - e**2) * np.sin(E_val)
                x = x0 * np.cos(theta_dep) - y0 * np.sin(theta_dep)
                y = x0 * np.sin(theta_dep) + y0 * np.cos(theta_dep)
                pos = [x, y, 0]
            elif tau <= t_transfer + t_wait:
                # 【火星待機】：火星上に滞在
                pos = mars_pos_global
            elif tau <= T_total:
                # 【リターン転移】
                tau_ret = tau - (t_transfer + t_wait)
                alpha = omega_mars * (dep + t_transfer + t_wait) + phi_initial + np.pi
                M = np.pi + (np.pi / t_transfer) * tau_ret
                E_val = solve_kepler(M, e)
                x0 = a_transfer * (np.cos(E_val) - e)
                y0 = a_transfer * np.sqrt(1 - e**2) * np.sin(E_val)
                x = x0 * np.cos(alpha) - y0 * np.sin(alpha)
                y = x0 * np.sin(alpha) + y0 * np.cos(alpha)
                pos = [x, y, 0]
            else:
                # ミッション完了（リターン到着）：到着時の地球位置に固定し、点は非表示
                theta_arrival = omega_earth * (dep + T_total)
                pos = [a_earth * np.cos(theta_arrival), a_earth * np.sin(theta_arrival), 0]
                mission['finished'] = True
                mission['final_pos'] = pos
                mission['artist'].set_data([], [])
                mission['artist'].set_3d_properties([])

            # ミッション中は常に宇宙船の現在位置を示す緑の点を更新
            if not mission.get('finished', False):
                mission['artist'].set_data([pos[0]], [pos[1]])
                mission['artist'].set_3d_properties([pos[2]])

        # 軌跡の履歴更新
        mission['history']['x'].append(pos[0])
        mission['history']['y'].append(pos[1])
        mission['history']['z'].append(pos[2])
        max_history = 30  # 軌跡として保持する最大点数
        for key in ['x', 'y', 'z']:
            if len(mission['history'][key]) > max_history:
                mission['history'][key] = mission['history'][key][-max_history:]
        mission['path_artist'].set_data(mission['history']['x'], mission['history']['y'])
        mission['path_artist'].set_3d_properties(mission['history']['z'])

    # 打ち上げた宇宙船数と帰還した宇宙船数のカウント
    launched_count = len(missions)
    returned_count = sum(1 for mission in missions if mission.get('finished', False))
    
    # 経過日数、年数、打ち上げた宇宙船数、帰還数を表示
    time_text.set_text(
        f"Time: {current_time*365:.0f} days / {current_time:.2f} years\n"
        f"Launched: {launched_count} spacecraft\n"
        f"Returned: {returned_count} spacecraft"
    )
    
    artists = [earth_point, mars_point, time_text]
    for mission in missions:
        artists.append(mission['path_artist'])
        artists.append(mission['artist'])
    return artists

# 無限フレームのイテレータを使用して、時間が継続的に増加するようにする
anim = FuncAnimation(fig, animate, frames=itertools.count(), interval=50, blit=False)
ax.legend(loc='upper right')
plt.show()
