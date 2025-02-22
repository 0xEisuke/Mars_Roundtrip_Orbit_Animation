import io
import itertools
from flask import Flask, Response, render_template_string
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非対話型バックエンド
import matplotlib.pyplot as plt

# ------------------------------
# シミュレーション用定数・グローバル変数
# ------------------------------
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

# 出発条件：出発時に火星は地球から見て約44°先（約0.775 rad）
phi_initial = np.pi - omega_mars * t_transfer

# 火星での待機時間（約1.247年）
t_wait = 455 / 365

# 1往復のミッション時間（アウトバウンド＋待機＋リターン）
T_total = 2 * t_transfer + t_wait

# 会合周期（約2.135年）
synodic_period = 2.135

# 発射時刻判定の許容幅（年）
launch_tolerance = 0.01

# 1フレームあたりの経過時間（年、約3.65日/フレーム）
dt = 0.01

missions = []          # 各ミッションは辞書で管理
next_departure_time = 0  # 次の出発予定時刻

# ------------------------------
# 補助関数
# ------------------------------
def solve_kepler(M, e, tol=1e-6, max_iter=10):
    E = M
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        if abs(f) < tol:
            break
        df = 1 - e * np.cos(E)
        E -= f / df
    return E

# ------------------------------
# matplotlib 図のセットアップ
# ------------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-a_mars - 0.5, a_mars + 0.5)
ax.set_ylim(-a_mars - 0.5, a_mars + 0.5)
ax.set_zlim(-0.1, 0.1)
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Earth ↔ Mars Continuous Round Trips')

theta = np.linspace(0, 2*np.pi, 100)
earth_orbit = [a_earth * np.cos(theta), a_earth * np.sin(theta), np.zeros_like(theta)]
mars_orbit  = [a_mars  * np.cos(theta), a_mars  * np.sin(theta), np.zeros_like(theta)]
ax.plot(earth_orbit[0], earth_orbit[1], earth_orbit[2], 'b-', label='Earth Orbit')
ax.plot(mars_orbit[0],  mars_orbit[1],  mars_orbit[2],  'r-', label='Mars Orbit')
ax.scatter([0], [0], [0], color='yellow', s=100, label='Sun')

earth_point, = ax.plot([], [], [], 'bo', ms=10, label='Earth')
mars_point,  = ax.plot([], [], [], 'ro', ms=10, label='Mars')

# ------------------------------
# アニメーション更新関数
# ------------------------------
def animate(frame):
    global next_departure_time, missions
    current_time = frame * dt  # 現在時刻（年）

    # 地球の位置更新
    theta_earth_global = omega_earth * current_time
    earth_pos_global = [a_earth * np.cos(theta_earth_global),
                        a_earth * np.sin(theta_earth_global), 0]
    earth_point.set_data([earth_pos_global[0]], [earth_pos_global[1]])
    earth_point.set_3d_properties([earth_pos_global[2]])

    # 火星の位置更新（初期位相 phi_initial を加える）
    theta_mars_global = omega_mars * current_time + phi_initial
    mars_pos_global = [a_mars * np.cos(theta_mars_global),
                       a_mars * np.sin(theta_mars_global), 0]
    mars_point.set_data([mars_pos_global[0]], [mars_pos_global[1]])
    mars_point.set_3d_properties([mars_pos_global[2]])

    # 新規ミッション発射判定
    if abs(current_time - next_departure_time) < launch_tolerance:
        theta_earth_dep = omega_earth * next_departure_time
        theta_mars_dep = omega_mars * next_departure_time + phi_initial
        phase_diff = (theta_mars_dep - theta_earth_dep) % (2*np.pi)
        if abs(phase_diff - phi_initial) < 0.05:
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
            continue

        tau = current_time - dep
        if mission.get('finished', False):
            pos = mission['final_pos']
            mission['artist'].set_data([], [])
            mission['artist'].set_3d_properties([])
        else:
            if tau <= t_transfer:
                theta_dep = omega_earth * dep
                M = (np.pi / t_transfer) * tau
                E_val = solve_kepler(M, e)
                x0 = a_transfer * (np.cos(E_val) - e)
                y0 = a_transfer * np.sqrt(1 - e**2) * np.sin(E_val)
                x = x0 * np.cos(theta_dep) - y0 * np.sin(theta_dep)
                y = x0 * np.sin(theta_dep) + y0 * np.cos(theta_dep)
                pos = [x, y, 0]
            elif tau <= t_transfer + t_wait:
                pos = mars_pos_global
            elif tau <= T_total:
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
                theta_arrival = omega_earth * (dep + T_total)
                pos = [a_earth * np.cos(theta_arrival), a_earth * np.sin(theta_arrival), 0]
                mission['finished'] = True
                mission['final_pos'] = pos
                mission['artist'].set_data([], [])
                mission['artist'].set_3d_properties([])

            if not mission.get('finished', False):
                mission['artist'].set_data([pos[0]], [pos[1]])
                mission['artist'].set_3d_properties([pos[2]])

        mission['history']['x'].append(pos[0])
        mission['history']['y'].append(pos[1])
        mission['history']['z'].append(pos[2])
        max_history = 150
        for key in ['x', 'y', 'z']:
            if len(mission['history'][key]) > max_history:
                mission['history'][key] = mission['history'][key][-max_history:]
        mission['path_artist'].set_data(mission['history']['x'], mission['history']['y'])
        mission['path_artist'].set_3d_properties(mission['history']['z'])

    launched_count = len(missions)
    returned_count = sum(1 for mission in missions if mission.get('finished', False))
    display_text = (
        f"Time: {current_time*365:.0f} days / {current_time:.2f} years\n"
        f"Launched: {launched_count} spacecraft\n"
        f"Returned: {returned_count} spacecraft"
    )
    if hasattr(animate, "time_text"):
        animate.time_text.set_text(display_text)
    else:
        animate.time_text = ax.text2D(0.05, 0.95, display_text, transform=ax.transAxes)
    return []

# ------------------------------
# Flaskアプリのセットアップ
# ------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    # Tailwind CSS を用いたモダンでスタイリッシュな UI
    return render_template_string('''
    <!doctype html>
    <html lang="ja">
      <head>
        <meta charset="utf-8">
        <title>宇宙ミッション シミュレーション</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <!-- Tailwind CSS CDN -->
        <script src="https://cdn.tailwindcss.com"></script>
      </head>
      <body class="bg-gray-900 text-gray-100">
        <header class="bg-gradient-to-r from-indigo-600 to-blue-500 py-8 shadow-lg">
          <div class="container mx-auto px-4">
            <h1 class="text-4xl md:text-5xl font-extrabold text-center">宇宙ミッション シミュレーション</h1>
            <p class="mt-4 text-center text-lg md:text-xl">地球と火星の往復ミッションをリアルタイムで体感</p>
          </div>
        </header>
        <main class="container mx-auto px-4 py-12">
          <div class="flex justify-center">
            <div class="w-full sm:w-3/4 md:w-2/3 lg:w-1/2 bg-white rounded-xl shadow-2xl overflow-hidden">
              <img src="{{ url_for('video_feed') }}" class="w-full" alt="Simulation">
            </div>
          </div>
          <div class="mt-8 text-center">
            <p class="text-xl">シミュレーションの進捗情報は上部のグラフにてご確認ください。</p>
          </div>
        </main>
        <footer class="bg-gray-800 py-4">
          <div class="container mx-auto text-center text-sm text-gray-400">
            &copy; 2025 宇宙ミッションシミュレーション. All rights reserved.
          </div>
        </footer>
      </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    for frame in itertools.count():
        animate(frame)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        frame_data = buf.read()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        buf.close()

if __name__ == '__main__':
    app.run(debug=True)
