import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants (in astronomical units and years)
a_earth = 1.0          # Earth's semi-major axis (AU)
a_mars = 1.524         # Mars' semi-major axis (AU)
T_earth = 1.0          # Earth's orbital period (years)
T_mars = 1.88          # Mars' orbital period (years)
omega_earth = 2 * np.pi / T_earth  # Earth's angular velocity (rad/year)
omega_mars = 2 * np.pi / T_mars    # Mars' angular velocity (rad/year)

# Hohmann transfer orbit parameters
a_transfer = (a_earth + a_mars) / 2  # Semi-major axis of transfer orbit (AU)
T_transfer = a_transfer**1.5          # Period of transfer orbit (years)
t_transfer = T_transfer / 2           # Transfer time (half period, ~0.7085 years)
e = (a_mars - a_earth) / (a_earth + a_mars)  # Eccentricity of transfer orbit (~0.208)

# Phase angles for Hohmann transfers
phi_initial = np.pi - omega_mars * t_transfer  # Mars ahead of Earth (~44 degrees)
phi_return = np.pi - omega_earth * t_transfer  # Mars ahead of Earth for return (~75 degrees)

# Calculate return launch time (t_return)
# Solve: (omega_earth - omega_mars) * t_return = phi_return - phi_initial + 2*pi*m
t_wait = 455 / 365  # Waiting period on Mars in years (~1.247 years)
t_return = t_transfer + t_wait
T_total = t_return + t_transfer  # Total mission time (~2.67 years)

# Animation parameters
N = 200  # Number of frames
t = np.linspace(0, T_total, N)  # Time array

# Function to solve Kepler's equation using Newton's method
def solve_kepler(M, e, tol=1e-6, max_iter=10):
    E = M  # Initial guess
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        if abs(f) < tol:
            break
        df = 1 - e * np.cos(E)
        E -= f / df
    return E

# Precompute static orbits
theta = np.linspace(0, 2 * np.pi, 100)
earth_orbit = [a_earth * np.cos(theta), a_earth * np.sin(theta), np.zeros_like(theta)]
mars_orbit = [a_mars * np.cos(theta), a_mars * np.sin(theta), np.zeros_like(theta)]

# Outbound transfer orbit (perihelion at Earth’s orbit, aphelion at Mars’ orbit)
E = np.linspace(0, 2 * np.pi, 100)
x_transfer_out = a_transfer * (np.cos(E) - e)
y_transfer_out = a_transfer * np.sqrt(1 - e**2) * np.sin(E)
transfer_orbit_outbound = [x_transfer_out, y_transfer_out, np.zeros_like(E)]

# Return transfer orbit (aphelion at Mars’ orbit, perihelion at Earth’s orbit)
alpha = omega_mars * t_return + phi_initial  # Orientation at return launch
x_transfer_ret = (a_transfer * (np.cos(E) - e) * np.cos(alpha) -
                  a_transfer * np.sqrt(1 - e**2) * np.sin(E) * np.sin(alpha))
y_transfer_ret = (a_transfer * (np.cos(E) - e) * np.sin(alpha) +
                  a_transfer * np.sqrt(1 - e**2) * np.sin(E) * np.cos(alpha))
transfer_orbit_return = [x_transfer_ret, y_transfer_ret, np.zeros_like(E)]

# Set up the 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-a_mars - 0.5, a_mars + 0.5)
ax.set_ylim(-a_mars - 0.5, a_mars + 0.5)
ax.set_zlim(-0.1, 0.1)
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Earth to Mars Round Trip')

# Plot static orbits and Sun
ax.plot(earth_orbit[0], earth_orbit[1], earth_orbit[2], 'b-', label='Earth Orbit')
ax.plot(mars_orbit[0], mars_orbit[1], mars_orbit[2], 'r-', label='Mars Orbit')
ax.scatter([0], [0], [0], color='yellow', s=100, label='Sun')

# Initialize dynamic points
earth_point, = ax.plot([], [], [], 'bo', ms=10, label='Earth')
mars_point, = ax.plot([], [], [], 'ro', ms=10, label='Mars')
spacecraft_point, = ax.plot([], [], [], 'go', ms=10, label='Spacecraft')
# Initialize spacecraft trajectory history
spacecraft_history = {'x': [], 'y': [], 'z': []}
spacecraft_path, = ax.plot([], [], [], 'g--', lw=2, label='Spacecraft Path')
max_history = 150  # Maximum number of points retained in spacecraft trajectory history
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Animation update function
def animate(i):
    # Earth and Mars positions
    theta_earth = omega_earth * t[i]
    earth_pos = [a_earth * np.cos(theta_earth), a_earth * np.sin(theta_earth), 0]
    theta_mars = omega_mars * t[i] + phi_initial
    mars_pos = [a_mars * np.cos(theta_mars), a_mars * np.sin(theta_mars), 0]

    # Spacecraft position
    if t[i] <= t_transfer:  # Outbound transfer
        M = (np.pi / t_transfer) * t[i]
        E = solve_kepler(M, e)
        x = a_transfer * (np.cos(E) - e)
        y = a_transfer * np.sqrt(1 - e**2) * np.sin(E)
        spacecraft_pos = [x, y, 0]
    elif t[i] <= t_return:  # On Mars
        spacecraft_pos = mars_pos
    else:  # Return transfer
        M = np.pi + (np.pi / t_transfer) * (t[i] - t_return)
        E = solve_kepler(M, e)
        x_orbit = a_transfer * (np.cos(E) - e)
        y_orbit = a_transfer * np.sqrt(1 - e**2) * np.sin(E)
        alpha = omega_mars * t_return + phi_initial + np.pi
        x = x_orbit * np.cos(alpha) - y_orbit * np.sin(alpha)
        y = x_orbit * np.sin(alpha) + y_orbit * np.cos(alpha)
        spacecraft_pos = [x, y, 0]

    # Update points
    earth_point.set_data([earth_pos[0]], [earth_pos[1]])
    earth_point.set_3d_properties([earth_pos[2]])
    mars_point.set_data([mars_pos[0]], [mars_pos[1]])
    mars_point.set_3d_properties([mars_pos[2]])
    spacecraft_point.set_data([spacecraft_pos[0]], [spacecraft_pos[1]])
    spacecraft_point.set_3d_properties([spacecraft_pos[2]])
    # Append current spacecraft position to trajectory history
    spacecraft_history['x'].append(spacecraft_pos[0])
    spacecraft_history['y'].append(spacecraft_pos[1])
    spacecraft_history['z'].append(spacecraft_pos[2])
    # Limit history to last max_history points
    if len(spacecraft_history['x']) > max_history:
        spacecraft_history['x'] = spacecraft_history['x'][-max_history:]
        spacecraft_history['y'] = spacecraft_history['y'][-max_history:]
        spacecraft_history['z'] = spacecraft_history['z'][-max_history:]
    spacecraft_path.set_data(spacecraft_history['x'], spacecraft_history['y'])
    spacecraft_path.set_3d_properties(spacecraft_history['z'])
    # Update elapsed time display
    time_text.set_text(f"Time: {t[i]*365:.0f} days")
    return earth_point, mars_point, spacecraft_point, spacecraft_path, time_text

# Create and display the animation
anim = FuncAnimation(fig, animate, frames=N, interval=50, blit=False)
ax.legend(loc='upper right')
plt.show()