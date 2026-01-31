import subprocess
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ----------------------
# Parameters
# ----------------------
num_particles = 1024  # match your CUDA run
iterations = 50        # number of timesteps

executable = './nbody_sim'
csv_naive = 'naive.csv'
csv_tiled = 'tiled.csv'
modes = ['naive', 'tiled']

# ----------------------
# Helper function: run simulation and save CSV
# ----------------------
def run_simulation(mode, output_csv):
    print(f'Running {mode} simulation...')
    start_time = time.time()
    with open(output_csv, 'w') as f:
        proc = subprocess.run([executable, '-n', str(num_particles), '-i', str(iterations), '-m', mode], stdout=f, text=True)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f'{mode} done in {exec_time:.2f} seconds.')
    return exec_time

# ----------------------
# Run both simulations
# ----------------------
exec_times = {}
for mode, csv_file in zip(modes, [csv_naive, csv_tiled]):
    exec_times[mode] = run_simulation(mode, csv_file)

# ----------------------
# Load CSV data into NumPy arrays
# ----------------------
def load_csv(filename):
    data = np.loadtxt(filename, delimiter=',')
    data = data.reshape((iterations, num_particles, 3))
    return data

data_naive = load_csv(csv_naive)
data_tiled = load_csv(csv_tiled)

# ----------------------
# Create synchronized Plotly 3D animation
# ----------------------
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]],
    subplot_titles=(
        f"Naive Simulation\nTime: {exec_times['naive']:.2f}s",
        f"Tiled Simulation\nTime: {exec_times['tiled']:.2f}s"
    )
)

# Initial frame (iteration 0)
x_naive, y_naive, z_naive = data_naive[0].T
x_tiled, y_tiled, z_tiled = data_tiled[0].T

scatter_naive = go.Scatter3d(x=x_naive, y=y_naive, z=z_naive, mode='markers', showlegend=False, marker=dict(size=3, color='blue'))
scatter_tiled = go.Scatter3d(x=x_tiled, y=y_tiled, z=z_tiled, mode='markers', showlegend=False, marker=dict(size=3, color='red'))

fig.add_trace(scatter_naive, row=1, col=1)
fig.add_trace(scatter_tiled, row=1, col=2)

# Frames for animation
frames = []
for i in range(iterations):
    frame = go.Frame(data=[
        go.Scatter3d(x=data_naive[i,:,0], y=data_naive[i,:,1], z=data_naive[i,:,2], showlegend=False),
        go.Scatter3d(x=data_tiled[i,:,0], y=data_tiled[i,:,1], z=data_tiled[i,:,2], showlegend=False)
    ], name=f'{i}')
    frames.append(frame)

fig.frames = frames

# Slider
slider = [
    dict(
        steps=[dict(method='animate', args=[[f'{i}'], dict(mode='immediate', frame=dict(duration=200, redraw=True), transition=dict(duration=0))], label=str(i)) for i in range(iterations)],
        transition=dict(duration=0),
        x=0, y=0, currentvalue=dict(font=dict(size=12), prefix='Iteration: ', visible=True),
        len=1.0
    )
]

# Layout
fig.update_layout(
    height=600, width=1200,
    updatemenus=[
        dict(type='buttons', showactive=False, y=1.05, x=1.05, xanchor='right', yanchor='top',
             buttons=[
                 dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True, mode='immediate', transition=dict(duration=0), loop=True)]),
                 dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
             ])
    ],
    sliders=slider
)

# Axes ranges (auto-scale based on all data)
xmin = min(data_naive[:,:,0].min(), data_tiled[:,:,0].min())
xmax = max(data_naive[:,:,0].max(), data_tiled[:,:,0].max())
ymin = min(data_naive[:,:,1].min(), data_tiled[:,:,1].min())
ymax = max(data_naive[:,:,1].max(), data_tiled[:,:,1].max())
zmin = min(data_naive[:,:,2].min(), data_tiled[:,:,2].min())
zmax = max(data_naive[:,:,2].max(), data_tiled[:,:,2].max())

for trace in fig.data:
    trace.update(marker=dict(size=3))
fig.update_scenes(xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]), zaxis=dict(range=[zmin, zmax]))

# ----------------------
# Show in browser
# ----------------------
fig.show()

# ----------------------
# Attempt GIF export gracefully
# ----------------------
fig.write_html('nbody_animation.html')  # always save interactive HTML
try:
    import kaleido
    if 'PLOTLY_CHROME_BIN' in os.environ:
        fig.write_image('nbody_animation.gif', engine='kaleido')
        print('GIF exported as nbody_animation.gif')
    else:
        print('GIF export skipped: PLOTLY_CHROME_BIN not set. Interactive HTML saved as fallback.')
except Exception as e:
    print('GIF export failed, saved HTML instead:', e)
