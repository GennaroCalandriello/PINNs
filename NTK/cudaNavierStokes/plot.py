import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dim = 300
data = np.loadtxt("snapshots.txt", delimiter=",")
N_space, M_times = data.shape
print("N_space:", N_space, "M_times:", M_times)
ux = data[:dim*dim, :]
uy = data[dim*dim:2*dim*dim, :]
ux_2d = ux.reshape(dim, dim, M_times)
uy_2d = uy.reshape(dim, dim, M_times)
umag = np.sqrt(ux_2d**2 + uy_2d**2)
# Animation of u_x evolution

# Animation in una NUOVA figura
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ux_2d[..., 0], origin='lower', cmap='jet')
ax.set_title("u_x evolution")
fig.colorbar(im, ax=ax)

ux_2d = umag  # Use uy_2d for the animation
def update(frame):
    im.set_data(ux_2d[..., frame])
    im.set_clim(np.min(ux_2d[..., frame]), np.max(ux_2d[..., frame]))
    ax.set_title(f"u_x, frame {frame}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=M_times, interval=50, blit=True)
plt.show()