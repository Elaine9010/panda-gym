import gymnasium as gym
import numpy as np
import panda_gym
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def show_images(observation):
    if observation.get("rgb") is None or observation.get("depth") is None:
        return
    color, depth = observation["rgb"], observation["depth"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(color)
    axes[0].set_title("Color")
    im = axes[1].imshow(depth)
    axes[1].set_title("Depth")

    # Add colorbar to the second subplot
    cbar = fig.colorbar(im, ax=axes[1])

    # Create a FigureCanvasAgg instance
    canvas = FigureCanvasAgg(fig)

    # Render the figure to a NumPy array
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    return image


# initialize cv2.window
cv2.namedWindow("RGBAndDepth")


env = gym.make(
    "PandaPush-v3",
    render_mode="human",
    render_target_position=[0.0, 0.0, 0.0],
    render_distance=1.0,
    render_yaw=45,
    render_pitch=-30,
    render_roll=0,
)

observation, info = env.reset()
cv2.imshow("RGBAndDepth", show_images(observation))

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    cv2.imshow("RGBAndDepth", show_images(observation))

    if terminated or truncated:
        observation, info = env.reset()
        cv2.imshow("RGBAndDepth", show_images(observation))

    if cv2.waitKey(1) & 0xFF == ord("q"):  # wait for 1 millisecond
        break


env.close()
