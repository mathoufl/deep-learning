

from model import Model
from training import train_model
from utils import demoToMP4

#%%



model = Model()
model.to("cuda")

epsilon_sch = (lambda t: 0.3 - t*( (0.4-0.05)/200. ) )

train_model(model,
            env_config_file="gamedata/training_maps/config_training.cfg",
            img_dim=(120,160),
            replay_buffer_capacity=10000,
            batch_size = 32,
            learning_rate =1e-4,
            weight_decay=5e-4,
            frame_skip=3,
            n_episodes=200,
            epsilon_schedule=epsilon_sch
            )

#%%




# path = r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\git_doom\training report 2023-12-01 21;43;53.104506\episode_demos\episode#74.lmp"

# demoToMP4("video_demo74.mp4", path, config_file="gamedata/training_maps/config_training.cfg")