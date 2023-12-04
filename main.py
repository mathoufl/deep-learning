
from model import Model, BasicModel
from training import train_model
from utils import demoToMP4

#%%



# model = Model()
# model.to("cuda")

# epsilon_sch = (lambda t: 0.3 - t*( (0.4-0.1)/200. ) )

# # train_model(model,
# #             env_config_file="gamedata/training_maps/config_training.cfg",
# #             img_dim=(120,160),
# #             replay_buffer_capacity=10000,
# #             batch_size = 32,
# #             learning_rate =1e-4,
# #             weight_decay=5e-4,
# #             frame_skip=3,
# #             n_episodes=200,
# #             epsilon_schedule=epsilon_sch
# #             )
# train_model(model,
#             env_config_file="gamedata/training_maps/config_training_simplified.cfg",
#             img_dim=(120,160),
#             n_imagesByState = 1,
#             frame_skip=0,
#             action_signature=[2,3,3],
#             replay_buffer_capacity=10000,
#             batch_size = 32,
#             learning_rate =1e-4,
#             weight_decay=5e-4,
#             n_episodes=200,
#             epsilon_schedule=epsilon_sch
            # )
# learning rate of 1e-4 seems to low : the loss stay big even after many episodes

#%%




# path = r"C:\Users\Gabriel\Documents\_Documents\Etudes\3A 2\9.6.1 deep learning and RL\projet\training_reports\training report 2-12-2023 - 14h47min2s\episode_demos\episode#20.lmp"

# demoToMP4("video_demo.mp4", path, config_file="gamedata/training_maps/config_training.cfg")

#%%

model = BasicModel()
model.to("cuda")

epsilon_sch = (lambda t: 0.3 - t*( (0.4-0.1)/200. ) )

train_model(model,
            env_config_file="gamedata/training_maps/fire_and_dodge.cfg",
            img_dim=(120,160),
            n_imagesByState = 1,
            frame_skip=0,
            action_signature=[2,3],
            replay_buffer_capacity=10000,
            batch_size = 32,
            learning_rate =1e-3,
            weight_decay=5e-4,
            n_episodes=200,
            epsilon_schedule=epsilon_sch,
            window_visible = True,
            doom_map_list=["FIR_DODG"]
            )
