import numpy as np
import re
import cv2
import imageio
import vizdoom

"""
This function replay a demo file, and save the render into a mp4 video file
"""
def demoToMP4(video_name, demo_file, config_file, resolution=vizdoom.ScreenResolution.RES_640X480):
    
    
    r = re.findall(r'\d+', resolution.name)
    vid_res = int(r[0]), int(r[1])
    video = cv2.VideoWriter(video_name,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            35, vid_res, isColor = True)
    
    game = vizdoom.DoomGame()
    game.load_config( config_file )
    game.set_screen_format( vizdoom.ScreenFormat.CRCGCB )
    game.set_screen_resolution(resolution)
    game.set_window_visible(True)
    game.set_render_hud(True)
    game.init()
    game.replay_episode(demo_file)
    
    while not game.is_episode_finished():
        state = game.get_state()
        img = np.moveaxis(state.screen_buffer,0, 2)
        
        video.write( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        game.advance_action()
    game.close()
    video.release()
    cv2.destroyAllWindows()


def demoToGIF(GIF_name, demo_file, config_file, resolution=vizdoom.ScreenResolution.RES_640X480):
    r = re.findall(r'\d+', resolution.name)
    vid_res = int(r[0]), int(r[1])

    game = vizdoom.DoomGame()
    game.load_config( config_file )
    game.set_screen_format( vizdoom.ScreenFormat.CRCGCB )
    game.set_screen_resolution(resolution)
    game.set_window_visible(False)
    game.set_render_hud(True)
    game.init()
    game.replay_episode(demo_file)
    
    images = []
    while not game.is_episode_finished():
        state = game.get_state()
        img = np.moveaxis(state.screen_buffer,0, 2)
        images.append(img)
        game.advance_action()
    game.close()
    imageio.mimsave(GIF_name, images, duration = 1/35)


def convertActionIdToButtonsArray(action_id, action_space_signature = [2,2,3,3,3]):
    # la signature correspond au cardinal d'un composant du vecteur d'action
    # le vecteur d'action est définit tel que chaque composant soit indépendant
    # exemple tourner à droite ou à gauche = 1 composant
    
    # available_buttons =
    #       ATTACK -> 2
    
    # 		SPEED  -> 2
    
    # 		MOVE_RIGHT |
    # 		MOVE_LEFT  |-> 3
    
    # 		MOVE_BACKWARD |
    # 		MOVE_FORWARD  |-> 3
    
    # 		TURN_RIGHT |
    # 		TURN_LEFT  |-> 3
    
    # \-> resulat to [2,2,3,3,3] as a signature

    # basically this convert an integer to a mixed binary/ternary representation
    action = []
    a = action_id
    if action_id < 0 or action_id > np.prod(action_space_signature):
        print("invalid action_id")
        return None
    for componant_cardinal in action_space_signature:
        r = a % componant_cardinal
        a = a // componant_cardinal
        if componant_cardinal == 2:
            action.append(r)
            
        elif componant_cardinal ==3:
            if r == 0:
                action.extend([0,0])
            if r == 1:
                action.extend([1,0])
            if r == 2:
                action.extend([0,1])   
        else: # non supported case
            print("invalid action signature")
            return None
    return action

# "unit test" of the fucntion
# test_sign = [3,3,2]
# for i in range(len(test_sign)):
#     action = convertActionIdToButtonsArray(i, test_sign)
#     print("action",i, ":=>", action)
