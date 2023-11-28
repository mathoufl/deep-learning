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
    game.set_window_visible(False)
    game.set_render_hud(True)
    game.init()
    game.replay_episode(demo_file)
    
    while not game.is_episode_finished():
        state = game.get_state()
        img = np.moveaxis(state.screen_buffer,0, 2)
        
        video.write( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        game.advance_action(1)
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
        game.advance_action(1)
    game.close()
    imageio.mimsave(GIF_name, images, duration = 1/35)
