shoot = [1, 0, 0, 0, 0, 0, 0, 0]
sprint = [0, 1, 0, 0, 0, 0, 0, 0]
left = [0, 0, 0, 1, 0, 0, 0, 0]
right = [0, 0, 1, 0, 0, 0, 0, 0]
forward = [0, 0, 0, 0, 0, 1, 0, 0]
backward = [0, 0, 0, 0, 1, 0, 0, 0]
turn_left = [0, 0, 0, 0, 0, 0, 0, 1]
turn_right = [0, 0, 0, 0, 0, 0, 1, 0]

actions = [
    shoot,
    sprint,
    right,
    left,
    backward,
    forward,
    turn_right,
    turn_left
]

available_actions_count = len(actions)