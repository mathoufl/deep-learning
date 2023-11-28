import torch

class Model (torch.nn.Module):
    """
    class représentant notre model sur lequel nous feront de l'apprentissage profond
        => permet d'inferer les action les plus efficaces à faire à partir d'un état donné
    """

    def __init__(self) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=2), 
            # plusieur images d'entrées me permetent de prendre en compte le déplacement des mobs
        self.con2 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1)
        torch.nn.Linear(8, D_out)

class Agent():

    def __init__(self):
        self.model = Model()



        