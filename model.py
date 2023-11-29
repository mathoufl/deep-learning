import torch
import model

class Model(torch.nn.Module):
    """
    class représentant notre model sur lequel nous feront de l'apprentissage profond
        => permet d'inferer les action les plus efficaces à faire à partir d'un état donné
    """

    def __init__(self) :
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d(2,2),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1),
            torch.nn.AvgPool2d(2,2),
            torch.nn.ReLU(),
        )

        self.flatten = torch.nn.Flatten()
        self.state = torch.nn.Linear(96, 1)
        self.advantage = torch.nn.Linear(96, model.available_actions_count)
    
    def forward(self, input) :
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.flatten(output)
        print(output.size())

class Agent():

    def __init__(self):
        self.model = Model()



        