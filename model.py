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
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1),
            torch.nn.AvgPool2d((4,4)),
            torch.nn.ReLU(),
        )

        self.flatten = torch.nn.Flatten()
        self.state_value = torch.nn.Linear(128, 1)
        self.advantage = torch.nn.Linear(128, 108)
    
    def forward(self, input) :
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flatten(output)
        state_value = self.state_value(output)
        advantage = self.advantage(output)
        return(state_value, advantage)
        

class Agent():

    def __init__(self):
        self.model = Model()



        