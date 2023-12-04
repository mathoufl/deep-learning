import torch
import model

class Model(torch.nn.Module):
    """
    class représentant notre model sur lequel nous feront de l'apprentissage profond
        => permet d'inferer les action les plus efficaces à faire à partir d'un état donné
    """

    def __init__(self) :
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
        
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),

            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
            
            torch.nn.Flatten()
            )
        
        self.state_value = torch.nn.Linear(96, 1)
        self.advantage = torch.nn.Linear(96, 18)
    
    def forward(self, input) :
        state_encoding = self.conv_layers(input)

        state_value = self.state_value(state_encoding)
        advantage = self.advantage(state_encoding)
        q_value = state_value + (advantage - advantage.mean())
        return(state_value, advantage, q_value)
        

class Agent():

    def __init__(self):
        self.model = Model()



        