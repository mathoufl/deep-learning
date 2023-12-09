import torch
import model


class Model(torch.nn.Module):
    """
    class représentant notre model sur lequel nous feront de l'apprentissage profond
        => permet d'inferer les action les plus efficaces à faire à partir d'un état donné
    """

    def __init__(self) :
        super().__init__()
        # we want groups = in_channel on the first conv layer ?
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
        



class BasicModel(torch.nn.Module):
    """
    class représentant notre model sur lequel nous feront de l'apprentissage profond
        => permet d'inferer les action les plus efficaces à faire à partir d'un état donné
    """

    def __init__(self) :
        super().__init__()
        # we want groups = in_channel on the first conv layer ?
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
        
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=2),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),

            torch.nn.Conv2d(8, 12, kernel_size=3, stride=1),
            torch.nn.AvgPool2d((2,2)),
            torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            
            torch.nn.Linear(72, 24),
            torch.nn.ReLU()
            )
        
        self.state_value = torch.nn.Linear(24, 1)
        self.advantage = torch.nn.Linear(24, 6)
    
    def forward(self, input) :
        state_encoding = self.conv_layers(input)

        state_value = self.state_value(state_encoding)
        advantage = self.advantage(state_encoding)
        q_value = state_value + (advantage - advantage.mean())
        return(state_value, advantage, q_value)
    
    
    
class Model_vgg(torch.nn.Module):
    def __init__(self) :
        super().__init__()
        # we want groups = in_channel on the first conv layer ?
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
        
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(2,2),
            torch.nn.ReLU(),
            
            # torch.nn.Conv2d(256, 512, kernel_size=3, stride=1),
            # torch.nn.Conv2d(512, 512, kernel_size=3, stride=1),
            # torch.nn.MaxPool2d(2,2),
            # torch.nn.ReLU(),
            
            torch.nn.Flatten(),
            
            torch.nn.Linear(8960, 64),
            torch.nn.ReLU(),
            
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
            )
        
        self.state_value = torch.nn.Linear(64, 1)
        self.advantage = torch.nn.Linear(64, 6)
    
    def forward(self, input) :
        state_encoding = self.conv_layers(input)

        state_value = self.state_value(state_encoding)
        advantage = self.advantage(state_encoding)
        q_value = state_value + (advantage - advantage.mean())
        return(state_value, advantage, q_value)
    
 