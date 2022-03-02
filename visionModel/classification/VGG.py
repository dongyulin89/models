import torch
import torch.nn as nn

class VGG(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 1000,
    ) -> None:
        super(VGG, self).__init__()
        
        self.features_vgg11 = nn.Sequential(
            nn.conv2d(3, 64, kernel_size=3, padding=1), 
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BarchNorm2d(256),
            nn.ReLU(inpalce=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            
            nn.Linear(4096, num_classes), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_vgg11(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
