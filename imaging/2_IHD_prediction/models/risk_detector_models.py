import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class efficientNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        if 'pretrained' in config.keys():
            pretrained = config['pretrained']
        else:
            pretrained = True
        if 'outFeatures' in config.keys():
            outFeatures = config['outFeatures']
        else:
            outFeatures = 2
        if 'frozenweights' in config.keys():
            self.frozenWeights = config['frozenweights']
        else:
            self.frozenWeights = False
        if 'efficientName' in config.keys():
            self.name = config['efficientName']
        else:
            self.name = 'efficientnet-b6'
        if not pretrained:
            self.model_conv = EfficientNet.from_name(self.name)
            # raise NotImplementedError('You are trying to start a non-pretrained efficientnet model.')
        self.model_conv = EfficientNet.from_pretrained(self.name, num_classes=outFeatures)

    def sendToDevice(self, device):
        self.model_conv = self.model_conv.to(device)

    def forward(self,x):
        return self.model_conv(x)    

    def trainableParams(self):
        if self.frozenWeights == True:
            return self.model_conv._fc.parameters()
        elif self.frozenWeights == False:
            return self.model_conv.parameters()

    def differentialTrainableParams(self):
        return [x for x in self.model_conv.parameters()][:582], self.model_conv._fc.parameters()
