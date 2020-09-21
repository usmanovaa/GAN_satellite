import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.content_loss = nn.L1Loss()
        self.text_loss = TextureLoss()

    def forward(self, out_labels, out_images, target_images, adversarial_loss):        
        # Perception Loss 
        perception_loss = self.content_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.content_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        
        # Texture Loss
        text_loss = self.text_loss(out_images, target_images)

        #Cycle consistency loss
        size = target_images.size()[2]
        lr_real = F.interpolate(target_images, size// 4)
        lr_generated = F.interpolate(out_images, size// 4)
        cycle_loss = self.mse_loss(lr_real, lr_generated)
      
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 1e-5*cycle_loss +  1e-2*text_loss


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def criterion(a, b):
    return torch.mean(torch.abs((a-b)**2).view(-1))


class TextureLoss(nn.Module):
    def __init__(self, layers = [8, 17, 26, 35], replace_pooling = False):
        super(TextureLoss, self).__init__()
        self.layers = layers
        self.model = vgg19(pretrained=True).features
        
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        k = 0
        text_loss = 0

        for name, layer in enumerate(self.model):
            if k == len(self.layers):
                break
            sr = layer(sr)
            hr = layer(hr)

            if name in self.layers:
                k+=1
                gram_fake = gram_matrix(sr)
                gram_real = gram_matrix(hr)
                text_loss += criterion(gram_fake, gram_real) * (1.0/len(self.layers))
        
        return text_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
