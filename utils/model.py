import os

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.utils import save_image

from .expert2 import MixtureOfExperts
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from .uncertainty_map import compute_uncertainty
# from .vmamba2 import VSSM


class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                            return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])  # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # pooling
        embed = self.project_head(embed)

        return {'feature': output['hidden_states'], 'project': embed}


class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Linear(768, project_dim)
        # hidden_size = self.model.config.hidden_size
        # hidden_size = self.model.config.hidden_sizes[-1]
        # self.project_head = nn.Linear(hidden_size, project_dim)
        self.spatial_dim = 768

        # self.vmunet = VSSM(in_chans=3,
        #                    num_classes=1,
        #                    depths=[2,2,9,2],
        #                    depths_decoder=[2,2,2,1],
        #                    drop_path_rate=0.2,
        #                    )
        #
        # model_dict = self.vmunet.state_dict()
        # modelCheckpoint = torch.load(os.path.join(os.path.dirname(__file__),
        #                                      'pre_trained_weights/vmamba_small_e238_ema.pth'))
        # pretrained_dict = modelCheckpoint['model']
        # # 过滤操作
        # new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        # model_dict.update(new_dict)
        # # 打印出来，更新了多少的参数
        # print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
        #                                                                            len(pretrained_dict),
        #                                                                            len(new_dict)))
        # self.vmunet.load_state_dict(model_dict)


    def forward(self, x):
        # f1, f2, f3, f4 = self.vmunet(x)
        # # b h w c --> b c h w
        # f1 = f1.permute(0, 3, 1, 2)  # torch.Size([24, 96, 56, 56])
        # f2 = f2.permute(0, 3, 1, 2)  # torch.Size([24, 192, 28, 28])
        # f3 = f3.permute(0, 3, 1, 2)  # torch.Size([24, 384, 14, 14])
        # f4 = f4.permute(0, 3, 1, 2)  # torch.Size([24, 768, 7, 7])
        # output = {'hidden_states':(f1, f1, f2, f3, f4),'pooler_output':None}
        # return {"feature":output['hidden_states'], "project":None}

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature": output['hidden_states'], "project": project}


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]  # 224*224
        feature_dim = [768, 384, 192, 96]
        # feature_dim = [768+384,384,192,96]

        num_expert = 4 # set num_expert for different dataset
        self.uncertainty_weight = nn.Parameter(torch.ones(1)*0.1,requires_grad=True)

        # self.fuzzy_measure = nn.Parameter(torch.linspace(0, 1, num_expert+1, device='cuda:0'))
        self.fuzzy_measure = torch.linspace(0, 1, num_expert+1, device='cuda:0')

        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        # self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        self.out = MixtureOfExperts(num_experts=num_expert, input_channels=24, output_channels=1, uncertainty_threshold=0.5)
        # self.out_os4 = MixtureOfExperts(num_experts=4, input_channels=96, output_channels=1, uncertainty_threshold=0.5)
        self.out_os4 = UnetOutBlock(2, in_channels=96, out_channels=1)
        # self.dinov2_vits14 = torch.hub.load('.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local').cuda()
        # self.dinov2_vits14.eval()

    def choquet_integral(self, features, weights):
        """
        计算Choquet积分
        """
        sorted_indices = torch.argsort(features, dim=1, descending=True)
        cum_weights = torch.cumsum(weights[sorted_indices], dim=1)
        integral = torch.sum(features * cum_weights, dim=1)
        return integral

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        # with torch.no_grad():
        #     image_dinov2 = self.dinov2_vits14.forward_features(image)['x_norm_patchtokens']
        #     image_dinov2 = image_dinov2.reshape(image_dinov2.shape[0], image_dinov2.shape[2],16, 16)
        #     image_dinov2_7 = F.interpolate(image_dinov2, size=7, mode='bilinear', align_corners=False)
        #     image_dinov2_7 = image_dinov2_7.permute(0,2,3,1)
        #     image_dinov2_7 = image_dinov2_7.reshape(image_dinov2_7.shape[0], -1, image_dinov2_7.shape[3])

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds, text_project = text_output['feature'], text_output['project']

        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]

        # os32 = torch.cat([image_features[3],image_dinov2_7],dim=2)
        os32 = image_features[3]
        text_feature = text_embeds[-1]

        os16 = self.decoder16(os32, image_features[2], text_feature)
        os8 = self.decoder8(os16, image_features[1], text_feature)
        os4 = self.decoder4(os8, image_features[0], text_feature)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        os4_out = self.out_os4(os4).sigmoid()
        os4_out = F.interpolate(os4_out.float(), scale_factor=4, mode='bilinear', align_corners=False)

        out, expert_outputs = self.out(os1)
        out = out.sigmoid()

        alloutput = torch.cat([expert_outputs.sigmoid(), out.unsqueeze(1)], dim=1).squeeze(2)

        alloutput = torch.where(alloutput >= 0.5, 1.0, 0.0)

        weight_um = torch.softmax(self.fuzzy_measure, dim=0)

        uncertainty_mask = compute_uncertainty(alloutput,weight_um)

        out_t = torch.where(out >= 0.5, out, 0.0).sum()/torch.where(out >= 0.5, 1.0, 0.0).sum()

        out_b = torch.where(out >= 0.5, 0.0, out).sum()/torch.where(out >= 0.5, 0.0, 1.0).sum()

        new_out = out + torch.clip(self.uncertainty_weight,0,0.1) * uncertainty_mask

        out = torch.where((out > out_b) & (out < out_t), new_out, out)

        return out, os4_out, uncertainty_mask
