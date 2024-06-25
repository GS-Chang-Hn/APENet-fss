# @Time : 2023/5/30
# @Author : Z.chang
# @FileName: fewshot.py
# @Software: PyCharm
# @Description：Few-shot

from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import Encoder
from models import resnet_50_101
from models.vit_model import VisionTransformer
from functools import partial
import math
# from pytorch_pretrained_vit import ViT
from util import utils
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from models.cross_attention import cross_att


class FewShotSeg(nn.Module):
    """
       Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, depth=12, act_layer=None, norm_layer=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        # # # Encoder
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', resnet_50_101.resnet101(pretrained=True)), ]))

        self.proj = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=1)
        self.channel = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.vit_model = VisionTransformer(img_size=448,
                                           patch_size=32,
                                           in_c=512,
                                           embed_dim=512,
                                           # embed_dim=1024,
                                           depth=12,
                                           num_heads=16,
                                           # distilled=True,
                                           representation_size=None,
                                           num_classes=0)
        self.cross_att = cross_att(56)

        from models.clip_model import load_clip
        self.clip = load_clip("ViT-B/32")
        from models.tokenization_clip import SimpleTokenizer
        self.tokenizer = SimpleTokenizer("models/bpe_simple_vocab_16e6.txt.gz")

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        # -------------------CLIP特征提取------------------------
        coco_labels_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
                            8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
                            14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
                            19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
                            25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
                            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                            49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                            55: 'orange', 56: 'broccoli', 57: 'carrot',
                            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
                            64: 'potted plant', 65: 'bed', 67: 'dining table',
                            70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                            77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                            82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
                            89: 'hair drier', 90: 'toothbrush'}
        voc_labels_dict = {1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                           9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                           15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'}
        support_fg_temp = fore_mask[0][0][0]
        label_index = torch.unique(
            support_fg_temp[torch.nonzero(support_fg_temp)[:, 0], torch.nonzero(support_fg_temp)[:, 1]]).tolist()
        label_text = [coco_labels_dict[index] for index in label_index]
        # label_index = torch.unique(
        #     support_fg_temp[torch.nonzero(support_fg_temp)[:, 0], torch.nonzero(support_fg_temp)[:, 1]]).item()
        # label_text = voc_labels_dict[label_index]
        template_text = 'a photo of {}'.format(label_text)
        tokenize_info = self.tokenizer(template_text, return_tensors='pt', padding=True, truncation=True).to(
            fore_mask[0][0].device)
        label_feature = self.clip.encode_text(tokenize_info).unsqueeze(0).unsqueeze(0).expand(1, 56, 56,
                                                                                              512)  # CLIP输出特征 [1 512]
        # print(label_feature.shape)
        clip_txt_feature = label_feature.squeeze().permute([2, 1, 0])  # [512 56 56]

        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        ###### Extract and map features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts_resnet_out = self.encoder(imgs_concat)  # 2 512 56 56
        # @ 通过 1*1 1的卷积核  降维到1*512 VGG不需要
        img_fts_proj_out = self.proj(img_fts_resnet_out)  # 2 512 56 56
        fts_size = img_fts_proj_out.shape[-2:]  # 最后输出的维度
        supp_fts_proj_out = img_fts_proj_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # [1 1 1 512 56 56] # support_Way x Shot x B x C x H' x W'
        qry_fts_proj_out = img_fts_proj_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W' [1 1 512 56 56]

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        query_outputs_foreground = []  # [1 2 448 448]
        qry_fts_bg = qry_fts_proj_out.squeeze()  # 512 56 56
        # qry_fts_fg = qry_fts_proj_out.squeeze()
        "--------------------------------Feature combination module FCM start-------------------------------"
        fore_mask_expend = fore_mask.squeeze().expand(3, -1, -1)  # [3 448 448]
        for way in supp_imgs:
            imgs_fore = torch.cat(way, dim=0).squeeze()
            feature_fore = torch.mul(fore_mask_expend, imgs_fore).unsqueeze(dim=0)
            image_fore_renset50 = self.encoder(feature_fore)
            # image_fore = self.encoder(feature_fore) # vgg-16
            image_fore = self.proj(image_fore_renset50) # czb 降维度 512->1024
        supp_fore = image_fore[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)

        supp_fts_proj_out_fusion = supp_fts_proj_out + supp_fore  # [1 1 1 512 56 56] # 前景编码和未编码特征融合
        "--------------------------------Feature combination module FCM end-------------------------------"
        for epi in range(batch_size):
            supp_fg_fts = []
            for way in range(n_ways):
                shot_list = []
                for shot in range(n_shots):
                    supp_fts_fg_proj_out_way_shot_epi = supp_fts_proj_out_fusion[way, shot, [epi]]
                    fore_mask_way_shot_epi = fore_mask[way, shot, [epi]]
                    fg_fts = self.getFeatures(supp_fts_fg_proj_out_way_shot_epi, fore_mask_way_shot_epi)
                    support_foreground_fusion = F.interpolate(fg_fts[..., None, None], size=56,
                                               mode='bilinear').squeeze()  # [512 56 56]
                    key_fg_sup = self.cross_att(support_foreground_fusion, clip_txt_feature, support_foreground_fusion) # [512 56 56]
                    final_fusion_fts = self.getFeatures(key_fg_sup.unsqueeze(dim=0),
                                                         fore_mask_way_shot_epi)  # [1 512]
                    shot_list.append(final_fusion_fts)
                supp_fg_fts.append(shot_list)

            # supp_fg_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
            #                                  fore_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]
            "--------------------------------The foreground regions of support images end-------------------------------"
            # supp_bg_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
            #                                  back_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.handle_vit(self.vit_model(F.interpolate(
                self.getFeatures(supp_fts_proj_out[way, shot, [epi]], back_mask[way, shot, [epi]])[..., None, None],
                size=back_mask.shape[-2:], mode='bilinear')).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
                                            back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            # supp_bg_fts = []
            # for way in range(n_ways):
            #     shot_list = []
            #     for shot in range(n_shots):
            #         supp_fts_bg_proj_out_way_shot_epi = supp_fts_proj_out[way, shot, [epi]]
            #         bg_mask_way_shot_epi = back_mask[way, shot, [epi]]
            #         bg_fts = self.getFeatures(supp_fts_bg_proj_out_way_shot_epi, bg_mask_way_shot_epi)
            #         up_fts = F.interpolate(bg_fts[..., None, None], size=back_mask.shape[-2:], mode='bilinear')
            #         vit_fts = self.vit_model(up_fts)
            #         handle_fts = self.handle_vit(vit_fts.reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
            #                                      bg_mask_way_shot_epi)
            #         shot_list.append(handle_fts)
            #     supp_bg_fts.append(shot_list)
            "--------------------------------The background regions of support images end------------------------------"
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            foreground_prototypes = [bg_prototype, ] + fg_prototypes  # 组建一个list，存放前景和背景tensor，计算只是利用512的维度，56和14仅仅是大小
            dist1 = [self.calDist(qry_fts_proj_out[:, epi], prototype) for prototype in foreground_prototypes]
            query_foreground_pred = torch.stack(dist1, dim=1)  # [1 2 56 56]#
            first_pre_query_pso_mask = torch.mean(query_foreground_pred, dim=1, keepdim=True)  # [1, 1, 56 56]
            query_pso_mask = F.interpolate(first_pre_query_pso_mask, size=(448, 448), mode='bilinear',
                                           align_corners=False)
            query_pso_mask = query_pso_mask.unsqueeze(dim=0)  # [1 1 1 448 448]
            "---------------第一阶段 The prediction of objects in the query image end, generating a pso-mask-------------"
            query_fts = []
            for way in range(n_ways):
                shot_list = []
                for shot in range(n_shots):
                    query_fts_fg_proj_out_way_shot_epi = qry_fts_proj_out.unsqueeze(dim=0)[way, shot, [epi]]
                    query_fore_mask_way_shot_epi = query_pso_mask[way, shot, [epi]]
                    query_fg_fts = self.getFeatures(query_fts_fg_proj_out_way_shot_epi, query_fore_mask_way_shot_epi)
                    query_key_fg_sup = F.interpolate(query_fg_fts[..., None, None], size=56,
                                                     mode='bilinear').squeeze()  # [512 56 56]
                    """将key_fg_sup和query_key_fg_sup下采样为两层，在每一层中建立自注意力机制，在层间建立交叉注意力"""
                    Fs_low = self.cross_att(key_fg_sup, key_fg_sup, key_fg_sup)  # [512 56 56] 支持前景浅层特征的自注意力计算
                    Fq_low = self.cross_att(query_key_fg_sup, query_key_fg_sup,
                                            query_key_fg_sup)  # [512 56 56] 查询浅层特征的自注意力计算
                    cos_sq_low = self.cross_att(Fs_low, Fq_low, Fs_low)  # [512 56 56]

                    Fs_hih = self.cross_att(key_fg_sup, key_fg_sup, key_fg_sup)  # [512 56 56] 支持前景浅层特征的自注意力计算
                    Fq_hih = self.cross_att(query_key_fg_sup, query_key_fg_sup,
                                            query_key_fg_sup)  # [512 56 56] 查询浅层特征的自注意力计算
                    cos_sq_hih = self.cross_att(Fs_hih, Fq_hih, Fs_hih)  # [512 56 56]
                    cos_sq = self.distillation(cos_sq_low, cos_sq_hih)  # 简单的知识蒸馏策略
                    cos_att_sq = self.cross_att(key_fg_sup, query_key_fg_sup, cos_sq)
                    final_fusion_fts = self.getFeatures(cos_att_sq.unsqueeze(dim=0),
                                                        query_fore_mask_way_shot_epi)  # [1 512]
                    shot_list.append(final_fusion_fts)
                query_fts.append(shot_list)

            fg2_prototypes, bg2_prototype = self.getPrototype(query_fts, supp_bg_fts)
            foreground_prototypes = [bg2_prototype, ] + fg2_prototypes  # 组建一个list，存放前景和背景tensor，计算只是利用512的维度，56和14仅仅是大小
            dist2 = [self.calDist(qry_fts_proj_out[:, epi], prototype) for prototype in foreground_prototypes]
            query_foreground_pred_final = torch.stack(dist2, dim=1)  # [1 2 56 56]#
            query_outputs_foreground.append(F.interpolate(query_foreground_pred_final, size=img_size, mode='bilinear'))
            "-------------------------------训练-超参数设置-------------------------"
            ###### Prototype alignment loss ######
            # 测试阶段 self.training为Flase 说明测试阶段没有执行if
            # if self.config['align'] and self.training:
            # flag = True  # 定义一个flag  True则执行CG, False 则不执行CG
            # if self.config['align'] and flag:
            #     query_loss_foreground = self.alignLoss(qry_fts_proj_out[:, epi], query_foreground_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     query_loss_background = self.alignLoss(qry_fts_proj_out[:, epi], query_background_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     support_loss_foreground = self.alignLoss(supp_fts_proj_out[:, :, epi], support_foreground_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     support_loss_background = self.alignLoss(supp_fts_proj_out[:, :, epi], support_background_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #
            #
            #
            #
            #     query_ls1 += query_loss_foreground
            #     query_ls2 += query_loss_background
            #     align_loss_total = query_ls1 + query_ls2
        query_outputs_foreground = torch.stack(query_outputs_foreground, dim=1)  # N x B x (1 + Wa) x H x W
        que_output_foreground = query_outputs_foreground.view(-1, *query_outputs_foreground.shape[2:])

        return que_output_foreground  # [1 2 448 448] [shot 2 448 448]

    ###################@czb计算Query->Resnet/VGG->Feature与Support->Resnet/VGG->Vit->prototype之间的余弦相似度 #################
    # def self_attention(query, key, value, mask=None, dropout=None):
    #
    #     d_k = query.size(-1)
    #     # (nbatch, h, seq_len, d_k) @ (nbatch, h, d_k, seq_len) => (nbatch, h, seq_len, seq_len)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     p_attn = F.softmax(scores, dim=-1)
    #     if dropout:
    #         p_attn = dropout(p_attn)
    #     # (nbatch, h, seq_len, seq_len) * (nbatch, h, seq_len, d_k) = > (nbatch, h, seq_len, d_k)
    #     return torch.matmul(p_attn, value), p_attn

    def distillation(self, low_feature, high_feature):
        return low_feature + high_feature

    def calDist(self, query_cnn_out, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(query_cnn_out, prototype[..., None, None], dim=1) * scaler
        return dist

    # def getFeatures(self, fts, mask, is_vit):
    #     """
    #     Extract foreground and background features via masked average pooling
    #     全卷积网络（FCN）能够保留输入图像的中每个像素相对位置；所以通过将二值 mask 与提取到的特征图相乘就可以完全保留目标的特征信息，
    #     排除掉背景等无关类别的特征
    #     Args:
    #         fts: input features, expect shape: 1 x C x H' x W'
    #         mask: binary mask, expect shape: 1 x H x W
    #     """
    #     fts = F.interpolate(fts, size=mask.shape[-2:],
    #                         mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)
    #     # @czb
    #     if is_vit:
    #         masked_fts = torch.sum(fts, dim=(2, 3)) \
    #                      / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
    #     else:
    #         masked_fts = fts * mask[None, ...]
    #     # masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
    #     #                  / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
    #     result1 = np.array(masked_fts.cpu())
    #     return masked_fts
    #  @GL  常规getFeatures
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        全卷积网络（FCN）能够保留输入图像的中每个像素相对位置；所以通过将二值 mask 与提取到的特征图相乘就可以完全保留目标的特征信息，
        排除掉背景等无关类别的特征
        Args:
            fts: input features, expect shape: 1 x C x H' x W' [1 512 448 448]
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:],
                            mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)

        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts  # [1 512]

    #  @GL 针对vit前后 进行mask 以及sum
    def handle_vit(self, fts, mask):
        """
            对vit输出求均值
        """
        # fts = F.interpolate(fts, size=mask.shape[-2:],
        #                     mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)
        # if is_fore_vit:  # 送入vit前mask
        #     masked_fts = fts * mask[None, ...]
        # else:  # vit 出来进行sum
        #     masked_fts = torch.sum(fts, dim=(2, 3)) \
        #                  / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        masked_fts = torch.sum(fts, dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_fts

    # @czb ################通过平均前景和背景特征获得原型###############
    def getPrototype(self, fg_fts, bg_fts):  # param: 1*512*56*56,  1*512*14*14
        """
        Average the features to obtain the prototype，单一原型无法完全准确表示（类似于聚类，通过聚类不同的类可以达到同样的效果），提升多原型（multi-prototype）

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])  # 1， 5
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype  # return [tensor], tensor   一个list 一个tensor?

    ############## # @CZB过渡段学习CCG(Query->Support)##################
    #####@
    # def alignLoss(self, query_vgg_out, pred, support_resnet_out, support_fore_mask, support_back_mask):
    #     """
    #     Compute the loss for the prototype alignment branch
    #
    #     Args:
    #         query_resnet_out: embedding features for query images
    #             expect shape: N x C x H' x W'
    #         pred: predicted segmentation score
    #             expect shape: N x (1 + Way) x H x W
    #         support_resnet_out: embedding features for support images
    #             expect shape: Way x Shot x C x H' x W'
    #         support_fore_mask: foreground masks for support images
    #             expect shape: way x shot x H x W
    #         support_back_mask: background masks for support images
    #             expect shape: way x shot x H x W
    #     """
    #     n_ways, n_shots = len(support_fore_mask), len(support_fore_mask[0])
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
    #     binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 前景+1个背景
    #     skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  # 没懂
    #     ##########@czb query-mask########
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Way) x 1 x H' x W'
    #     query_prototypes = torch.sum(query_vgg_out.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
    #     ###########获取query的原型###########
    #     query_prototypes = query_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Way) x C
    #     # Compute the support loss
    #     loss = 0
    #     for way in range(n_ways):
    #         if way in skip_ways:
    #             continue
    #         # Get the query prototypes
    #         prototypes = [query_prototypes[[0]], query_prototypes[[way + 1]]]
    #         for shot in range(n_shots):
    #             img_fts = support_resnet_out[way, [shot]]
    #             supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
    #             supp_pred = torch.stack(supp_dist, dim=1)
    #             supp_pred = F.interpolate(supp_pred, size=support_fore_mask.shape[-2:],
    #                                       mode='bilinear')
    #             # Construct the support Ground-Truth segmentation
    #             supp_label = torch.full_like(support_fore_mask[way, shot], 255,
    #                                          device=img_fts.device).long()
    #             supp_label[support_fore_mask[way, shot] == 1] = 1
    #             supp_label[support_back_mask[way, shot] == 1] = 0
    #             # Compute Loss
    #             loss = loss + F.cross_entropy(
    #                 supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
    #     return loss
