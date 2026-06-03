import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from collections import OrderedDict

def initialize_kaiming_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def initialize_classifier_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
import clip.clip as clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class DayNightReID(nn.Module):
    def __init__(self, num_classes, num_img_day, num_img_night, cfg):
        super(DayNightReID, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(initialize_kaiming_weights)
        
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(initialize_kaiming_weights)

        self.classifier_day = nn.Linear(self.in_planes+self.in_planes_proj, 3000, bias=False)
        self.classifier_day.apply(initialize_classifier_weights)


        self.classifier_night = nn.Linear(self.in_planes+self.in_planes_proj, 3000, bias=False)
        self.classifier_night.apply(initialize_classifier_weights)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        self.prompt_learner = CrossDomainPromptLearner(num_img_day, num_img_night, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = CLIPTextEncoder(clip_model)

            
    def extract_all_features(self, x):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)
        # image_features_proj = image_features_proj.permute(1,0,2)
        return F.normalize(image_features[:, 0]), F.normalize(image_features_proj[:, 0])


    def forward(self, x=None, cam_label= None, view_label=None, idx=None, modal=1, vis_feat=None, get_text=None):

        if get_text==True:
            prompts = self.prompt_learner(idx, vis_feature=vis_feat, modal=modal)
            if modal == 1:
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_day)
            elif modal == 2:
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_night)
            else:
                return 0
            return text_features.float()

        _, image_features, image_features_proj = self.image_encoder(x) 
        img_feature = image_features[:,0]
        # img_feature_proj = image_features_proj.permute(1,0,2)
        img_feature_proj = image_features_proj[:,0]
            
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        out_feat = torch.cat([feat, feat_proj], dim=1)

        if self.training:
            if modal == 1:
                logit = self.classifier_day(out_feat)
            elif modal == 2:
                logit = self.classifier_night(out_feat)
            else:
                logit = 0
            return out_feat, logit, feat_proj
        else:
            return out_feat

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_classes, num_img_day, num_img_night):
    model = DayNightReID(num_classes, num_img_day, num_img_night, cfg)
    return model


class CrossDomainPromptLearner(nn.Module):
    def __init__(self, num_img_day, num_img_night, dtype, token_embedding):
        super().__init__()

        ctx_init_day = "A daytime photo of a X X X X vehicle."
        ctx_init_night = "A nighttime photo of a X X X X vehicle."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init_day = ctx_init_day.replace("_", " ")
        ctx_init_night = ctx_init_night.replace("_", " ")
        n_ctx = 5

        tokenized_prompts_day = clip.tokenize(ctx_init_day).cuda()
        tokenized_prompts_night = clip.tokenize(ctx_init_night).cuda()
        with torch.no_grad():
            embedding_day = token_embedding(tokenized_prompts_day).type(dtype)
            embedding_night = token_embedding(tokenized_prompts_night).type(dtype)
        self.tokenized_prompts_day = tokenized_prompts_day  # torch.Tensor
        self.tokenized_prompts_night = tokenized_prompts_night  # torch.Tensor

        n_cls_ctx = 4
        self.n_cls_ctx = n_cls_ctx
        num_vectors_day = torch.empty(num_img_day, n_cls_ctx, ctx_dim, dtype=dtype)
        num_vectors_night = torch.empty(num_img_night, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(num_vectors_day, std=0.02)
        nn.init.normal_(num_vectors_night, std=0.02)
        self.num_ctx_day = nn.Parameter(num_vectors_day)
        self.num_ctx_night = nn.Parameter(num_vectors_night)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_day", embedding_day[:, :n_ctx + 1, :])
        self.register_buffer("token_prefix_night", embedding_night[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix_day", embedding_day[:, n_ctx + 1 + n_cls_ctx:, :])
        self.register_buffer("token_suffix_night", embedding_night[:, n_ctx + 1 + n_cls_ctx:, :])

        self.instance_aware_bias_net = nn.Sequential(OrderedDict([
            ("Linear1", nn.Linear(1280, 1280 // 16)),
            ("ReLu", nn.ReLU(inplace=True)),
            ("Linear", nn.Linear(1280 // 16, ctx_dim))
        ]))

        
    def forward(self, index, vis_feature=None, modal=0):
        # ADD for CoCoOp
        vis_bias = self.instance_aware_bias_net(vis_feature)
        vis_bias = vis_bias.unsqueeze(1).repeat(1, self.n_cls_ctx, 1)

        if modal == 1:
            num_ctx_day = self.num_ctx_day[index] + vis_bias
            b = index.shape[0]
            prefix_day = self.token_prefix_day.expand(b, -1, -1)
            suffix_day = self.token_suffix_day.expand(b, -1, -1)

            prompts = torch.cat(
                [
                    prefix_day,  # (n_cls, 1, dim)
                    num_ctx_day,  # (n_cls, n_ctx, dim)
                    suffix_day,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return  prompts
        elif modal == 2:
            num_ctx_night = self.num_ctx_night[index] + vis_bias
            b = index.shape[0]
            prefix_night = self.token_prefix_night.expand(b, -1, -1)
            suffix_night = self.token_suffix_night.expand(b, -1, -1)

            prompts = torch.cat(
                [
                    prefix_night,  # (n_cls, 1, dim)
                    num_ctx_night,  # (n_cls, n_ctx, dim)
                    suffix_night,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts

        else:
            prompts = None
            return 0
