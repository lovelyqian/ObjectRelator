import torch
from torch import nn
import torch.nn.functional as F
import math

### --- Positional encoding--- ###
### --- Borrowed from Detr--- ###
class PositionEncodingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super(PositionEncodingSine2D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, isTarget = False):
        '''
        input x: B, C, H, W
        return pos: B, C, H, W
        
        '''
        not_mask = torch.ones(x.size()[0], x.size()[2], x.size()[3]).to(x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            ## no diff between source and target
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class EncoderLayerInnerAttention(nn.Module):
    """
    Transformer encoder with all paramters
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight):
        super(EncoderLayerInnerAttention, self).__init__()
        
        
        self.pos_weight = pos_weight
        self.feat_weight = feat_weight
        self.inner_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation = activation)
        self.posEncoder = PositionEncodingSine2D(d_model // 2)
        
        self.cross_encoder_layer = EncoderLayerCrossAttention(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                                dropout=dropout, activation=activation)
        
    def forward(self, x, y, featmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        
        # do cross attention on x and featmask
        x = self.cross_encoder_layer(x, featmask, None)[0]

        bx, cx, hx, wx = x.size()
        
        by, cy, hy, wy = y.size()
        
        posx = self.posEncoder(x)
        posy = self.posEncoder(y)
        
        
        featx = self.feat_weight * x + self.pos_weight * posx
        featy = self.feat_weight * y + self.pos_weight * posy
        
        
        ## input of transformer should be : seq_len * batch_size * feat_dim 
        featx = featx.flatten(2).permute(2, 0, 1) 
        featy = featy.flatten(2).permute(2, 0, 1)
        x_mask = x_mask.flatten(2).squeeze(1) if x_mask is not None else torch.cuda.BoolTensor(bx, hx * wx).fill_(False)
        y_mask = y_mask.flatten(2).squeeze(1) if y_mask is not None else torch.cuda.BoolTensor(by, hy * wy).fill_(False)
        
        ## input of transformer: (seq_len*2) * batch_size * feat_dim
        len_seq_x, len_seq_y = featx.size()[0], featy.size()[0]
        
        output = torch.cat([featx, featy], dim=0)
        src_key_padding_mask = torch.cat((x_mask, y_mask), dim=1)
        with torch.no_grad() : 
            src_mask = torch.cuda.BoolTensor(hx * wx + hy * wy, hx * wx + hy * wy).fill_(True)
            src_mask[:hx * wx, :hx * wx] = False
            src_mask[hx * wx :, hx * wx:] = False
        
        output = self.inner_encoder_layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        outx, outy = output.narrow(0, 0, len_seq_x), output.narrow(0, len_seq_x, len_seq_y)  
        outx, outy = outx.permute(1, 2, 0).view(bx, cx, hx, wx), outy.permute(1, 2, 0).view(by, cy, hy, wy)
        x_mask, y_mask = x_mask.view(bx, 1, hx, wx), y_mask.view(bx, 1, hy, wy)
        
        return  outx, outy, x_mask, y_mask
    
class EncoderLayerCrossAttention(nn.Module):
    """
    Transformer encoder with all paramters
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(EncoderLayerCrossAttention, self).__init__()
        
        self.cross_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation = activation)
        
    def forward(self, featx, featy, featmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        
        bx, cx, hx, wx = featx.size()
        by, cy, hy, wy = featy.size()
        
        ## input of transformer should be : seq_len * batch_size * feat_dim 
        featx = featx.flatten(2).permute(2, 0, 1) 
        featy = featy.flatten(2).permute(2, 0, 1)
        x_mask = x_mask.flatten(2).squeeze(1) if x_mask is not None else torch.cuda.BoolTensor(bx, hx * wx).fill_(False)
        y_mask = y_mask.flatten(2).squeeze(1) if y_mask is not None else torch.cuda.BoolTensor(by, hy * wy).fill_(False)
        
        ## input of transformer: (seq_len*2) * batch_size * feat_dim
        len_seq_x, len_seq_y = featx.size()[0], featy.size()[0]
        
        output = torch.cat([featx, featy], dim=0)
        src_key_padding_mask = torch.cat((x_mask, y_mask), dim=1)
        with torch.no_grad() : 
            src_mask = torch.cuda.BoolTensor(hx * wx + hy * wy, hx * wx + hy * wy).fill_(False)
            src_mask[:hx * wx, :hx * wx] = True
            src_mask[hx * wx :, hx * wx:] = True
            
        output = self.cross_encoder_layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        outx, outy = output.narrow(0, 0, len_seq_x), output.narrow(0, len_seq_x, len_seq_y)  
        outx, outy = outx.permute(1, 2, 0).view(bx, cx, hx, wx), outy.permute(1, 2, 0).view(by, cy, hy, wy)
        x_mask, y_mask = x_mask.view(bx, 1, hx, wx), y_mask.view(bx, 1, hy, wy)
        
        return  outx, outy, x_mask, y_mask
    
class EncoderLayerEmpty(nn.Module):
    """
    Transformer encoder with all paramters
    """
    def __init__(self):
        super(EncoderLayerEmpty, self).__init__()
        
    def forward(self, featx, featy, featmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        return  featx, featy, x_mask, y_mask
    
class EncoderLayerBlock(nn.Module):
    """
    Transformer encoder with all paramters
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight, layer_type) :
        super(EncoderLayerBlock, self).__init__()
        
        cross_encoder_layer = EncoderLayerCrossAttention(d_model, nhead, dim_feedforward, dropout, activation)
        att_encoder_layer = EncoderLayerInnerAttention(d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight)
        
        if layer_type[0] == 'C' :
            self.layer1 = cross_encoder_layer 
        elif layer_type[0] == 'I' :
            self.layer1 = att_encoder_layer 
        elif layer_type[0] == 'N' :
            self.layer1 = EncoderLayerEmpty()
        
        if layer_type[1] == 'C' :
            self.layer2 = cross_encoder_layer 
        elif layer_type[1] == 'I' :
            self.layer2 = att_encoder_layer 
        elif layer_type[1] == 'N' :
            self.layer2 = EncoderLayerEmpty()

    def forward(self, featx, featy, featmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        
        featx, featy, x_mask, y_mask = self.layer1(featx, featy, featmask, x_mask, y_mask)
        featx, featy, x_mask, y_mask = self.layer2(featx, featy, featmask, x_mask, y_mask)

        return  featx, featy, x_mask, y_mask
    
### --- Transformer Encoder --- ###
class Decoder(nn.Module):

    def __init__(self, in_channels=256, out_channels=3):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels, 128, 2, stride=2) # 60
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(128, out_channels, 2, stride=2) # 120

    def forward(self, x):

        x = self.deconv1(x)
        x = self.relu1(x)

        x = self.deconv2(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x
    
class ClsBranch(nn.Module):
    # branch to predict if the object exists or not.
    def __init__(self, in_dim):
        super(ClsBranch, self).__init__()

        self.conv = nn.Conv2d(in_dim, 1, 3)
        self.relu = nn.ReLU()

        self.mlp = nn.Sequential(*[nn.Linear(28*28, 32), 
                                    nn.ReLU(), 
                                    nn.Linear(32, 1), 
                                    nn.Sigmoid()])
        
    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        return x
        
class Encoder(nn.Module):
    """
    Transformer encoder with all paramters
    """
    def __init__(self, feat_dim, pos_weight = 0.1, feat_weight=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_type = ['I', 'C', 'I', 'C', 'I', 'C'], drop_feat = 0.1):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.feat_proj = nn.Conv2d(feat_dim, d_model, kernel_size=1)
        self.drop_feat = nn.Dropout2d(p=drop_feat)
        self.encoder_blocks = nn.ModuleList([EncoderLayerBlock(d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight, layer_type[i * 2 : i * 2 + 2]) for i in range(num_layers)])

        self.decoder = Decoder(d_model, 3)
        self.cls_branch = ClsBranch(in_dim=256)
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-7
        
        
    def forward(self, x, y, fmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        featx = self.feat_proj (x)
        featx = self.drop_feat(featx)
        
        bx, cx, hx, wx = featx.size()
        
        featy = self.feat_proj (y)
        featy = self.drop_feat(featy)
        
        by, cy, hy, wy = featy.size()

        featmask = self.feat_proj(fmask)

        for i in range(self.num_layers) : 
            featx, featy, x_mask, y_mask = self.encoder_blocks[i](featx, featy, featmask, x_mask, y_mask)
        
        out_cls = self.cls_branch(featy)
        outx = self.sigmoid(self.decoder(featx))
        outy = self.sigmoid(self.decoder(featy))

        outx = torch.clamp(outx, min=self.eps, max=1-self.eps)
        outy = torch.clamp(outy, min=self.eps, max=1-self.eps)
        
        return  outx, outy, out_cls
    
### --- Transformer Encoder --- ###
class TransEncoder(nn.Module):
    """
    Transformer encoder: small and large variants
    """
    def __init__(self, feat_dim=1024, pos_weight = 0.1, feat_weight = 1, dropout=0.1, activation='relu', mode='small', layer_type=['I', 'C', 'I', 'C', 'I', 'N'], drop_feat=0.1):
        super(TransEncoder, self).__init__()
        
        if mode == 'tiny' : 
            d_model=128
            nhead=2
            num_layers=3
            dim_feedforward=256
            
        elif mode == 'small' : 
            d_model=256
            nhead=2
            num_layers=3
            dim_feedforward=256
            
        elif mode == 'base' : 
            d_model=512
            nhead=8
            num_layers=3
            dim_feedforward=2048
            
        elif mode == 'large' : 
            d_model=512
            nhead=8
            num_layers=6
            dim_feedforward=2048
            
        self.net = Encoder(feat_dim, pos_weight, feat_weight, d_model, nhead, num_layers, dim_feedforward, dropout, activation, layer_type, drop_feat)
        
    def forward(self, x, y, fmask, x_mask = None, y_mask = None):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        
        '''
        outx, outy, out_cls = self.net(x, y, fmask, x_mask, y_mask)

        return  outx, outy, out_cls
    
if __name__ == '__main__' : 
    
    feat_dim = 256
    mode = 'small'
    x = torch.cuda.FloatTensor(2, feat_dim, 10, 10)
    x_mask = torch.cuda.BoolTensor(2, 1, 10, 10)
    
    net = TransEncoder()
    
    print (net)