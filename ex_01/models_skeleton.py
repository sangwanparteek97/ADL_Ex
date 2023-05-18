import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        # TODO
        self.heads = heads
        self.dim_head = dim_head
        self.scale = self.dim_head ** 0.5
        # we need softmax layer and dropout
        # TODO
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
        # as well as the q linear layer
        # TODO
        self.q = nn.Linear(dim, dim_head * heads)  # need to checked, dim, dim or this is also correct after computation
        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
        # TODO
        self.k = nn.Linear(dim, dim_head * heads)
        # and the output linear layer followed by dropout
        self.v = nn.Linear(dim, dim_head * heads)
        # TODO
        self.output_linear = nn.Linear(dim_head * heads, dim)#### mistake
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, kv_include_self=False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention 
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        b, n, _, h = *x.shape, self.heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        # TODO: attention
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # calculating attention
        dot_product = torch.matmul(q, k.transpose(1, 2)) / torch.tensor(self.scale)
        attention = self.softmax(dot_product)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        #print(out.shape)
        #out = out.view(b, n, self.heads * self.dim_head)  # doubt
        out = self.output_linear(out)
        #print(f'output:{out.shape}')
        out = self.output_dropout(out)
        return out

    # ViT & CrossViT


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):   ###note :This projection step allows the original input
                                             # data to be combined with the newly learned features, resulting in a richer
                                             # and more expressive representation of the input.
        # TODO
        x = self.project_in(x)
        x = self.fn(x)  # function that transforms data into new dim
        x = self.project_out(x)  # output dimension
        return x


# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        # TODO: create # depth encoders using ProjectInOut
        # Note: no positional FFN here
        for d in range(depth): #### create twice because we want to project small and large patches and then cross attention with each other.
            ### small patch*large patch and large patch *small patch so two layers
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim,
                             PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, sm_dim,
                             PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))


    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (sm_tokens, lg_tokens))

        # Forward pass through the layers,
        for sm_attend_lg, lg_attend_sm in self.layers:  ##defined two projinout layers which has cross attention
            # cross attend to
            # 1. small cls token to large patches and ########doubt small cls or small patch???????
            # 2. large cls token to small patches
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls

        # TODO
        # finally concat sm/lg cls tokens with patch tokens
        sm_tokens = torch.cat((sm_cls,sm_patch_tokens),1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens),1)
        # TODO
        return sm_tokens, lg_tokens


# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            *,
            depth,
            sm_dim,
            lg_dim,
            sm_enc_params,
            lg_enc_params,
            cross_attn_heads,
            cross_attn_depth,
            cross_attn_dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): ### here we take modeule list because se didnt fix layer sequence.
            self.layers.append(nn.ModuleList([
                # 2 transformer branches, one for small, one for large patchs
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                # + 1 cross transformer block
                CrossTransformer(sm_dim, lg_dim, cross_attn_depth, cross_attn_heads,
                                 cross_attn_dim_head,dropout)

            ]))

    def forward(self, sm_tokens, lg_tokens):
        # forward through the transformer encoders and cross attention block
        for sm_patch_transformer, large_patch_transformer , cross_transformer in self.layers:
            sm_tokens = sm_patch_transformer(sm_tokens) ### normal attention for each small andlarge tokens
            lg_tokens = large_patch_transformer(lg_tokens)
            sm_tokens,lg_tokens = cross_transformer(sm_tokens,lg_tokens) ##cross atention.
        return sm_tokens, lg_tokens  ###same dimension


# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            patch_size,
            dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim, p2=patch_dim),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            ###copied from function below and changed dimention
           # TODO

        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        # TODO
        ##for each patch we are doing this
        self.position = nn.Parameter(torch.randn(1, num_patches, dim),requires_grad=True) ##required grad learns weights
        # create #dim cls tokens (for each patch embedding)
        # TODO
        self.class_tokens = nn.Parameter(torch.randn(1, 1, dim),requires_grad=True) ##one cls token for 1 image for each side
        # create dropput layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # forward through patch embedding layer
        x = self.to_patch_embedding(img)
        x_class = ###### ????/
        # concat class tokens
        x = torch.cat((x,x_class),dim=1)
        # and add positional embedding
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            num_classes,
            sm_dim,
            lg_dim,
            sm_patch_size=12,
            sm_enc_depth=1,
            sm_enc_heads=8,
            sm_enc_mlp_dim=2048,
            sm_enc_dim_head=64,
            lg_patch_size=16,
            lg_enc_depth=4,
            lg_enc_heads=8,
            lg_enc_mlp_dim=2048,
            lg_enc_dim_head=64,
            cross_attn_depth=2,
            cross_attn_heads=8,
            cross_attn_dim_head=64,
            depth=3,
            dropout=0.1,
            emb_dropout=0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        # TODO

        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        # TODO

        # and the multi-scale encoder
        # TODO

        # call the mlp heads w. the class tokens 
        # TODO

        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size=32, patch_size=8, num_classes=10, dim=64, depth=2, heads=8, mlp_dim=128, dropout=0.1,
              emb_dropout=0.1)
    cvit = CrossViT(image_size=32, num_classes=10, sm_dim=64, lg_dim=128, sm_patch_size=8,
                    sm_enc_depth=2, sm_enc_heads=8, sm_enc_mlp_dim=128, sm_enc_dim_head=64,
                    lg_patch_size=16, lg_enc_depth=2, lg_enc_heads=8, lg_enc_mlp_dim=128,
                    lg_enc_dim_head=64, cross_attn_depth=2, cross_attn_heads=8, cross_attn_dim_head=64,
                    depth=3, dropout=0.1, emb_dropout=0.1)
    print(vit(x).shape)
    print(cvit(x).shape)
