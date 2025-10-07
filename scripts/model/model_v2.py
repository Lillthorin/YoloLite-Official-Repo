import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import math

def init_detect_bias(head_moduledict, num_classes, p_obj=0.01):
    """Sätter rimliga start-bias för obj/cls (decoupled heads)."""
    obj_bias = -math.log((1 - p_obj) / p_obj)      # ~ -4.595 för p=0.01
    cls_bias = (-math.log(num_classes)) if num_classes > 1 else 0.0
    with torch.no_grad():
        head_moduledict["out"]["obj"].bias.fill_(obj_bias)
        head_moduledict["out"]["cls"].bias.fill_(cls_bias)
        head_moduledict["out"]["box"].bias.zero_()
def conv_block(c_in, c_out, n=1):
    """Bygger n st conv-baserade block."""
    layers = []
    for i in range(n):
        layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)
class DWConvBlock(nn.Module):
    """Depthwise separable conv block med ReLU."""
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        layers = []
        for i in range(n):
            layers.extend([
                nn.Conv2d(c_in if i==0 else c_out, c_in if i==0 else c_out,
                          kernel_size=3, padding=1, groups=(c_in if i==0 else c_out), bias=False),
                nn.Conv2d(c_in if i==0 else c_out, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def make_head(A, head_depth, C, fpn_channels):
    """Head med depthwise conv + ReLU, samma outputformat."""
    trunk = nn.Sequential(*[
        DWConvBlock(fpn_channels, fpn_channels) for _ in range(head_depth)
    ])

    out_layers = nn.ModuleDict({
        "box": nn.Conv2d(fpn_channels, A * 4, 1),
        "obj": nn.Conv2d(fpn_channels, A * 1, 1),
        "cls": nn.Conv2d(fpn_channels, A * C, 1)
    })
    return nn.ModuleDict({"trunk": trunk, "out": out_layers})



def _flatten_level_outputs(outs, export_concat: bool):
    if not export_concat:
        return outs
    flat = []
    for p in outs:  # p: [B, A, S, S, E]
        B, A, S, _, E = p.shape
        flat.append(p.view(B, -1, E))
    return torch.cat(flat, dim=1)  # [B, N_total, E]


class YOLOLiteMS(nn.Module):
    """
    Multi-scale YOLO-lite med FPN och decoupled heads (P3,P4,P5,P6).
    - Självbeskrivande: self.fpn_strides, self.num_anchors_per_level
    - Export: self.export_concat (False i träning, True i export)
    """
    def __init__(self, backbone="resnet18", num_classes=3, fpn_channels=128,
                 num_anchors_per_level=(3, 3, 3, 3), pretrained=True,
                 depth_multiple: float = 1.0, width_multiple: float = 1.0,
                 head_depth: int = 1, use_p6: bool = True):
        super().__init__()

        # Backbone
        tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        out_idx, reductions, chs = self._pick_out_indices(tmp.feature_info)

        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, out_indices=out_idx
        )
        self.reductions = reductions  # t.ex. [8,16,32]
        c3, c4, c5 = chs
        self.use_p6 = use_p6

        # Width/Depth scaling
        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        # FPN
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        self.smooth3  = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth4  = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth5  = conv_block(fpn_channels, fpn_channels, n=d)

        # P6 path (downsample P5)
        self.p6_down  = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn    = nn.BatchNorm2d(fpn_channels)
        self.p6_act   = nn.SiLU(inplace=True)
        self.smooth6  = conv_block(fpn_channels, fpn_channels, n=d)

        # Anchors per level (om 3 givna, duplicera P5→P6)
        C = num_classes
        if len(num_anchors_per_level) == 3:
            A3, A4, A5 = map(int, num_anchors_per_level)
            A6 = A5
            self.num_anchors_per_level = (A3, A4, A5, A6)
        else:
            A3, A4, A5, A6 = map(int, num_anchors_per_level)
            self.num_anchors_per_level = (A3, A4, A5, A6)
        self.num_classes = C

        # Export switches
        self.export_concat = False  # sätt True i exportscript
        self.export_decode = False  # finns för kompatibilitet om dina scripts kollar den

        # Heads
        self.head3 = make_head(A3, head_depth, C, fpn_channels)
        self.head4 = make_head(A4, head_depth, C, fpn_channels)
        self.head5 = make_head(A5, head_depth, C, fpn_channels)
        self.head6 = make_head(A6, head_depth, C, fpn_channels)

        # Init bias (obj/cls)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        init_detect_bias(self.head6, C)

        # Självbeskrivande strides (P3,P4,P5,(P6))
        base = list(self.reductions[-3:])  # [8,16,32] typiskt
        self.fpn_strides = base + [base[-1] * 2]  # [8,16,32,64]

    @staticmethod
    def _pick_out_indices(feature_info):
        n = len(feature_info)
        out_idx = list(range(n-3, n))  # tre sista
        reductions = [feature_info[i]["reduction"] for i in out_idx]
        chs = [feature_info[i]["num_chs"] for i in out_idx]
        return out_idx, reductions, chs

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def _forward_head(self, p, head_dict, A):
        p = head_dict["trunk"](p)
        box = head_dict["out"]["box"](p)
        obj = head_dict["out"]["obj"](p)
        cls = head_dict["out"]["cls"](p)

        B, _, S, _ = box.shape
        box = box.view(B, A, 4, S, S)
        obj = obj.view(B, A, 1, S, S)
        cls = cls.view(B, A, self.num_classes, S, S)
        out = torch.cat([box, obj, cls], dim=2)
        return out.permute(0, 1, 3, 4, 2).contiguous()  # [B,A,S,S,5+C]

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))
        outs = [
            self._forward_head(p3, self.head3, self.num_anchors_per_level[0]),
            self._forward_head(p4, self.head4, self.num_anchors_per_level[1]),
            self._forward_head(p5, self.head5, self.num_anchors_per_level[2]),
        ]
        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5))))
            outs.append(self._forward_head(p6, self.head6, self.num_anchors_per_level[3]))
        return _flatten_level_outputs(outs, self.export_concat)

    # --- Hjälp för export/scripts ---
    def get_strides(self):
        """Returnerar strides per FPN-nivå i samma ordning som outs."""
        return self.fpn_strides[:4 if self.use_p6 else 3]

    def get_num_anchors_per_level(self):
        """Returnerar (A3,A4,A5,(A6))."""
        return self.num_anchors_per_level[:4 if self.use_p6 else 3]

    def print_strides(self, img_size=640):
        with torch.no_grad():
            d = next(self.parameters()).device
            x = torch.zeros(1, 3, img_size, img_size, device=d)
            c3, c4, c5 = self.backbone(x)
            Ss = [c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            if self.use_p6:
                Ss.append(Ss[-1] // 2)
            strides = [img_size // S for S in Ss]
            print(f"[YOLOLiteMS] grids={Ss} → strides={strides}")


class YOLOLiteMS_CPU(nn.Module):
    """
    CPU-optimerad YOLO-lite (DW-separable, ReLU) med P3,P4,P5,P6.
    - Självbeskrivande: self.fpn_strides, self.num_anchors_per_level
    - Export: self.export_concat / self.export_decode
    Output: lista av [B,A,S,S,5+C] eller platt [B,N,5+C] om export_concat=True.
    """
    def __init__(self, backbone="mobilenetv3_small_100", num_classes=3, fpn_channels=96,
                 num_anchors_per_level=(3,3,3,3), pretrained=True,
                 depth_multiple=0.75, width_multiple=0.75, head_depth=1, use_p6: bool = True):
        super().__init__()

        tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        out_idx, reductions, chs = self._pick_out_indices(tmp.feature_info)
        self.backbone = timm.create_model(backbone, features_only=True,
                                          pretrained=pretrained, out_indices=out_idx)
        c3, c4, c5 = chs
        self.reductions = reductions
        self.use_p6 = use_p6

        # Width/Depth scaling
        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        # FPN (DW-vänlig)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        self.smooth3  = DWConvBlock(fpn_channels, fpn_channels, n=d)
        self.smooth4  = DWConvBlock(fpn_channels, fpn_channels, n=d)
        self.smooth5  = DWConvBlock(fpn_channels, fpn_channels, n=d)

        # P6 path (DW-vänlig)
        self.p6_down  = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn    = nn.BatchNorm2d(fpn_channels)
        self.p6_act   = nn.ReLU(inplace=True)
        self.smooth6  = DWConvBlock(fpn_channels, fpn_channels, n=d)

        # Anchors per level
        C = num_classes
        if len(num_anchors_per_level) == 3:
            A3, A4, A5 = map(int, num_anchors_per_level)
            A6 = A5
            self.num_anchors_per_level = (A3, A4, A5, A6)
        else:
            A3, A4, A5, A6 = map(int, num_anchors_per_level)
            self.num_anchors_per_level = (A3, A4, A5, A6)
        self.num_classes = C

        # Export switches
        self.export_concat = False
        self.export_decode = False

        # Heads (DW-vänlig)
        self.head3 = make_head(A3, head_depth, C, fpn_channels)
        self.head4 = make_head(A4, head_depth, C, fpn_channels)
        self.head5 = make_head(A5, head_depth, C, fpn_channels)
        self.head6 = make_head(A6, head_depth, C, fpn_channels)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        init_detect_bias(self.head6, C)

        # Självbeskrivande strides
        base = list(self.reductions[-3:])  # [8,16,32] typiskt för timm backbones
        self.fpn_strides = base + [base[-1] * 2]  # [8,16,32,64]

    @staticmethod
    def _pick_out_indices(feature_info):
        n = len(feature_info)
        out_idx = list(range(n-3, n))
        reductions = [feature_info[i]["reduction"] for i in out_idx]
        chs = [feature_info[i]["num_chs"] for i in out_idx]
        return out_idx, reductions, chs

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def _forward_head(self, p, head_dict, A):
        p = head_dict["trunk"](p)
        box = head_dict["out"]["box"](p)
        obj = head_dict["out"]["obj"](p)
        cls = head_dict["out"]["cls"](p)

        B, _, S, _ = box.shape
        box = box.view(B, A, 4, S, S)
        obj = obj.view(B, A, 1, S, S)
        cls = cls.view(B, A, self.num_classes, S, S)
        out = torch.cat([box, obj, cls], dim=2)
        return out.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))
        outs = [
            self._forward_head(p3, self.head3, self.num_anchors_per_level[0]),
            self._forward_head(p4, self.head4, self.num_anchors_per_level[1]),
            self._forward_head(p5, self.head5, self.num_anchors_per_level[2]),
        ]
        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5))))
            outs.append(self._forward_head(p6, self.head6, self.num_anchors_per_level[3]))
        return _flatten_level_outputs(outs, self.export_concat)

    # --- Hjälp för export/scripts ---
    def get_strides(self):
        return self.fpn_strides[:4 if self.use_p6 else 3]

    def get_num_anchors_per_level(self):
        return self.num_anchors_per_level[:4 if self.use_p6 else 3]

    def print_strides(self, img_size=640):
        with torch.no_grad():
            d = next(self.parameters()).device
            x = torch.zeros(1, 3, img_size, img_size, device=d)
            c3, c4, c5 = self.backbone(x)
            Ss = [c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            if self.use_p6:
                Ss.append(Ss[-1] // 2)
            strides = [img_size // S for S in Ss]
            print(f"[YOLOLiteMS_CPU] grids={Ss} → strides={strides}")