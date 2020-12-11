import timm
import torch


def build_backbone(model_achitecture, num_in_channels):
    if model_achitecture == "resnet18":
        backbone = timm.create_model("resnet18", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 512

    elif model_achitecture == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 512

    elif model_achitecture == "resnet50":
        backbone = timm.create_model("resnet50", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "seresnext26t_32x4d":
        backbone = timm.create_model("seresnext26t_32x4d", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "seresnext50_32x4d":
        backbone = timm.create_model("seresnext50_32x4d", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "resnetblur18":
        backbone = timm.create_model("resnetblur18", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 512

    elif model_achitecture == "efficientnet_b1":
        backbone = timm.create_model("tf_efficientnet_b1_ns", pretrained=True, in_chans=num_in_channels)
        print(backbone.feature_info)
        backbone_out_features = 1280

    elif model_achitecture == "efficientnet_b2":
        backbone = timm.create_model("efficientnet_b2", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 1408

    elif model_achitecture == "efficientnet_b3":
        backbone = timm.create_model("tf_efficientnet_b3_ns", pretrained=True, in_chans=num_in_channels)
        print(backbone.feature_info)
        backbone_out_features = 1536

    elif model_achitecture == "efficientnet_b4":
        backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 1792

    elif model_achitecture == "efficientnet_b5":
        backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "efficientnet_b6":
        backbone = timm.create_model("tf_efficientnet_b6_ns", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2304

    elif model_achitecture == "mobilenetv3_large_100":
        backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 1280

    elif model_achitecture == "hrnet_w18":
        backbone = timm.create_model("hrnet_w18", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "hrnet_w18_small":
        backbone = timm.create_model("hrnet_w18_small", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "hrnet_w18_small_v2":
        backbone = timm.create_model("hrnet_w18_small_v2", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture.startswith("hrnet_"):
        backbone = timm.create_model(model_achitecture, pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "xception41":
        backbone = timm.create_model("xception41", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "xception65":
        backbone = timm.create_model("xception65", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture == "xception71":
        backbone = timm.create_model("xception71", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 2048

    elif model_achitecture.startswith("regnet"):
        backbone = timm.create_model(model_achitecture, pretrained=True, in_chans=num_in_channels)
        backbone_out_features = dict(regnetx_004=384, regnety_004=440)[model_achitecture]

    elif model_achitecture == "densenet121":
        backbone = timm.create_model("densenet121", pretrained=True, in_chans=num_in_channels)
        backbone_out_features = 1024

    else:
        raise RuntimeError("Invalid model_achitecture:", model_achitecture)

    return backbone, backbone_out_features


if __name__ == "__main__":
    import pytorch_model_summary

    nb_raster_channels = 13

    backbone, size = build_backbone("hrnet_w48", nb_raster_channels)

    print(
        pytorch_model_summary.summary(
            backbone,
            torch.zeros((1, nb_raster_channels, 288, 288)),
            show_input=False,
            show_hierarchical=False,
            print_summary=True,
        )
    )
