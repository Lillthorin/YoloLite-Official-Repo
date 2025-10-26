# augment.py
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- safe wrappers to handle Albumentations API differences ----
def make_affine(**kw):
    """
    Try modern args first, fall back if not supported:
    - Some versions accept border_mode/cval, others ignore them.
    - We keep the geometric part the same either way.
    """
    try:
        return A.Affine(
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,  # newer/most common
            cval=(114, 114, 114),
            fit_output=False,
            **kw
        )
    except TypeError:
        # Fall back: drop cval/border_mode if not accepted
        return A.Affine(fit_output=False, **kw)

def make_pad(img_size):
    try:
        return A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # some versions accept tuple; others ignore -> still fine
        )
    except TypeError:
        # Fall back without value
        return A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT
        )

def make_gauss_noise(p=0.15):
    try:
        # Common signature
        return A.GaussNoise(var_limit=(5.0, 20.0), p=p)
    except TypeError:
        # Fall back to defaults
        return A.GaussNoise(p=p)

def letterbox_block(img_size):
    return [
        A.LongestMaxSize(max_size=img_size),
        make_pad(img_size),
    ]


def get_base_transform(img_size=416):
    """
    Stabil, industri-vänlig augmentering (letterbox i både train/val).
    Robust mot Albumentations versionsskillnader via wrappers ovan.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Resize(img_size,img_size, interpolation=cv2.INTER_LINEAR, p=0.3),
            make_affine(
                rotate=(-20, 20),
                shear=(-10, 10),
                scale=(0.85, 1.15),
                translate_percent=(0.05, 0.1),
                p=0.20,
            ),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
                A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.05),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15),
                A.RGBShift(),
                A.ChannelShuffle()
            ], p=0.40),

            A.OneOf([
                make_gauss_noise(p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.15),

            *letterbox_block(img_size),

            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.25,
            min_area=16,
            clip=True,
            filter_invalid_bboxes=True,
        ),
        is_check_shapes=True,
        p=1.0
    )

def get_strong_transform(img_size):
    return A.Compose([
        # A.Resize(imgsz, imgsz, p=1),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-20, 20),
            shear=(-10, 10),
            scale=(0.85, 1.15),
            translate_percent=(0.05, 0.1),
            p=0.3,
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.1),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.ChannelShuffle()
        ], p=0.1),

        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),  shadow_dimension=5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0.0, 1.0))
        ], p=0.2),

        A.CoarseDropout(num_holes_range=(3, 10), hole_height_range=(
            0.01, 0.05), hole_width_range=(0.01, 0.05), p=0.2),

        A.OneOf([
            A.GaussNoise(),
            A.MotionBlur(blur_limit=3)
        ], p=0.3),
        A.LongestMaxSize(max_size=img_size),
        make_pad(img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()

    ],
        bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.3,
        label_fields=['class_labels'],
        clip=True,
        filter_invalid_bboxes=True
    ),
        is_check_shapes=True,
        p=1.0
    )
def get_val_transform(img_size=416):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            make_pad(img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            clip=True,
            filter_invalid_bboxes=True,
        ),
        is_check_shapes=True,
        p=1.0
    )
