"""Script to demo the INN's debiasing of face images.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, ToTensor
from torchvision.utils import save_image

from nifr.models import PartitionedInn, build_conv_inn

_INPUT_SHAPE = (3, 64, 64)


def main():
    # ========================== get checkpoint path and CSV file name ============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="Path to the checkpoint file.")
    parser.add_argument("--image-path", help="File path of the data sample to be transformed.")
    parser.add_argument(
        "--save-path",
        default="null-sampled-image.png",
        help="Where to save the transformed data sample to.",
    )
    demo_args = parser.parse_args()
    chkpt_path = Path(demo_args.checkpoint_path)

    # ============================= load ARGS from checkpoint file ================================
    print(f"Loading from '{chkpt_path}' ...")
    chkpt = torch.load(chkpt_path, map_location=torch.device("cpu"))

    if "args" in chkpt:
        model_args = chkpt["args"]
    elif "ARGS" in chkpt:
        model_args = chkpt["ARGS"]
    else:
        raise RuntimeError("Checkpoint doesn't contain args.")
    # ============================== construct commandline arguments ==============================
    model_args = SimpleNamespace(**model_args)

    image = Image.open(demo_args.image_path)
    cropper = CenterCrop(min(image.height, image.width))
    resizer = Resize((64, 64), Image.BICUBIC)
    tensorizer = ToTensor()
    image_tensor = tensorizer(resizer(cropper(image))).unsqueeze(0)

    # The INN expects inputs normalized to the range [-1, 1]
    image_tensor = (image_tensor - 0.5) / 0.5
    inn = PartitionedInn(
        args=model_args, input_shape=_INPUT_SHAPE, model=build_conv_inn(model_args, INPUT_SHAPE)
    )
    inn.load_state_dict(chkpt["model"])

    z = inn(image_tensor)
    zd_masked, zb_masked = inn.zero_mask(z)
    xd = inn.decode(zd_masked, partials=False)
    # Reverse the normalization
    xd = (xd * 0.5) + 0.5
    save_image(xd, f"{demo_args.save_path}.png")


if __name__ == "__main__":
    main()
