#!/usr/bin/env python3
"""Start the shared SAM3 service, optionally with its encoder weights stored in bfloat16.

The service runs SAM3 under bfloat16 autocast, which casts the float32 weights of every linear and
convolution layer to bfloat16 on each call. ``--bfloat16-encoders`` stores the weights of the image and
text encoders in bfloat16 once instead. Those layers only run under autocast, so they compute with the same
values and the service returns the same masks, scores and boxes, with less GPU memory.
"""

import argparse
import hashlib
import runpy
import sys
from pathlib import Path

EXPECTED_HTTP = "93a55e38e5d51aaa302c5fb4edb5d58635bee26d77d65d6a74f93668afda29a8"
EXPECTED_LOCAL = "b33d89de366930daafb7cf789155101a4d41cb3f1f526e809989c6c0ea74629c"
SOURCE = Path(__file__).resolve().parent / "pyrua-shared-sam3"
# The layers below these names are reached only from the image and text encoding calls.
ENCODERS = ("backbone.vision_backbone.trunk.", "backbone.language_backbone.encoder.")


def store_encoders_in_bfloat16(model):
    import torch

    converted = 0
    for name, module in model.named_modules():
        if not name.startswith(ENCODERS) or not isinstance(
            module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        ):
            continue
        for parameter in (module.weight, module.bias):
            if parameter is not None and parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
                converted += parameter.numel()
    torch.cuda.empty_cache()
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"SAM3 encoder weights stored in bfloat16: {converted} of {total} parameters", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfloat16-encoders", action="store_true")
    args, remaining = parser.parse_known_args()
    for name, expected in (("sam3_http.py", EXPECTED_HTTP), ("sam3_local.py", EXPECTED_LOCAL)):
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"the vendored {name} changed")
    if args.bfloat16_encoders:
        import sam3.model_builder as builder

        build = builder.build_sam3_image_model
        builder.build_sam3_image_model = lambda *a, **k: store_encoders_in_bfloat16(build(*a, **k))
    sys.argv = [str(SOURCE / "sam3_http.py"), *remaining]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
