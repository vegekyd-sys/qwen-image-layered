"""Qwen-Image-Layered on Replicate — decomposes an image into RGBA layers."""

import io
import os
import random
import tempfile
import zipfile
from typing import Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import QwenImageLayeredPipeline
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max


class Predictor(BasePredictor):
    def setup(self):
        """Load model into GPU memory."""
        self.pipeline = QwenImageLayeredPipeline.from_pretrained(
            "Qwen/Qwen-Image-Layered",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image to decompose into layers"),
        layers: int = Input(
            description="Number of layers to decompose into",
            default=4,
            ge=2,
            le=10,
        ),
        seed: int = Input(
            description="Random seed (0 for random)",
            default=0,
            ge=0,
            le=MAX_SEED,
        ),
        prompt: str = Input(
            description="Optional text prompt describing the image",
            default="",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default=" ",
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale",
            default=4.0,
            ge=1.0,
            le=10.0,
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps",
            default=50,
            ge=1,
            le=100,
        ),
        resolution: int = Input(
            description="Processing resolution (640 recommended, 1024 for higher quality)",
            default=640,
            choices=[640, 1024],
        ),
        cfg_normalize: bool = Input(
            description="Enable CFG normalization",
            default=True,
        ),
    ) -> Path:
        """Run layer decomposition and return a ZIP of RGBA PNG layers."""
        if seed == 0:
            seed = random.randint(1, MAX_SEED)

        pil_image = Image.open(str(image)).convert("RGBA")

        inputs = {
            "image": pil_image,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
            "true_cfg_scale": guidance_scale,
            "prompt": prompt or None,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
            "layers": layers,
            "resolution": resolution,
            "cfg_normalize": cfg_normalize,
            "use_en_prompt": True,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_images = output.images[0]

        # Package layers as ZIP
        out_path = Path(tempfile.mktemp(suffix=".zip"))
        with zipfile.ZipFile(str(out_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for i, layer_img in enumerate(output_images):
                buf = io.BytesIO()
                layer_img.save(buf, format="PNG")
                zf.writestr(f"layer_{i}.png", buf.getvalue())

        return out_path
