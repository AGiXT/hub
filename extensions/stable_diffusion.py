import os
from io import BytesIO
import requests
import numpy as np
import base64
import io

try:
    from PIL import Image
except ImportError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow==9.5.0"])
    from PIL import Image
import logging


class stable_diffusion:
    def __init__(
        self,
        STABLE_DIFFUSION_API_URL="https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
        HUGGINGFACE_API_KEY=None,
        **kwargs,
    ):
        self.requirements = ["pillow"]
        self.STABLE_DIFFUSION_API_URL = (
            STABLE_DIFFUSION_API_URL
            if STABLE_DIFFUSION_API_URL
            else "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        )
        self.WORKING_DIRECTORY = os.path.join(os.getcwd(), "WORKSPACE")
        self.HUGGINGFACE_API_KEY = HUGGINGFACE_API_KEY
        os.makedirs(self.WORKING_DIRECTORY, exist_ok=True)
        self.commands = {
            "Generate Image with Stable Diffusion": self.generate_image,
        }

    async def generate_image(
        self,
        prompt: str,
        filename: str = "",
        negative_prompt: str = "",
        batch_size: int = 1,
        cfg_scale: int = 7,
        denoising_strength: int = 0,
        enable_hr: bool = False,
        eta: int = 0,
        firstphase_height: int = 0,
        firstphase_width: int = 0,
        height: int = 1080,
        n_iter: int = 1,
        restore_faces: bool = False,
        s_churn: int = 0,
        s_noise: int = 1,
        s_tmax: int = 0,
        s_tmin: int = 0,
        sampler_index: str = "Euler a",
        seed: int = -1,
        seed_resize_from_h: int = -1,
        seed_resize_from_w: int = -1,
        steps: int = 20,
        styles: list = [],
        subseed: int = -1,
        subseed_strength: int = 0,
        tiling: bool = False,
        width: int = 1920,
    ) -> str:
        if filename == "":
            filename = f"image_{np.random.randint(0, 1000000)}.png"
            if os.path.exists(os.path.join(self.WORKING_DIRECTORY, filename)):
                filename = f"image_{np.random.randint(0, 1000000)}.png"
        image_path = os.path.join(self.WORKING_DIRECTORY, filename)

        headers = {}
        if (
            self.STABLE_DIFFUSION_API_URL.startswith(
                "https://api-inference.huggingface.co/models"
            )
            and self.HUGGINGFACE_API_KEY is not None
        ):
            headers = {"Authorization": f"Bearer {self.HUGGINGFACE_API_KEY}"}
            generation_settings = {
                "inputs": prompt,
            }
        else:
            self.STABLE_DIFFUSION_API_URL = (
                f"{self.STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img"
            )
            generation_settings = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "batch_size": batch_size if batch_size else 1,
                "cfg_scale": cfg_scale if cfg_scale else 7,
                "denoising_strength": denoising_strength if denoising_strength else 0,
                "enable_hr": enable_hr if enable_hr else False,
                "eta": eta if eta else 0,
                "firstphase_height": firstphase_height if firstphase_height else 0,
                "firstphase_width": firstphase_width if firstphase_width else 0,
                "height": height if height else 1080,
                "n_iter": n_iter if n_iter else 1,
                "restore_faces": restore_faces if restore_faces else False,
                "s_churn": s_churn if s_churn else 0,
                "s_noise": s_noise if s_noise else 1,
                "s_tmax": s_tmax if s_tmax else 0,
                "s_tmin": s_tmin if s_tmin else 0,
                "sampler_index": sampler_index if sampler_index else "Euler a",
                "seed": seed if seed else -1,
                "seed_resize_from_h": seed_resize_from_h if seed_resize_from_h else -1,
                "seed_resize_from_w": seed_resize_from_w if seed_resize_from_w else -1,
                "steps": steps if steps else 20,
                "styles": styles if styles else [],
                "subseed": subseed if subseed else -1,
                "subseed_strength": subseed_strength if subseed_strength else 0,
                "tiling": tiling if tiling else False,
                "width": width if width else 1920,
            }

        try:
            response = requests.post(
                self.STABLE_DIFFUSION_API_URL,
                headers=headers,
                json=generation_settings,  # Use the 'json' parameter instead
            )
            if self.HUGGINGFACE_API_KEY is not None:
                image = Image.open(BytesIO(response.content))
            else:
                response = response.json()
                image_data = response["images"][-1]
                print(len(response["images"]))
                image_data = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_data))
            logging.info(f"Image Generated for prompt: {prompt} at {image_path}.")
            image.save(image_path)
            return f"Stable Diffusion image saved to disk as {image_path}"
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return f"Error generating image: {e}"
