import requests
import os
from Extensions import Extensions
from io import BytesIO
import requests

try:
    from PIL import Image
except ImportError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow==9.5.0"])
    from PIL import Image
import logging


class stable_diffusion(Extensions):
    def __init__(
        self,
        STABLE_DIFFUSION_API_URL: str = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
        **kwargs,
    ):
        self.requirements = ["pillow"]
        self.STABLE_DIFFUSION_API_URL = (
            STABLE_DIFFUSION_API_URL
            if STABLE_DIFFUSION_API_URL
            else "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        )
        self.WORKING_DIRECTORY = os.path.join(os.getcwd(), "WORKSPACE")
        self.HUGGINGFACE_API_KEY = None

        if "settings" in self.agent_config:
            if "WORKING_DIRECTORY" in self.agent_config["settings"]:
                self.WORKING_DIRECTORY = self.agent_config["settings"][
                    "WORKING_DIRECTORY"
                ]
            if "HUGGINGFACE_API_KEY" in self.agent_config["settings"]:
                self.HUGGINGFACE_API_KEY = self.agent_config["settings"][
                    "HUGGINGFACE_API_KEY"
                ]
        os.makedirs(self.WORKING_DIRECTORY, exist_ok=True)
        self.commands = {
            "Generate Image with Stable Diffusion": self.generate_image,
        }

    async def generate_image(
        self,
        prompt: str,
        filename: str,
        negative_prompt: str = "",
        batch_size: int = 1,
        cfg_scale: int = 7,
        denoising_strength: int = 0,
        enable_hr: bool = False,
        eta: int = 0,
        firstphase_height: int = 0,
        firstphase_width: int = 0,
        height: int = 64,
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
        steps: int = 3,
        styles: list = [],
        subseed: int = -1,
        subseed_strength: int = 0,
        tiling: bool = False,
        width: int = 64,
    ) -> str:
        image_path = os.path.join(self.WORKING_DIRECTORY, filename)
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
            headers = {}
            self.STABLE_DIFFUSION_API_URL = (
                f"{self.STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img"
            )
            generation_settings = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "batch_size": batch_size,
                "cfg_scale": cfg_scale,
                "denoising_strength": denoising_strength,
                "enable_hr": enable_hr,
                "eta": eta,
                "firstphase_height": firstphase_height,
                "firstphase_width": firstphase_width,
                "height": height,
                "n_iter": n_iter,
                "restore_faces": restore_faces,
                "s_churn": s_churn,
                "s_noise": s_noise,
                "s_tmax": s_tmax,
                "s_tmin": s_tmin,
                "sampler_index": sampler_index,
                "seed": seed,
                "seed_resize_from_h": seed_resize_from_h,
                "seed_resize_from_w": seed_resize_from_w,
                "steps": steps,
                "styles": styles,
                "subseed": subseed,
                "subseed_strength": subseed_strength,
                "tiling": tiling,
                "width": width,
            }

        try:
            response = requests.post(
                self.STABLE_DIFFUSION_API_URL,
                headers=headers,
                json=generation_settings,
            )
            image = Image.open(BytesIO(response.content))
            logging.info(f"Image Generated for prompt: {prompt} at {image_path}.")
            image.save(image_path)
            return f"Stable Diffusion image saved to disk as {image_path}"
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return f"Error generating image: {e}"
