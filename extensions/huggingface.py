import requests
import json
import os
from Extensions import Extensions
import requests


class huggingface(Extensions):
    def __init__(
        self,
        HUGGINGFACE_API_KEY: str = "",
        HUGGINGFACE_AUDIO_TO_TEXT_MODEL: str = "facebook/wav2vec2-large-960h-lv60-self",
        **kwargs,
    ):
        self.requirements = ["pillow"]
        self.HUGGINGFACE_API_KEY = HUGGINGFACE_API_KEY
        self.HUGGINGFACE_AUDIO_TO_TEXT_MODEL = HUGGINGFACE_AUDIO_TO_TEXT_MODEL
        self.WORKING_DIRECTORY = os.path.join(os.getcwd(), "WORKSPACE")
        if self.HUGGINGFACE_API_KEY is not None:
            self.commands = {
                "Read Audio from File": self.read_audio_from_file,
                "Read Audio": self.read_audio,
            }

    async def read_audio_from_file(self, audio_path: str):
        audio_path = os.path.join(self.WORKING_DIRECTORY, audio_path)
        with open(audio_path, "rb") as audio_file:
            audio = audio_file.read()
        return await self.read_audio(audio)

    async def read_audio(self, audio):
        model = self.HUGGINGFACE_AUDIO_TO_TEXT_MODEL
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        api_token = self.HUGGINGFACE_API_KEY
        headers = {"Authorization": f"Bearer {api_token}"}

        if api_token is None:
            raise ValueError(
                "You need to set your Hugging Face API token in the config file."
            )

        response = requests.post(
            api_url,
            headers=headers,
            data=audio,
        )

        text = json.loads(response.content.decode("utf-8"))["text"]
        return "The audio says: " + text
