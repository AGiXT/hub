try:
    from transformers import AutoTokenizer
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import AutoTokenizer
try:
    from petals import AutoDistributedModelForCausalLM
except ImportError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "petals"])
    from petals import AutoDistributedModelForCausalLM


class PetalsProvider:
    def __init__(
        self,
        MAX_TOKENS: int = 4096,
        AI_MODEL="stabilityai/StableBeluga2",
        AI_TEMPERATURE: float = 0.7,
        **kwargs,
    ):
        self.MAX_TOKENS = MAX_TOKENS if MAX_TOKENS else 4096
        self.AI_MODEL = AI_MODEL if AI_MODEL else "stabilityai/StableBeluga2"
        self.AI_TEMPERATURE = AI_TEMPERATURE if AI_TEMPERATURE else 0.7

    async def instruct(self, prompt, tokens: int = 0):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.AI_MODEL)
            model = AutoDistributedModelForCausalLM.from_pretrained(self.AI_MODEL)
            inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
            max_new_tokens = int(self.MAX_TOKENS) - int(tokens)
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
            return tokenizer.decode(outputs[0])
        except Exception as e:
            return f"Petals Error: {e}"
