from uuid import uuid4

from diffusers import StableDiffusionPipeline
import torch


class ImageKit:
    device = "mps"  # change to cuda for prod
    pipeline = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipeline.to(device)
    pipeline.load_lora_weights("loras/style18-v2.safetensors", adapter_name="style18")
    pipeline.load_lora_weights("loras/Graffiti_v1.safetensors", adapter_name="grafiti")

    def __call__(self, prompt: str):
        uid = f"{uuid4()}"
        self.pipeline.set_adapters(
            ["style18", "grafiti"],
            adapter_weights=[0.25, 0.75],
        )
        image, *_ = self.pipeline(
            prompt,
            width=480,
            height=640,
            num_inference_steps=30,
            cross_attention_kwargs={"scale": 1.0},
        ).images
        image.save(f"output/{uid}.png")


if __name__ == "__main__":
    gen = ImageKit()
    gen("kids drawing, sketch, monochrome, dog chasing a ball")
