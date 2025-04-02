import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from PIL import Image

import folder_paths

from .easycontrol.lora_helper import set_single_lora
from .easycontrol.pipeline import FluxPipeline
from .easycontrol.transformer_flux import FluxTransformer2DModel


def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()


class EasyControlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_path": ("STRING",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "cond_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 64},
                ),
            },
        }

    RETURN_TYPES = ("MODEL_EASYCONTROL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "EasyControl"

    def load_model(self, base_path, lora_name, lora_weight, cond_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = FluxPipeline.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device=device,
        )
        lora_path = folder_paths.get_full_path("loras", lora_name)
        set_single_lora(
            transformer,
            lora_path,
            lora_weights=[lora_weight],
            cond_size=cond_size,
            device=device,
        )
        pipe.transformer = transformer
        pipe.to(device)

        return (pipe,)


class EasyControlSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL_EASYCONTROL",),
                "prompt": ("STRING", {"multiline": True}),
                "height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 2048, "step": 64},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 25, "min": 1, "max": 100, "step": 1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cond_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 64},
                ),
            },
            "optional": {
                "spatial_image": ("IMAGE",),
                "subject_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "EasyControl"

    def generate(
        self,
        pipe,
        prompt,
        height,
        width,
        guidance_scale,
        num_inference_steps,
        seed,
        cond_size,
        spatial_image=None,
        subject_image=None,
    ):
        # Prepare spatial images
        spatial_images = []
        if spatial_image is not None:
            # Convert from tensor or numpy to PIL
            if isinstance(spatial_image, torch.Tensor):
                # Handle single image or batch
                if spatial_image.dim() == 4:  # [batch, height, width, channels]
                    for i in range(spatial_image.shape[0]):
                        img = spatial_image[i].cpu().numpy()
                        spatial_image_pil = Image.fromarray(
                            (img * 255).astype(np.uint8)
                        )
                        spatial_images.append(spatial_image_pil)
                else:  # [height, width, channels]
                    img = spatial_image.cpu().numpy()
                    spatial_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    spatial_images.append(spatial_image_pil)
            elif isinstance(spatial_image, np.ndarray):
                spatial_image_pil = Image.fromarray(
                    (spatial_image * 255).astype(np.uint8)
                )
                spatial_images.append(spatial_image_pil)

        # Prepare subject images
        subject_images = []
        if subject_image is not None:
            # Convert from tensor or numpy to PIL
            if isinstance(subject_image, torch.Tensor):
                # Handle single image or batch
                if subject_image.dim() == 4:  # [batch, height, width, channels]
                    for i in range(subject_image.shape[0]):
                        img = subject_image[i].cpu().numpy()
                        subject_image_pil = Image.fromarray(
                            (img * 255).astype(np.uint8)
                        )
                        subject_images.append(subject_image_pil)
                else:  # [height, width, channels]
                    img = subject_image.cpu().numpy()
                    subject_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    subject_images.append(subject_image_pil)
            elif isinstance(subject_image, np.ndarray):
                subject_image_pil = Image.fromarray(
                    (subject_image * 255).astype(np.uint8)
                )
                subject_images.append(subject_image_pil)

        # Generate image
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            spatial_images=spatial_images,
            subject_images=subject_images,
            cond_size=cond_size,
        )

        # Convert PIL image to numpy array, then to torch.Tensor
        if isinstance(output, FluxPipelineOutput):
            image = np.array(output.images[0]) / 255.0
        else:
            image = np.array(output[0]) / 255.0

        # Convert numpy array to torch.Tensor
        image = torch.from_numpy(image).float()

        # Add batch dimension to make it [batch, height, width, channels]
        if image.dim() == 3:  # [height, width, channels]
            image = image.unsqueeze(
                0
            )  # Add batch dimension to make it [1, height, width, channels]

        clear_cache(pipe.transformer)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "EasyControlLoader": EasyControlLoader,
    "EasyControlSampler": EasyControlSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlLoader": "EasyControlLoader",
    "EasyControlSampler": "EasyControlSampler",
}
