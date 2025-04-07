import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from PIL import Image

import folder_paths

from .easycontrol.lora_helper import set_single_lora
from .easycontrol.pipeline import FluxPipeline
from .easycontrol.transformer_flux import EasyControlFluxTransformer2DModel


def clear_cache(transformer):
    """Clears the attention processor cache in the transformer."""
    if hasattr(transformer, "attn_processors"):
        for name, attn_processor in transformer.attn_processors.items():
            if hasattr(attn_processor, "bank_kv") and hasattr(
                attn_processor.bank_kv, "clear"
            ):
                attn_processor.bank_kv.clear()


def comfy_tensor_to_pil(tensor: torch.Tensor | np.ndarray) -> list[Image.Image]:
    """Converts a ComfyUI IMAGE tensor or NumPy array to a list of PIL Images."""
    images = []
    if tensor is None:
        return images

    if isinstance(tensor, torch.Tensor):
        # Ensure tensor is on CPU and detached before converting to numpy
        tensor = tensor.cpu().detach()
        if tensor.dim() == 4:  # Batch dimension present B, H, W, C
            for i in range(tensor.shape[0]):
                img_np = tensor[i].numpy()
                # Ensure range is [0, 255] and type is uint8
                img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
                images.append(Image.fromarray(img_np))
        elif tensor.dim() == 3:  # Single image H, W, C
            img_np = tensor.numpy()
            img_np = (img_np.clip(0, 1) * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        else:
            print(
                f"Warning: Unexpected tensor dimension {tensor.dim()}, skipping conversion."
            )
    elif isinstance(tensor, np.ndarray):
        # Assuming numpy array follows [0, 1] float or [0, 255] uint8 convention
        if tensor.ndim == 4:  # Batch, H, W, C
            for i in range(tensor.shape[0]):
                img = tensor[i]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img.clip(0, 1) * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    print(
                        f"Warning: Unsupported numpy dtype {img.dtype}, attempting conversion."
                    )
                    img = (img.clip(0, 1) * 255).astype(np.uint8)  # Best guess
                images.append(Image.fromarray(img))
        elif tensor.ndim == 3:  # H, W, C
            img = tensor
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img.clip(0, 1) * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                print(
                    f"Warning: Unsupported numpy dtype {img.dtype}, attempting conversion."
                )
                img = (img.clip(0, 1) * 255).astype(np.uint8)  # Best guess
            images.append(Image.fromarray(img))
        else:
            print(
                f"Warning: Unexpected numpy array dimension {tensor.ndim}, skipping conversion."
            )
    else:
        print(
            f"Warning: Input type {type(tensor)} is not torch.Tensor or np.ndarray, skipping conversion."
        )

    return images


class EasyControlLoader:
    """Loads the Flux model, creates a custom pipeline, and applies an EasyControl LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_path": ("STRING",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
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

    def load_model(self, base_path, ckpt_name, lora_name, lora_weight, cond_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.backends.mps.is_available():
            device = "mps"

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        transformer = EasyControlFluxTransformer2DModel.from_single_file(
            ckpt_path,
            config=f"{base_path}/transformer",
            local_files_only=True,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        lora_path = folder_paths.get_full_path("loras", lora_name)
        set_single_lora(
            transformer,
            lora_path,
            lora_weights=[lora_weight],
            cond_size=cond_size,
            device=device,
        )
        pipe = FluxPipeline.from_pretrained(
            base_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.to(device)
        return (pipe,)


class EasyControlSampler:
    """Generates images using the EasyControl pipeline with optional spatial and subject conditioning."""

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
        """Generates an image using the pipeline and conditioning images."""
        # Prepare conditioning images by converting them to PIL format
        spatial_images_pil = comfy_tensor_to_pil(spatial_image)
        subject_images_pil = comfy_tensor_to_pil(subject_image)

        # Use None if list is empty, as pipeline might expect None instead of []
        spatial_images_arg = spatial_images_pil if spatial_images_pil else None
        subject_images_arg = subject_images_pil if subject_images_pil else None

        print(f"EasyControl: Generating image with seed {seed}")
        # Generate image
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,  # Consider making this configurable if needed
            generator=torch.Generator("cpu").manual_seed(
                seed
            ),  # Use pipeline's device for generator
            spatial_images=spatial_images_arg,
            subject_images=subject_images_arg,
            cond_size=cond_size,
        )

        # Convert output PIL image(s) back to ComfyUI tensor format (B, H, W, C)
        output_images = []
        if isinstance(output, FluxPipelineOutput) and output.images:
            pil_images = output.images  # Assuming this is a list of PIL images
        elif isinstance(
            output, list
        ):  # Handle case where output is directly a list of PIL
            pil_images = output
        elif isinstance(output, Image.Image):  # Handle single PIL image output
            pil_images = [output]
        else:
            print(f"Warning: Unexpected output type from pipeline: {type(output)}")
            pil_images = []

        for pil_image in pil_images:
            if isinstance(pil_image, Image.Image):
                try:
                    image_np = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)
                    # Ensure tensor has 4 dims: [Batch, Height, Width, Channel]
                    if image_tensor.dim() == 3:  # HWC -> BHWC
                        image_tensor = image_tensor.unsqueeze(0)
                    elif image_tensor.dim() != 4:
                        print(
                            f"Warning: Skipping image with unexpected dimensions: {image_tensor.shape}"
                        )
                        continue
                    output_images.append(image_tensor)
                except Exception as e:
                    print(f"Error converting PIL image to tensor: {e}")
            else:
                print(f"Warning: Item in output is not a PIL image: {type(pil_image)}")

        if not output_images:
            print("Error: No valid images generated or converted.")
            # Return an empty tensor with expected channel dimension
            # Use float() for consistency with typical ComfyUI IMAGE tensors
            return (torch.zeros((0, height, width, 3), dtype=torch.float32),)

        # Concatenate batch if multiple images were generated/returned
        final_output_tensor = torch.cat(output_images, dim=0)

        # Clear cache after generation
        clear_cache(pipe.transformer)
        print("EasyControl: Cache cleared.")

        return (final_output_tensor,)  # Return as tuple


NODE_CLASS_MAPPINGS = {
    "EasyControlLoader": EasyControlLoader,
    "EasyControlSampler": EasyControlSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyControlLoader": "EasyControl Loader",  # Added space for readability
    "EasyControlSampler": "EasyControl Sampler",  # Added space for readability
}
