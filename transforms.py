from typing import Tuple, Optional

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T



class ObjCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.75,
        max_aspect_ratio: float = 1.5,
        sampler_options: Optional[list[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            # sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            sampler_options = [0.2, 0.4, 0.8, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F.get_image_size(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class Resize(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = list(size)

    def forward(self, image: Tensor, target: Optional[dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F.get_image_size(image)
        if orig_w != self.size[1] or orig_h != self.size[0]:
            h_factor = self.size[0] / orig_h
            w_factor = self.size[1] / orig_w
            image = F.resize(image, size=self.size)

            # resize the boxes, assuming they are xyxy
            if target is not None:
                target["boxes"][:, 0::2] *= w_factor
                target["boxes"][:, 0::2] = torch.clamp(target["boxes"][:, 0::2], min=0, max=self.size[1])

                target["boxes"][:, 1::2] *= h_factor
                target["boxes"][:, 1::2] = torch.clamp(target["boxes"][:, 1::2], min=0, max=self.size[0])

        return image, target

class RandomZoomOut(nn.Module):
    def __init__(
        self, side_range: Tuple[float, float] = (1.0, 2.0), p: float = 0.5
    ):
        super().__init__()
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        orig_w, orig_h = F.get_image_size(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        img = image.reshape(3, -1)
        fill = torch.mean(img.type(torch.float32), dim=1)
        fill = fill.type(torch.uint8)
        fill = fill.detach().cpu().numpy()
        fill = tuple([int(x) for x in fill])

        tmp_f = fill
        if isinstance(image, torch.Tensor):
            tmp_f = int(0)

        image = F.pad(image, [left, top, right, bottom], fill=tmp_f)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w):] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


# Modification of RandomPhotometricDistort from COCO to remove the chanel shuffle in the original
class RandomPhotometricDistort(torch.nn.Module):
    def __init__(
        self,
        contrast: Tuple[float] = (0.5, 1.5),
        saturation: Tuple[float] = (0.5, 1.5),
        hue: Tuple[float] = (-0.05, 0.05),
        brightness: Tuple[float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        return image



class RandomPhotometricDistortObjDet(torch.nn.Module):
    def __init__(
        self,
        contrast: Tuple[float] = (0.5, 1.5),
        saturation: Tuple[float] = (0.5, 1.5),
        hue: Tuple[float] = (-0.05, 0.05),
        brightness: Tuple[float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image: torch.Tensor, target: Optional[dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        return image, target