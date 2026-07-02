"""Runtime compatibility shims loaded via PYTHONPATH for remote GR00T runs."""

from __future__ import annotations


def _patch_transformers_video_input() -> None:
    try:
        from typing import Any

        import transformers.image_utils as image_utils
    except Exception:
        return

    if not hasattr(image_utils, "VideoInput"):
        # Eagle2 dynamic modules in Isaac-GR00T import VideoInput from
        # transformers.image_utils. Some installed Transformers builds omit the
        # alias even though the runtime only needs it for annotations.
        image_utils.VideoInput = Any


def _patch_transformers_fast_docstrings() -> None:
    try:
        import transformers.image_processing_utils_fast as fast_utils
    except Exception:
        return

    if not hasattr(fast_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
        fast_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    if not hasattr(fast_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
        fast_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""


def _patch_transformers_fast_image_inputs() -> None:
    try:
        from functools import partial

        from transformers.image_processing_utils_fast import BaseImageProcessorFast
    except Exception:
        return

    if hasattr(BaseImageProcessorFast, "_prepare_input_images"):
        return

    def _prepare_input_images(
        self,
        images,
        do_convert_rgb=None,
        input_data_format=None,
        device=None,
    ):
        images = self._prepare_images_structure(images)
        process_image_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        return [process_image_fn(image) for image in images]

    BaseImageProcessorFast._prepare_input_images = _prepare_input_images


def _patch_transformers_group_images_by_shape() -> None:
    try:
        import transformers.image_processing_utils_fast as fast_utils
        import transformers.image_transforms as image_transforms
    except Exception:
        return

    for module in (fast_utils, image_transforms):
        original = getattr(module, "group_images_by_shape", None)
        if original is None or getattr(original, "_quantvla_compat_patched", False):
            continue

        def _compat_group_images_by_shape(
            images,
            *args,
            _original=original,
            disable_grouping=False,
            **kwargs,
        ):
            return _original(
                images,
                *args,
                disable_grouping=disable_grouping,
                **kwargs,
            )

        _compat_group_images_by_shape._quantvla_compat_patched = True
        module.group_images_by_shape = _compat_group_images_by_shape


def _patch_transformers_validate_init_kwargs() -> None:
    try:
        from transformers.processing_utils import ProcessorMixin
    except Exception:
        return

    original = ProcessorMixin.validate_init_kwargs
    if getattr(original, "_quantvla_compat_patched", False):
        return

    class _ValidateInitKwargsCompat(dict):
        def __init__(self, unused_kwargs: dict, valid_kwargs: dict) -> None:
            super().__init__(unused_kwargs)
            self.unused_kwargs = unused_kwargs
            self.valid_kwargs = valid_kwargs

        def __iter__(self):
            yield self.unused_kwargs
            yield self.valid_kwargs

        def __getitem__(self, key):
            if key == 0:
                return self.unused_kwargs
            if key == 1:
                return self.valid_kwargs
            return super().__getitem__(key)

    def _compat_validate_init_kwargs(processor_config, valid_kwargs):
        result = original(processor_config, valid_kwargs)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], dict)
            and isinstance(result[1], dict)
        ):
            return _ValidateInitKwargsCompat(result[0], result[1])
        return result

    _compat_validate_init_kwargs._quantvla_compat_patched = True
    ProcessorMixin.validate_init_kwargs = staticmethod(_compat_validate_init_kwargs)


def _patch_eagle_auto_processor() -> None:
    try:
        from transformers import AutoProcessor
        from transformers.image_utils import ChannelDimension
    except Exception:
        return

    original = AutoProcessor.from_pretrained
    if getattr(original, "_quantvla_compat_patched", False):
        return

    def _patch_eagle_image_processor(processor) -> None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return
        image_processor_cls = image_processor.__class__
        if image_processor_cls.__name__ != "Eagle2_5_VLImageProcessorFast":
            return

        original_preprocess = image_processor_cls.preprocess
        if getattr(original_preprocess, "_quantvla_compat_patched", False):
            return

        default_keys = (
            "crop_size",
            "data_format",
            "default_to_square",
            "device",
            "do_center_crop",
            "do_convert_rgb",
            "do_normalize",
            "do_pad",
            "do_rescale",
            "do_resize",
            "image_mean",
            "image_std",
            "input_data_format",
            "max_dynamic_tiles",
            "min_dynamic_tiles",
            "pad_during_tiling",
            "resample",
            "rescale_factor",
            "return_tensors",
            "size",
            "use_thumbnail",
        )
        annotations = getattr(image_processor_cls.valid_kwargs, "__annotations__", None)
        if isinstance(annotations, dict):
            for key in default_keys:
                annotations.setdefault(key, object)

        def _compat_preprocess(self, images, videos=None, **kwargs):
            defaults = {
                "crop_size": getattr(self, "crop_size", None),
                "data_format": ChannelDimension.FIRST,
                "default_to_square": getattr(self, "default_to_square", False),
                "device": None,
                "do_center_crop": getattr(self, "do_center_crop", None),
                "do_convert_rgb": getattr(self, "do_convert_rgb", True),
                "do_normalize": getattr(self, "do_normalize", True),
                "do_pad": getattr(self, "do_pad", True),
                "do_rescale": getattr(self, "do_rescale", True),
                "do_resize": getattr(self, "do_resize", True),
                "image_mean": getattr(self, "image_mean", None),
                "image_std": getattr(self, "image_std", None),
                "input_data_format": None,
                "max_dynamic_tiles": getattr(self, "max_dynamic_tiles", 12),
                "min_dynamic_tiles": getattr(self, "min_dynamic_tiles", 1),
                "pad_during_tiling": getattr(self, "pad_during_tiling", False),
                "resample": getattr(self, "resample", None),
                "rescale_factor": getattr(self, "rescale_factor", 1 / 255),
                "return_tensors": None,
                "size": getattr(self, "size", None),
                "use_thumbnail": getattr(self, "use_thumbnail", True),
            }
            for key, value in defaults.items():
                kwargs.setdefault(key, value)

            do_convert_rgb = kwargs.pop("do_convert_rgb")
            input_data_format = kwargs.pop("input_data_format")
            device = kwargs.pop("device")

            if images is not None:
                images = self._prepare_input_images(
                    images=images,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    device=device,
                )
            if videos is not None:
                videos = self._prepare_input_images(
                    images=videos,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    device=device,
                )

            kwargs = self._further_process_kwargs(**kwargs)
            self._validate_preprocess_kwargs(**kwargs)
            kwargs.pop("data_format", None)
            kwargs.pop("default_to_square", None)
            kwargs.pop("pad_size", None)

            if images is not None:
                return self._preprocess(images, **kwargs)
            if videos is not None:
                return self._preprocess(videos, **kwargs)
            return original_preprocess(self, images, videos=videos, **kwargs)

        _compat_preprocess._quantvla_compat_patched = True
        image_processor_cls.preprocess = _compat_preprocess

    def _compat_from_pretrained(cls, *args, **kwargs):
        processor = original(*args, **kwargs)
        _patch_eagle_image_processor(processor)
        return processor

    _compat_from_pretrained._quantvla_compat_patched = True
    AutoProcessor.from_pretrained = classmethod(_compat_from_pretrained)


_patch_transformers_video_input()
_patch_transformers_fast_docstrings()
_patch_transformers_fast_image_inputs()
_patch_transformers_group_images_by_shape()
_patch_transformers_validate_init_kwargs()
_patch_eagle_auto_processor()
