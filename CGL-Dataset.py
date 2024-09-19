# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Type

import datasets as ds
from datasets.utils.logging import get_logger
from hfcocoapi.models import CategoryData, ImageData
from hfcocoapi.processors import MsCocoProcessor
from hfcocoapi.tasks.annotation import AnnotationData
from hfcocoapi.typehint import Bbox, CategoryId, JsonDict
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ralfpt.saliency_detection import SaliencyTester
    from simple_lama_inpainting import SimpleLama

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{ijcai2022p692,
    title     = {Composition-aware Graphic Layout GAN for Visual-Textual Presentation Designs},
    author    = {Zhou, Min and Xu, Chenchen and Ma, Ye and Ge, Tiezheng and Jiang, Yuning and Xu, Weiwei},
    booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    editor    = {Lud De Raedt},
    pages     = {4995--5001},
    year      = {2022},
    month     = {7},
    note      = {AI and Arts},
    doi       = {10.24963/ijcai.2022/692},
    url       = {https://doi.org/10.24963/ijcai.2022/692},
}
"""

_DESCRIPTION = """\
The CGL-Dataset is a dataset used for the task of automatic graphic layout design for advertising posters. It contains 61,548 samples and is provided by Alibaba Group.
"""

_HOMEPAGE = "https://github.com/minzhouGithub/CGL-GAN"

_LICENSE = "cc-by-sa-4.0"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "layout_images": [
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_1.zip",
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_2.zip",
    ],
    "layout_json": "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_json_yinhe.zip",
}

# The correspondence of the following category names
# is referred to https://tianchi.aliyun.com/dataset/142692#json-file-structure
CATEGORIES: Dict[str, str] = {
    "Logo": "logo",
    "文字": "text",
    "衬底": "underlay",
    "符号元素": "embellishment",
    "强调突出子部分文字": "highlighted text",
}


class CGLAnnotationData(AnnotationData):
    area: float
    bbox: Bbox
    category_id: CategoryId


class CGLProcessor(MsCocoProcessor):
    def get_features_base_dict(self, is_ralf_style: bool):
        base_features = {
            "image_id": ds.Value("int64"),
            "file_name": ds.Value("string"),
            "width": ds.Value("int64"),
            "height": ds.Value("int64"),
        }
        image_features = (
            {
                "original_poster": ds.Image(),
                "inpainted_poster": ds.Image(),
            }
            if is_ralf_style
            else {
                "image": ds.Image(),
            }
        )
        return {**base_features, **image_features}

    def get_features_instance_dict(self, rename_category_names: bool):
        category_names = (
            list(CATEGORIES.values())
            if rename_category_names
            else list(CATEGORIES.keys())
        )
        return {
            "area": ds.Value("int64"),
            "bbox": ds.Sequence(ds.Value("int64")),
            "category": {
                "category_id": ds.Value("int64"),
                "name": ds.ClassLabel(
                    num_classes=len(category_names), names=category_names
                ),
                "supercategory": ds.Value("string"),
            },
        }

    def get_features(
        self, rename_category_names: bool, is_ralf_style: bool
    ) -> ds.Features:
        features_dict = self.get_features_base_dict(is_ralf_style)
        annotations = ds.Sequence(
            self.get_features_instance_dict(
                rename_category_names=rename_category_names,
            )
        )
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_images_data(
        self,
        image_dicts: List[JsonDict],
        image_data_class: Type[ImageData] = ImageData,
        tqdm_desc: str = "Load images",
    ) -> List[ImageData]:
        images = [
            image_data_class(**image_dict)
            for image_dict in tqdm(image_dicts, desc=tqdm_desc)
        ]
        return images

    def load_categories_data(
        self,
        category_dicts: List[JsonDict],
        rename_category_names: bool,
        category_data_class: Type[CategoryData] = CategoryData,
        tqdm_desc: str = "Load categories",
    ) -> Dict[CategoryId, CategoryData]:
        categories = {}
        for cat_dict in tqdm(category_dicts, desc=tqdm_desc):
            if rename_category_names:
                cat_dict["name"] = CATEGORIES[cat_dict["name"]]
                cat_dict["supercategory"] = CATEGORIES[cat_dict["supercategory"]]
            category_data = category_data_class(**cat_dict)
            categories[category_data.category_id] = category_data
        return categories

    def load_data(
        self,
        ann_dicts: List[List[JsonDict]],
        tqdm_desc: str = "Load annotation data",
    ) -> List[List[CGLAnnotationData]]:
        annotations = [
            [
                CGLAnnotationData(id=ann_id, **ann_dict)
                for ann_id, ann_dict in enumerate(ann_dict_list)
            ]
            for ann_dict_list in tqdm(ann_dicts, desc=tqdm_desc)
        ]
        return annotations

    def generate_examples(
        self,
        images: List[ImageData],
        annotations: List[List[CGLAnnotationData]],
        image_files: Dict[str, pathlib.Path],
        categories: Dict[CategoryId, CategoryData],
    ):
        for idx, (img_data, ann_list) in enumerate(zip(images, annotations)):
            img_path = image_files[img_data.file_name]
            img = self.load_image(img_path)

            example = img_data.model_dump()
            example["image"] = img

            example["annotations"] = []
            for ann in ann_list:
                ann_dict = ann.model_dump()
                category = categories[ann.category_id]
                ann_dict["category"] = category.model_dump()
                example["annotations"].append(ann_dict)

            yield idx, example


def ralf_style_example(
    example,
    inpainter: SimpleLama,
    saliency_testers: List[SaliencyTester],
    saliency_map_cols: Sequence[str],
):
    from ralfpt.inpainting import apply_inpainting
    from ralfpt.saliency_detection import apply_saliency_detection
    from ralfpt.transforms import has_valid_area, load_from_cgl_ltwh
    from ralfpt.typehints import Element

    assert len(saliency_testers) == len(saliency_map_cols)

    original_image = example.pop("image")
    example["original_poster"] = original_image

    def get_cgl_layout_elements(
        annotations, image_w: int, image_h: int
    ) -> List[Element]:
        elements = []

        for ann in annotations:
            category = ann["category"]
            label = category["name"]

            coordinates = load_from_cgl_ltwh(
                ltwh=ann["bbox"], global_width=image_w, global_height=image_h
            )
            if has_valid_area(**coordinates):
                element: Element = {"label": label, "coordinates": coordinates}
                elements.append(element)

        return elements

    image_w, image_h = example["width"], example["height"]

    elements = get_cgl_layout_elements(
        annotations=example["annotations"], image_w=image_w, image_h=image_h
    )

    #
    # Apply RALF-style inpainting
    #
    inpainted_image = apply_inpainting(
        image=original_image, elements=elements, inpainter=inpainter
    )
    example["inpainted_poster"] = inpainted_image

    #
    # Apply Ralf-style saliency detection
    #
    saliency_maps = apply_saliency_detection(
        image=inpainted_image,
        saliency_testers=saliency_testers,  # type: ignore
    )
    for sal_col, sal_map in zip(saliency_map_cols, saliency_maps):
        example[sal_col] = sal_map

    return example


@dataclass
class CGLDatasetConfig(ds.BuilderConfig):
    rename_category_names: bool = False
    processor: CGLProcessor = CGLProcessor()

    saliency_maps: Sequence[str] = (
        "saliency_map",
        "saliency_map_sub",
    )
    saliency_testers: Sequence[str] = (
        "creative-graphic-design/ISNet-general-use",
        "creative-graphic-design/BASNet-SmartText",
    )

    def __post_init__(self):
        super().__post_init__()
        assert len(self.saliency_maps) == len(self.saliency_testers)


class CGLDataset(ds.GeneratorBasedBuilder):
    """A class for loading CGL-Dataset dataset."""

    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetConfig
    BUILDER_CONFIGS = [
        CGLDatasetConfig(name="default", version=VERSION, description=_DESCRIPTION),
        CGLDatasetConfig(name="ralf", version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        config: CGLDatasetConfig = self.config  # type: ignore
        processor = config.processor
        features = processor.get_features(
            rename_category_names=config.rename_category_names,
            is_ralf_style=config.name == "ralf",
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def get_image_file_paths(
        self, image_dirs: List[pathlib.Path]
    ) -> Dict[str, pathlib.Path]:
        image_file_paths: Dict[str, pathlib.Path] = {}

        for image_dir in image_dirs:
            jpg_files = list(image_dir.glob("**/*.jpg"))
            png_files = list(image_dir.glob("**/*.png"))
            image_files = jpg_files + png_files
            for image_file in image_files:
                assert image_file.name not in image_file_paths, image_file
                image_file_paths[image_file.name] = image_file

        return image_file_paths

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        file_paths = dl_manager.download_and_extract(_URLS)

        layout_json_dir = pathlib.Path(file_paths["layout_json"])  # type: ignore

        tng_json_file = layout_json_dir / "layout_train_6w_fixed_v2.json"
        val_json_file = layout_json_dir / "layout_test_6w_fixed_v2.json"
        tst_json_file = layout_json_dir / "yinhe.json"

        tng_val_image_files = self.get_image_file_paths(
            image_dirs=[pathlib.Path(d) for d in file_paths["layout_images"]]  # type: ignore
        )
        tst_image_files = self.get_image_file_paths(
            image_dirs=[layout_json_dir / "yinhe_imgs"]
        )

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "image_files": tng_val_image_files,
                    "annotation_path": tng_json_file,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "image_files": tng_val_image_files,
                    "annotation_path": val_json_file,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={
                    "image_files": tst_image_files,
                    "annotation_path": tst_json_file,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self, image_files: Dict[str, pathlib.Path], annotation_path: pathlib.Path
    ):
        config: CGLDatasetConfig = self.config  # type: ignore
        processor: CGLProcessor = config.processor

        ann_json = processor.load_annotation_json(annotation_path)
        categories = processor.load_categories_data(
            category_dicts=ann_json["categories"],
            rename_category_names=config.rename_category_names,
        )
        images = processor.load_images_data(
            image_dicts=ann_json["images"],
        )
        annotations = processor.load_data(
            ann_dicts=ann_json["annotations"],
        )
        assert len(images) == len(annotations)

        generator = processor.generate_examples(
            images=images,
            image_files=image_files,
            categories=categories,
            annotations=annotations,
        )

        def _generate_default(generator):
            for idx, example in generator:
                yield idx, example

        def _generate_ralf_style(generator):
            from ralfpt.saliency_detection import SaliencyTester
            from simple_lama_inpainting import SimpleLama

            inpainter = SimpleLama()
            saliency_testers = [
                SaliencyTester(model_name=model) for model in config.saliency_testers
            ]

            for idx, example in generator:
                example = ralf_style_example(
                    example,
                    inpainter=inpainter,
                    saliency_map_cols=config.saliency_maps,
                    saliency_testers=saliency_testers,
                )
                yield idx, example

        if config.name == "default":
            yield from _generate_default(generator)

        elif config.name == "ralf":
            yield from _generate_ralf_style(generator)

        else:
            raise ValueError(f"Invalid config name: {config.name}")
