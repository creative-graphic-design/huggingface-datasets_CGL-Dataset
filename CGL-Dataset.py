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
# TODO: Address all TODOs and remove all explanatory comments
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Union

import datasets as ds
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage
from tqdm.auto import tqdm

JsonDict = Dict[str, Any]
ImageId = int
CategoryId = int
AnnotationId = int
Bbox = Tuple[float, float, float, float]

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

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "layout_images": [
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_1.zip",
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_2.zip",
    ],
    "layout_json": "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_json_yinhe.zip",
}


@dataclass
class ImageData(object):
    image_id: ImageId
    file_name: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            file_name=json_dict["file_name"],
            width=json_dict["width"],
            height=json_dict["height"],
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class CategoryData(object):
    category_id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CategoryData":
        return cls(
            category_id=json_dict["id"],
            name=json_dict["name"],
            supercategory=json_dict["supercategory"],
        )


@dataclass
class AnnotationData(object):
    area: float
    bbox: Bbox
    category_id: CategoryId
    image_id: ImageId

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "AnnotationData":
        return cls(
            image_id=json_dict["image_id"],
            area=json_dict["area"],
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
        )


def load_imgs_data(
    img_dicts: List[JsonDict],
    tqdm_desc="Load images",
) -> List[ImageData]:
    images = [
        ImageData.from_dict(img_dict) for img_dict in tqdm(img_dicts, desc=tqdm_desc)
    ]
    return images


def load_ann_data(
    ann_dicts: List[List[JsonDict]],
    tqdm_desc: str = "Load annotation data",
) -> List[List[AnnotationData]]:
    annotations = [
        [AnnotationData.from_dict(ann_dict) for ann_dict in ann_dict_list]
        for ann_dict_list in tqdm(ann_dicts, desc=tqdm_desc)
    ]
    return annotations


def load_categories_data(
    category_dicts: List[JsonDict],
    tqdm_desc: str = "Load categories",
) -> Dict[CategoryId, CategoryData]:
    categories = {}
    for category_dict in tqdm(category_dicts, desc=tqdm_desc):
        category_data = CategoryData.from_dict(category_dict)
        categories[category_data.category_id] = category_data
    return categories


def load_image(filepath: Union[str, pathlib.Path]) -> PilImage:
    logger.info(f'Loading image from "{filepath}"')
    return Image.open(filepath)


def load_json(filepath: Union[str, pathlib.Path]) -> JsonDict:
    logger.info(f'Loading JSON from "{filepath}"')
    with open(filepath, "r") as f:
        json_dict = json.load(f)
    return json_dict


class CGLDataset(ds.GeneratorBasedBuilder):
    """A class for loading CGL-Dataset dataset."""

    VERSION = ds.Version("1.0.0")

    BUILDER_CONFIGS = [
        ds.BuilderConfig(version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "image_id": ds.Value("int64"),
                "image": ds.Image(),
                "file_name": ds.Value("string"),
                "width": ds.Value("int64"),
                "height": ds.Value("int64"),
                "annotations": ds.Sequence(
                    {
                        "area": ds.Value("int64"),
                        "bbox": ds.Sequence(ds.Value("int64")),
                        "category": {
                            "category_id": ds.Value("int64"),
                            "name": ds.Value("string"),
                            "supercategory": ds.Value("string"),
                        },
                    }
                ),
            }
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
        ann_json = load_json(annotation_path)

        categories = load_categories_data(category_dicts=ann_json["categories"])

        imgs = load_imgs_data(img_dicts=ann_json["images"])
        anns = load_ann_data(ann_dicts=ann_json["annotations"])
        assert len(imgs) == len(anns)

        for idx, (img_data, ann_list) in enumerate(zip(imgs, anns)):
            img_path = image_files[img_data.file_name]
            img = Image.open(img_path)

            example = asdict(img_data)
            example["image"] = img

            example["annotations"] = []
            for ann in ann_list:
                ann_dict = asdict(ann)
                category = categories[ann.category_id]
                ann_dict["category"] = asdict(category)
                example["annotations"].append(ann_dict)

            yield idx, example
