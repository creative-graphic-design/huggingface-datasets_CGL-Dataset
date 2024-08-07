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
import pathlib
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Type

import datasets as ds
from datasets.utils.logging import get_logger
from hfcocoapi.models import ImageData
from hfcocoapi.processors import CaptionsProcessor
from hfcocoapi.typehint import Bbox, ImageId, JsonDict
from pydantic import BaseModel

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


class CGLAnnotationsData(BaseModel):
    image_id: int
    area: int
    bbox: Bbox
    categpry_id: Optional[int] = None


class CGLProcessor(CaptionsProcessor):
    def get_features_base_dict(self):
        return {
            "image_id": ds.Value("int64"),
            "image": ds.Image(),
            "file_name": ds.Value("string"),
            "width": ds.Value("int64"),
            "height": ds.Value("int64"),
        }

    def load_data(
        self,
        ann_dicts: List[List[JsonDict]],
        images: Dict[ImageId, ImageData],
        cgl_annotation_class: Type[CGLAnnotationsData] = CGLAnnotationsData,
        tqdm_desc: str = "Load CGL annotation data",
        **kwargs,
    ) -> Dict[ImageId, List[CGLAnnotationsData]]:
        ann_dict_list: List[JsonDict] = list(chain.from_iterable(ann_dicts))

        annotations: Dict[ImageId, List[CGLAnnotationsData]]
        annotations = super().load_data(  # type: ignore
            ann_dicts=ann_dict_list,
            captions_annotation_class=cgl_annotation_class,  # type: ignore
            tqdm_desc=tqdm_desc,
            **kwargs,
        )

        breakpoint()

        return annotations


@dataclass
class CGLDatasetConfig(ds.BuilderConfig):
    processor: CGLProcessor = CGLProcessor()


class CGLDataset(ds.GeneratorBasedBuilder):
    """A class for loading CGL-Dataset dataset."""

    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetConfig
    BUILDER_CONFIGS = [
        CGLDatasetConfig(version=VERSION, description=_DESCRIPTION),
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
        config: CGLDatasetConfig = self.config  # type: ignore
        processor: CGLProcessor = config.processor

        ann_json = processor.load_annotation_json(
            ann_file_path=annotation_path,
        )
        images = processor.load_images_data(
            image_dicts=ann_json["images"],
        )
        categories = processor.load_categories_data(
            category_dicts=ann_json["categories"]
        )
        annotations = processor.load_data(
            ann_dicts=ann_json["annotations"], images=images
        )
        yield from processor.generate_examples(
            images=images,
            annotations=annotations,  # type: ignore
            categories=categories,
        )
