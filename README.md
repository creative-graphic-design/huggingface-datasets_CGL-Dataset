---
annotations_creators:
- crowdsourced
language:
- zh
language_creators:
- found
license:
- cc-by-nc-sa-4.0
multilinguality:
- monolingual
pretty_name: CGL-Dataset
size_categories: []
source_datasets:
- original
tags:
- graphic-design
- layout-generation
- poster-generation
task_categories:
- other
task_ids: []
--- 

# Dataset Card for CGL-Dataset

[![CI](https://github.com/creative-graphic-design/huggingface-datasets_CGL-Dataset/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets_CGL-Dataset/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/creative-graphic-design/huggingface-datasets_CGL-Dataset/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-datasets_CGL-Dataset/actions/workflows/push_to_hub.yaml)

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/minzhouGithub/CGL-GAN
- **Repository:** https://github.com/creative-graphic-design/huggingface-datasets_CGL-Dataset
- **Paper (Preprint):** https://arxiv.org/abs/2205.00303
- **Paper (IJCAI2022):** https://www.ijcai.org/proceedings/2022/692

### Dataset Summary

The CGL-Dataset is a dataset used for the task of automatic graphic layout design for advertising posters. It contains 61,548 samples and is provided by Alibaba Group.

### Supported Tasks and Leaderboards

The task is to generate high-quality graphic layouts for advertising posters based on clean product images and their visual contents. The training set and validation set are collections of 60,548 e-commerce advertising posters, with manual annotations of the categories and positions of elements (such as logos, texts, backgrounds, and embellishments on the posters). Note that the validation set also consists of posters, not clean product images. The test set contains 1,000 clean product images without graphic elements such as logos or texts, consistent with real application data.

### Languages

[More Information Needed]

<!-- Provide a brief overview of the languages represented in the dataset. Describe relevant details about specifics of the language such as whether it is social media text, African American English,...

When relevant, please provide [BCP-47 codes](https://tools.ietf.org/html/bcp47), which consist of a [primary language subtag](https://tools.ietf.org/html/bcp47#section-2.2.1), with a [script subtag](https://tools.ietf.org/html/bcp47#section-2.2.3) and/or [region subtag](https://tools.ietf.org/html/bcp47#section-2.2.4) if available. -->

## Dataset Structure

### Data Instances

[More Information Needed]

<!-- Provide an JSON-formatted example and brief description of a typical instance in the dataset. If available, provide a link to further examples.

```
{
  'example_field': ...,
  ...
}
```

Provide any additional information that is not covered in the other sections about the data here. In particular describe any relationships between data points and if these relationships are made explicit. -->

### Data Fields

[More Information Needed]

<!-- List and describe the fields present in the dataset. Mention their data type, and whether they are used as input or output in any of the tasks the dataset currently supports. If the data has span indices, describe their attributes, such as whether they are at the character level or word level, whether they are contiguous or not, etc. If the datasets contains example IDs, state whether they have an inherent meaning, such as a mapping to other datasets or pointing to relationships between data points.

- `example_field`: description of `example_field`

Note that the descriptions can be initialized with the **Show Markdown Data Fields** output of the [Datasets Tagging app](https://huggingface.co/spaces/huggingface/datasets-tagging), you will then only need to refine the generated descriptions. -->

### Data Splits

[More Information Needed]

<!-- Describe and name the splits in the dataset if there are more than one.

Describe any criteria for splitting the data, if used. If there are differences between the splits (e.g. if the training annotations are machine-generated and the dev and test ones are created by humans, or if different numbers of annotators contributed to each example), describe them here.

Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:

|                         | train | validation | test |
|-------------------------|------:|-----------:|-----:|
| Input Sentences         |       |            |      |
| Average Sentence Length |       |            |      | -->

## Dataset Creation

### Curation Rationale

[More Information Needed]

<!-- What need motivated the creation of this dataset? What are some of the reasons underlying the major choices involved in putting it together? -->

### Source Data

[More Information Needed]

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences,...) -->

#### Initial Data Collection and Normalization

[More Information Needed]

<!-- Describe the data collection process. Describe any criteria for data selection or filtering. List any key words or search terms used. If possible, include runtime information for the collection process.

If data was collected from other pre-existing datasets, link to source here and to their [Hugging Face version](https://huggingface.co/datasets/dataset_name).

If the data was modified or normalized after being collected (e.g. if the data is word-tokenized), describe the process and the tools used. -->

#### Who are the source language producers?

[More Information Needed]

<!-- State whether the data was produced by humans or machine generated. Describe the people or systems who originally created the data.

If available, include self-reported demographic or identity information for the source data creators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

Describe the conditions under which the data was created (for example, if the producers were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here.

Describe other people represented or mentioned in the data. Where possible, link to references for the information. -->

### Annotations

[More Information Needed]

<!-- If the dataset contains annotations which are not part of the initial data collection, describe them in the following paragraphs. -->

#### Annotation process

[More Information Needed]

<!-- If applicable, describe the annotation process and any tools used, or state otherwise. Describe the amount of data annotated, if not all. Describe or reference annotation guidelines provided to the annotators. If available, provide interannotator statistics. Describe any annotation validation processes. -->

#### Who are the annotators?

[More Information Needed]

<!-- If annotations were collected for the source data (such as class labels or syntactic parses), state whether the annotations were produced by humans or machine generated.

Describe the people or systems who originally created the annotations and their selection criteria if applicable.

If available, include self-reported demographic or identity information for the annotators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

Describe the conditions under which the data was annotated (for example, if the annotators were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here. -->

### Personal and Sensitive Information

[More Information Needed]

<!-- State whether the dataset uses identity categories and, if so, how the information is used. Describe where this information comes from (i.e. self-reporting, collecting from profiles, inferring, etc.). See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender. State whether the data is linked to individuals and whether those individuals can be identified in the dataset, either directly or indirectly (i.e., in combination with other data).

State whether the dataset contains other data that might be considered sensitive (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history).  

If efforts were made to anonymize the data, describe the anonymization process. -->

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

<!-- Please discuss some of the ways you believe the use of this dataset will impact society.

The statement should include both positive outlooks, such as outlining how technologies developed through its use may improve people's lives, and discuss the accompanying risks. These risks may range from making important decisions more opaque to people who are affected by the technology, to reinforcing existing harmful biases (whose specifics should be discussed in the next section), among other considerations.

Also describe in this section if the proposed dataset contains a low-resource or under-represented language. If this is the case or if this task has any impact on underserved communities, please elaborate here. -->

### Discussion of Biases

[More Information Needed]

<!-- Provide descriptions of specific biases that are likely to be reflected in the data, and state whether any steps were taken to reduce their impact.

For Wikipedia text, see for example [Dinan et al 2020 on biases in Wikipedia (esp. Table 1)](https://arxiv.org/abs/2005.00614), or [Blodgett et al 2020](https://www.aclweb.org/anthology/2020.acl-main.485/) for a more general discussion of the topic.

If analyses have been run quantifying these biases, please add brief summaries and links to the studies here. -->

### Other Known Limitations

[More Information Needed]

<!-- If studies of the datasets have outlined other limitations of the dataset, such as annotation artifacts, please outline and cite them here. -->

## Additional Information

### Dataset Curators

[More Information Needed]

<!-- List the people involved in collecting the dataset and their affiliation(s). If funding information is known, include it here. -->

### Licensing Information

[More Information Needed]

<!-- Provide the license and link to the license webpage if available. -->

### Citation Information

```bibtex
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
```

### Contributions

<!-- TODO: Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset. -->
