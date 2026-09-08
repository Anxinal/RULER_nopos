# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License

import json


def read_manifest(manifest_path):
    """
    Read a JSONL manifest file into a list of dicts.

    Drop-in replacement for ``nemo.collections.asr.parts.utils.manifest_utils.read_manifest``
    so that the prediction and evaluation paths do not require the full NeMo toolkit
    for what is a six-line JSONL reader.

    Args:
        manifest_path (str or Path): Path to the manifest file.

    Returns:
        list: One dict per non-empty line.

    Raises:
        FileNotFoundError: if the manifest does not exist.
        json.JSONDecodeError: if a line is not valid JSON, with the line number attached.
    """
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (in {manifest_path} at line {lineno})", e.doc, e.pos
                ) from None
    return data


def write_manifest(output_path, target_manifest, ensure_ascii: bool = True):
    """
    Write to manifest file

    Args:
        output_path (str or Path): Path to output manifest file
        target_manifest (list): List of manifest file entries
        ensure_ascii (bool): default is True, meaning the output is guaranteed to have all incoming
                             non-ASCII characters escaped. If ensure_ascii is false, these characters
                             will be output as-is.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')
