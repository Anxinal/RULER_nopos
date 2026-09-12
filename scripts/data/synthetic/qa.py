# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Create a dataset jsonl file for QA task.

python qa.py \
    --save_dir=./ \
    --save_name=niah_single \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --max_seq_length=4096 \
    --tokens_to_generate=128 \
    --num_samples=10 \
    --template="Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query} Answer:"
"""
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizer import select_tokenizer
from manifest_utils import write_manifest
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

from constants import TASKS

parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--pre_samples", type=int, default=0, help='number of samples are already generated')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, required=True, help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--model_template_token", type=int, default=0, help='used for nemo skills, minus num of model template token')
# Complexity Configurations
parser.add_argument("--dataset", type=str, required=True, help='dataset file')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })

    return total_qas, total_docs

# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })

    return total_qas, total_docs


DOCUMENT_PROMPT = "Document {i}:\n{document}"
if args.dataset == 'squad':
    QAS, DOCS = read_squad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/squad.json"))
elif args.dataset == 'hotpotqa':
    QAS, DOCS = read_hotpotqa(os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/hotpotqa.json"))
else:
    raise NotImplementedError(f'{args.dataset} is not implemented.')


def generate_input_output(index, num_docs):
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        # Repeat DOCS as many times as needed and slice to num_docs
        repeats = (num_docs + len(DOCS) - 1) // len(DOCS)  # Ceiling division
        all_docs = (DOCS * repeats)[:num_docs]

    random.Random(args.random_seed).shuffle(all_docs)

    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    input_text = args.template.format(
        context=context,
        query=curr_q
    )
    return input_text, curr_a


def generate_samples(num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 10):

    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    max_seq_length -= args.model_template_token

    # A question's own documents are the smallest haystack that can be built for it;
    # see the note above the generation loop. Probing below that raises ValueError from
    # random.sample rather than producing a shorter sample.
    #
    # The probe must be a question that actually fits. Using QAS[0] unconditionally lets
    # one unlucky first question abort a task that every other question could have
    # satisfied, since document lengths vary a lot between questions.
    PROBE_WINDOW = 200
    probe_index = None
    for candidate in range(args.pre_samples, min(len(QAS), args.pre_samples + PROBE_WINDOW)):
        floor = max(1, len(QAS[candidate]['context']))
        text, _ = generate_input_output(candidate, floor)
        if len(TOKENIZER.text_to_tokens(text)) + tokens_to_generate <= max_seq_length:
            probe_index, probe_floor = candidate, floor
            break

    if probe_index is None:
        raise RuntimeError(
            f"qa/{args.save_name}: none of the first {PROBE_WINDOW} questions fit within "
            f"max_seq_length={max_seq_length} (tokens_to_generate={tokens_to_generate}) "
            f"even reduced to their own documents, which cannot be split. "
            f"Raise --max_seq_length."
        )
    if probe_index != args.pre_samples:
        logger.info(f'Using question {probe_index} to size the haystack (earlier ones do not fit)')

    # Estimate tokens per question to determine reasonable upper bound
    probe_docs = max(incremental, probe_floor)
    sample_input_text, _ = generate_input_output(probe_index, probe_docs)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))
    tokens_per_doc = sample_tokens / probe_docs

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
    estimated_max_docs = int((max_seq_length / tokens_per_doc) * 3)

    # Binary search for optimal haystack size.
    # NOTE: the lower bound must not be `incremental`. If it is, and even that smallest
    # size overflows the budget (which happens at short --max_seq_length, since a single
    # SQuAD/HotpotQA paragraph is already ~100-200 tokens), the search returns nothing,
    # `num_docs` falls back to `incremental`, and the size-reduction loop below can never
    # decrement -- an unrecoverable silent hang. It cannot go below probe_floor either,
    # so that is the bound.
    lower_bound = probe_floor
    upper_bound = max(estimated_max_docs, incremental * 2)  # Ensure upper_bound is reasonable

    optimal_num_docs = None

    logger.info(f"Estimated {tokens_per_doc:.1f} tokens per doc")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, answer = generate_input_output(probe_index, mid)
        total_tokens = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            # This size works, can we go larger?
            optimal_num_docs = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1

    if optimal_num_docs is None:
        raise RuntimeError(
            f"qa/{args.save_name}: the probe question's own {probe_floor} document(s) "
            f"already exceed max_seq_length={max_seq_length} "
            f"(tokens_to_generate={tokens_to_generate}). Raise --max_seq_length."
        )
    num_docs = optimal_num_docs
    logger.info(f'Final optimal haystack size (number of docs): {num_docs}')

    # Generate samples.
    #
    # A question's own documents are a hard floor on how far the haystack can shrink.
    # For HotpotQA, `context` holds all ten distractor paragraphs and the loader keeps
    # no record of which two are the gold ones, so dropping any of them risks removing
    # the answer and producing an unanswerable sample. Below that floor,
    # generate_input_output also computes a negative sample count and raises
    # ValueError: Sample larger than population or is negative.
    #
    # A question whose own documents already exceed the budget is therefore skipped
    # rather than mangled, and the next question is tried instead. Length is compared
    # explicitly rather than via assert-and-bare-except, so a genuine bug propagates
    # with its own traceback instead of being relabelled as a length problem.
    n_skipped = 0
    qa_index = args.pre_samples
    n_questions = len(QAS)
    progress = tqdm(total=num_samples)

    while len(write_jsons) < num_samples and qa_index < n_questions:
        floor = len(QAS[qa_index]['context'])
        used_docs = max(num_docs, floor)
        fits = False

        while True:
            input_text, answer = generate_input_output(qa_index, used_docs)
            length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
            if length <= max_seq_length:
                fits = True
                break
            if used_docs <= floor:
                break
            used_docs = max(floor, used_docs - incremental)

        if not fits:
            n_skipped += 1
            qa_index += 1
            continue

        if args.remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
        answer_prefix_index = input_text.rfind(TASKS['qa']['answer_prefix'][:10]) # use first 10 char of answer prefix to locate it
        answer_prefix = input_text[answer_prefix_index:]
        input_text = input_text[:answer_prefix_index]
        formatted_output = {
            "index": len(write_jsons),
            "input": input_text,
            "outputs": answer,
            "length": length,
            'length_w_model_temp': length + args.model_template_token,
            'answer_prefix': answer_prefix,
        }
        write_jsons.append(formatted_output)
        progress.update(1)
        qa_index += 1

    progress.close()

    if n_skipped:
        logger.warning(
            f'{n_skipped} question(s) skipped: their own documents exceed '
            f'max_seq_length={max_seq_length}.'
        )
    if len(write_jsons) < num_samples:
        raise RuntimeError(
            f"qa/{args.save_name}: only {len(write_jsons)} of {num_samples} samples fit "
            f"within max_seq_length={max_seq_length} after trying all "
            f"{n_questions - args.pre_samples} questions ({n_skipped} skipped). "
            f"Raise --max_seq_length or lower --num_samples."
        )

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir
    )

    write_manifest(save_file, write_jsons)

if __name__=="__main__":
    main()
