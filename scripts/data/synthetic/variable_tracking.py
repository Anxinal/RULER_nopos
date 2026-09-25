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
Create a dataset jsonl file for variable tracking.

python variable_tracking.py   \
    --save_dir=./ \
    --save_name=vt \
    --tokenizer_path='EleutherAI/gpt-neox-20b' \
    --tokenizer_type hf \
    --max_seq_length 4096 \
    --tokens_to_generate 30 \
    --num_samples 10 \
    --random_seed 42  \
    --num_chains 1  --num_hops 4 \
    --template "[INST] Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above. [/INST] Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: "
"""
import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import string
from constants import TASKS
import sys
import pdb
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizer import select_tokenizer
from manifest_utils import write_manifest
from nltk.tokenize import sent_tokenize
import numpy as np
import heapq
import json
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, default=120, help='number of tokens to generate')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default='', help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

parser.add_argument("--type_haystack", type=str, default='noise', help='[Options] noise or essay.')
parser.add_argument("--num_chains", type=int, default=1, help='number of inserted variable chains')
parser.add_argument("--num_hops", type=int, default=4, help='number of hops in each chain')
parser.add_argument("--add_fewshot", action="store_true", default=False)
parser.add_argument("--model_template_token", type=int, default=0, help='used for nemo skills, minus num of model template token')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Define Needle/Haystack Format
if args.type_haystack == 'essay':
    essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/PaulGrahamEssays.json")
    essay = json.load(open(essay))['text']
    haystack = re.sub(r'\s+', " ", essay).split(" ")
elif args.type_haystack == 'noise':
    haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
else:
    raise NotImplementedError(f'{args.type_haystack} is not implemented.')

# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

def generate_chains(num_chains, num_hops, is_icl=False):

    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    need = (num_hops+1) * num_chains

    # Draw DISTINCT names, keeping the list at exactly `need`.
    #
    # This used to draw `need` names at once and then, on a collision, append more until
    # the SET was large enough -- which left the LIST longer than `need`. The grouping
    # loop below walks that list with a fixed stride of num_hops+1, so the extra entries
    # formed a final short group and `this_vars[j+1]` raised IndexError. Any collision
    # was therefore a hard crash of data generation, not a degraded sample.
    #
    # It went unnoticed because the odds scale with the square of the name count. The
    # ICL example that main() builds first uses k=3 (17,576 possible names), so at the
    # RULER default of one chain it is a ~3% flake across a whole run, but at four
    # chains it is ~43% and at eight ~90%.
    #
    # Distinctness is required, not cosmetic: a name repeated across two chains would
    # make the assignment graph ambiguous, and one repeated inside a chain would make
    # the chain self-referential.
    space = 26 ** k
    if need > space // 2:
        # Rejection sampling degrades badly as the space fills, and past it the loop
        # cannot terminate at all. Fail with the arithmetic rather than hang.
        raise RuntimeError(
            f"variable_tracking: {num_chains} chains of {num_hops} hops need {need} "
            f"distinct {k}-letter names, which is too many to draw from {space}. "
            f"Lower --num_chains/--num_hops."
        )
    vars_all = []
    seen = set()
    while len(vars_all) < need:
        # Redrawing only on a collision keeps the RNG stream identical to the old code
        # whenever the old code would have succeeded, so seeds reproduce byte for byte.
        var = ''.join(random.choices(string.ascii_uppercase, k=k)).upper()
        # "VAR" itself is excluded: it is the keyword every assignment starts with, so a
        # variable of that name makes the text unparseable for anything that reads the
        # chains back, randomize_icl included.
        if var in seen or var == "VAR":
            continue
        seen.add(var)
        vars_all.append(var)

    # DISTINCT root values, one per chain.
    #
    # The query is "find all variables assigned the value V", so two chains sharing a
    # root value means two chains answer it while `answers` returns only chain 0's --
    # the sample is then simply mislabelled. Drawing independently per chain made that
    # a 1-in-90,000 event per pair, which is rare enough never to be noticed and common
    # enough to mislabel a couple of samples in a 25k-sample training set.
    #
    # The is_icl branch was worse than rare, it was certain: every chain was seeded with
    # the literal 12345 so that randomize_icl could find and swap it, and that swap
    # rewrote all of them to the same new value. At RULER's shipped num_chains=1 there
    # is only one root, so neither problem can occur; both appear the moment there are
    # two.
    values_all = []
    seen_values = set()
    while len(values_all) < num_chains:
        value = str(np.random.randint(10000, 99999))
        if value in seen_values:
            continue
        seen_values.add(value)
        values_all.append(value)

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops+1):
        this_vars = vars_all[i:i+num_hops+1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {values_all[i // (num_hops+1)]}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret

def shuffle_sublists_heap(lst):
    heap = []
    for i in range(len(lst)):
        heapq.heappush(heap, (random.random(), i, 0))  # Push first element of each list with random priority
    shuffled_result = []
    while heap:
        _, list_idx, elem_idx = heapq.heappop(heap)  # Get the lowest random priority element
        shuffled_result.append(lst[list_idx][elem_idx])

        # If there are more elements in the same sublist, add the next one
        if elem_idx + 1 < len(lst[list_idx]):
            heapq.heappush(heap, (random.random(), list_idx, elem_idx + 1))
    return shuffled_result

def generate_input_output(num_noises, num_chains, num_hops, is_icl=False):

    vars, chains = generate_chains(num_chains, num_hops, is_icl=is_icl)
    value = chains[0][0].split("=")[-1].strip()

    if args.type_haystack == 'essay':
        text = " ".join(haystack[:num_noises])
        document_sents = sent_tokenize(text.strip())
        chains_flat = shuffle_sublists_heap(chains)
        insertion_positions = [0] + \
                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(chains_flat))]) + \
                              [len(document_sents)]
        document_sents_list = []
        for i in range(1,len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i-1 < len(chains_flat):
                document_sents_list.append(chains_flat[i-1].strip() + ".")
        context = " ".join(document_sents_list)

    elif args.type_haystack == 'noise':
        sentences = [haystack] * num_noises
        for chain in chains:
            positions = list(sorted(random.sample(range(len(sentences)), len(chain))))
            for insert_pi, j in zip(positions, range(len(chain))):
                sentences.insert(insert_pi+j, chain[j])
        context = "\n".join(sentences)

    context = context.replace(". \n", ".\n")

    template = args.template
    if is_icl and template != TASKS['variable_tracking']['template'] + TASKS['variable_tracking']['answer_prefix']:
        # remove model template
        new_template = ""
        if len(TASKS['variable_tracking']['template']) > 0:
            new_template = TASKS['variable_tracking']['template']
        if len(TASKS['variable_tracking']['answer_prefix']) > 0:
            new_template += TASKS['variable_tracking']['answer_prefix']
        template = new_template

    input_text = template.format(
        context=context,
        query=value,
        num_v=num_hops+1
    )

    return input_text, vars[0]

def randomize_icl(icl_example):
    """Re-randomise every surface form in the cached few-shot example.

    The example is built once and prepended to every sample, so this is the only thing
    stopping it from being identical text 25,000 times over. It must therefore rewrite
    *every* variable name and *every* root value, and must do so in a single pass.

    All three of the following were wrong, and all three are invisible at RULER's
    shipped num_chains=1, which is the only variable-tracking configuration the
    benchmark defines:

    * Only the answer chain was renamed. ``split()[-num_hops-1:]`` takes the trailing
      answer, which at one chain is every name in the example; with more chains, every
      distractor chain's names stayed frozen for the life of the dataset.
    * Every root was the literal ``12345`` and a single ``str.replace`` rewrote them all
      to one new value, so the demonstrated question had ``num_chains * (num_hops+1)``
      correct answers while the demonstrated answer listed ``num_hops+1`` of them and
      the sentence above it asserted that was the count.
    * Replacements were applied one at a time, so a freshly drawn name could collide
      with a name still queued for replacement, or be rewritten again by a later
      iteration -- observed in 5 of 1000 records at four chains, producing a variable
      that was both a chain root and a link in a different chain.

    Names keep their original length so the token budget measured in ``main`` still
    holds, and the replacement pass is simultaneous, so a new name colliding with an
    old one is harmless: the old one is being replaced in the same pass and the result
    is never rescanned.
    """
    # Every variable appears after the VAR keyword at least once (a root as
    # "VAR x = <n>", a link as "VAR x = VAR y"), so this finds all of them. The answer
    # line lists names without the keyword, but they are all chain variables and so are
    # already in this set; rewriting on word boundaries catches them there too.
    names = sorted(set(re.findall(r"VAR ([A-Z]+)", icl_example)))
    values = sorted(set(re.findall(r"VAR [A-Z]+ = (\d+)", icl_example)))

    mapping = {}
    used = set()
    for name in names:
        while True:
            new_name = ''.join(random.choices(string.ascii_uppercase, k=len(name))).upper()
            if new_name != "VAR" and new_name not in used:
                break
        used.add(new_name)
        mapping[name] = new_name
    used_values = set()
    for value in values:
        while True:
            new_value = str(np.random.randint(10000, 99999))
            if new_value not in used_values:
                break
        used_values.add(new_value)
        mapping[value] = new_value

    if not mapping:
        return icl_example
    # One simultaneous pass. Word boundaries keep a value from matching inside a longer
    # number and a name from matching inside a longer word, and they let the question
    # line ("assigned the value 38693") and the answer line be rewritten consistently
    # with the assignments.
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in mapping) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(0)], icl_example)

def sys_vartrack_w_noise_random(num_samples: int, max_seq_length: int, incremental: int = 10,
                                num_chains: int = 1, num_hops: int = 4,
                                add_fewshot: bool = True,
                                icl_example: str = None,
                                final_output: bool = False):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate if icl_example is not None else 0
    max_seq_length -= args.model_template_token

    # Find the perfect num_noises
    if icl_example:
        if args.type_haystack == 'essay':
            incremental = 500
        elif args.type_haystack == 'noise':
            incremental = 10

        if args.type_haystack != 'essay' and args.max_seq_length < 4096:
            incremental = 5
    else:
        if args.type_haystack == 'essay':
            incremental = 50
        elif args.type_haystack == 'noise':
            incremental = 5

    example_tokens = 0
    if add_fewshot and (icl_example is not None):
        icl_example_out = ' '.join(icl_example['outputs'])
        icl_example = icl_example['input'] + " " + icl_example_out + '\n'
        example_tokens = len(TOKENIZER.text_to_tokens(icl_example))

    # Estimate tokens per question to determine reasonable upper bound
    sample_input_text, _ = generate_input_output(incremental, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None))
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input_text))
    tokens_per_haystack = sample_tokens / incremental

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
    estimated_max_noises = int((max_seq_length / tokens_per_haystack) * 3)

    # Smallest haystack `generate_input_output` can actually build.
    #
    # The noise branch inserts each chain with
    # `random.sample(range(len(sentences)), len(chain))`, which needs at least one
    # position per link, so the haystack cannot start smaller than a single chain --
    # num_hops+1 units. Below that, random.sample raises "Sample larger than population"
    # rather than returning a shorter sample. Note the floor depends on num_hops only:
    # later chains sample from a list that earlier insertions have already grown.
    #
    # The essay branch has no such floor -- it slices sentences by depth percentage and
    # copes with a short document -- so it keeps 1.
    min_noises = (num_hops + 1) if args.type_haystack == 'noise' else 1

    # Binary search for optimal haystack size.
    # NOTE: the lower bound must be `min_noises`, not `incremental`. If it is
    # `incremental` and even that smallest size overflows the budget (which happens at
    # short --max_seq_length), the search returns nothing, `num_noises` falls back to
    # `incremental`, and the size-reduction loop below can never decrement -- an
    # unrecoverable silent hang. It must not be below `min_noises` either: the search
    # probes small sizes whenever the budget is tight, and every probe below the floor
    # is a ValueError out of random.sample. That is why this surfaced only once
    # num_chains went up -- more chain text leaves less room, so the search reaches
    # further down.
    lower_bound = min_noises
    upper_bound = max(estimated_max_noises, incremental * 2, min_noises)

    optimal_num_noises = None

    logger.info(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, answer = generate_input_output(mid, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None))
        total_tokens = len(TOKENIZER.text_to_tokens(input_text)) + example_tokens + tokens_to_generate

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            # This size works, can we go larger?
            optimal_num_noises = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1

    if optimal_num_noises is None:
        raise RuntimeError(
            f"variable_tracking/{args.save_name}: cannot fit the smallest buildable "
            f"haystack ({min_noises} noise unit(s)) within max_seq_length="
            f"{max_seq_length} with num_chains={num_chains}, num_hops={num_hops}. "
            f"The chains alone are too long for the budget -- raise --max_seq_length "
            f"or lower --num_chains."
        )
    num_noises = optimal_num_noises
    logger.info(f'Final optimal haystack size (number of haystack): {num_noises}')

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_noises = num_noises
        while(True):
            try:
                input_text, answer = generate_input_output(used_noises, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None))
                if add_fewshot and (icl_example is not None):
                    # insert icl_example between model template and input
                    cutoff = input_text.index(TASKS['variable_tracking']['template'][:20])
                    input_text = input_text[:cutoff] + randomize_icl(icl_example) + '\n' + input_text[cutoff:]
                if args.remove_newline_tab:
                    input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                # Decrement toward the smallest buildable haystack and fail loudly rather
                # than spinning forever. The floor is min_noises, not 1: going below it
                # swaps the "too long" ValueError for a "Sample larger than population"
                # one, which this bare except would swallow and retry until the counter
                # bottomed out, reporting a length problem that was really a floor
                # violation.
                if used_noises <= min_noises:
                    raise RuntimeError(
                        f"variable_tracking/{args.save_name}: sample {index} does not fit "
                        f"within max_seq_length={max_seq_length} even at the smallest "
                        f"buildable haystack ({min_noises} noise unit(s))."
                    )
                used_noises = max(min_noises, used_noises - incremental)

        if final_output:
            answer_prefix_index = input_text.rfind(TASKS['variable_tracking']['answer_prefix'][:10]) # use first 10 char of answer prefix to locate it
            answer_prefix = input_text[answer_prefix_index:]
            input_text = input_text[:answer_prefix_index]
            formatted_output = {
                'index': index,
                "input": input_text,
                "outputs": answer,
                "length": length,
                'length_w_model_temp': length + args.model_template_token,
                'answer_prefix': answer_prefix,
            }
        else:
            formatted_output = {
                'index': index,
                "input": input_text,
                "outputs": answer,
                "length": length,
            }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # The few-shot example is prepended to every sample, and its budget was a hardcoded
    # 500 tokens. That is not a property of the example but of how much chain text it
    # has to contain: a minimal one costs 247 tokens at one chain, 404 at four and 627
    # at eight, so from eight chains up it could not be built at all and generation died
    # before writing a line. Size it from the real minimum, with 500 as a floor so every
    # configuration that already fitted keeps generating byte-identical data.
    #
    # The 15% slack matters: variable names are random and a 3-letter name is two or
    # three GPT-2 tokens, so the same configuration varies by a few dozen tokens between
    # draws. A budget set at exactly the measured minimum fails on an unlucky one.
    min_noises = (args.num_hops + 1) if args.type_haystack == 'noise' else 1
    _rng_state, _np_state = random.getstate(), np.random.get_state()
    _probe, _ = generate_input_output(min_noises, args.num_chains, args.num_hops,
                                      is_icl=True)
    # Rewind both RNGs. Measuring must not consume draws, or every sample downstream
    # would shift and the same --random_seed would stop reproducing earlier datasets.
    random.setstate(_rng_state)
    np.random.set_state(_np_state)
    icl_budget = max(500, int(len(TOKENIZER.text_to_tokens(_probe)) * 1.15))
    logger.info(f"Few-shot example budget: {icl_budget} tokens "
                f"(minimum for {args.num_chains} chain(s): {len(TOKENIZER.text_to_tokens(_probe))})")

    icl_example = sys_vartrack_w_noise_random(num_samples=1,
                                              max_seq_length=icl_budget,
                                              incremental=5,
                                              num_chains=args.num_chains,
                                              num_hops=args.num_hops)[0]
    logger.info(icl_example)
    write_jsons = sys_vartrack_w_noise_random(num_samples=args.num_samples,
                                              max_seq_length=args.max_seq_length,
                                              num_chains=args.num_chains,
                                              num_hops=args.num_hops,
                                              icl_example=icl_example,
                                              final_output=True)

    write_manifest(save_file, write_jsons)

if __name__=="__main__":
    main()
