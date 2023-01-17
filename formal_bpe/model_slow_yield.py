from collections import defaultdict
from formal_bpe.utils import pairs_in_list, flat_seq, debug_flat_seq
from typing import Dict, List, Tuple
from rich.progress import track

class SlowBPEYield:
    def __init__(self, fix_overlap=False, tokenize=False):
        if fix_overlap:
            self.get_word_pair_counts = self.get_word_pair_counts_fix_overlap

        self.tokenize = tokenize

    @staticmethod
    def apply_merge_slow(tokens_freqs, pair):
        for token_id, (token, token_freq) in tokens_freqs.items():
            ys_word = []
            i = 0
            N = len(token)
            while i < N:
                if i < N - 1 and (token[i], token[i + 1]) == pair:
                    ys_word.append(pair[0]+pair[1])
                    i += 2
                else:
                    ys_word.append(token[i])
                    i += 1
            tokens_freqs[token_id] = [ys_word, token_freq]
        return tokens_freqs

    @staticmethod
    def get_word_pair_counts(tokens_freqs):
        pairs = defaultdict(int)
        for token, token_freq in tokens_freqs.values():
            for (x, y) in pairs_in_list(token):
                pairs[x, y] += token_freq

        return pairs

    @staticmethod
    def get_word_pair_counts_fix_overlap(tokens_freqs):
        raise NotImplemented()
        pairs = defaultdict(int)
        prev_pair = None
        for token, token_freq in tokens_freqs.values():
            for (x, y) in pairs_in_list(token):
                # increment only if the prev suffix does not match our prefix
                # otherwise wrong estimate on `aaa`
                if (x,y) != prev_pair:
                    pairs[x, y] += token_freq
                    prev_pair = (x, y)
                else:
                    # make sure to clear it so that for `aaaa` we count it twice
                    prev_pair = None

        return pairs

    @staticmethod
    def top_pair(pairs):
        return max(pairs, key=pairs.__getitem__)

    @staticmethod
    def token_dictionary(tokens) -> Tuple[List, Dict[int, Tuple[str, int]]]:
        tokens_dict = {}
        tokens_freqs = {}
        tokens_ids = []
        # first id is 1 (reserve 0 for something else)
        i = 1
        for line in tokens:
            token_ids_line = []
            for token in line:
                if token not in tokens_dict:
                    tokens_dict[token] = i
                    tokens_freqs[i] = [token, 0]
                    token_id = i
                    i += 1
                else:
                    token_id = tokens_dict[token]
                
                tokens_freqs[token_id][1] += 1
                token_ids_line.append(token_id)

            tokens_ids.append(token_ids_line)

        return tokens_ids, tokens_freqs

    def fit_greedy(self, tokens, T, debug_output=False, progress_bar=False, indecision_output=False):
        if not self.tokenize:
            # treat the whole line as one word
            tokens = [[line.replace(" ", "▁")] for line in tokens.split("\n")]
        else:
            # split to individual words (by spaces)
            tokens = [line.replace(" ", " ▁").split(" ") for line in tokens.split("\n")]
        
        tokens_ids, tokens_freqs = self.token_dictionary(tokens)
        indecision = False

        iterator = track(range(T)) if progress_bar else range(T)
        for t in iterator:
            pairs = self.get_word_pair_counts(tokens_freqs)
            if len(pairs) == 0:
                break
            pair = self.top_pair(pairs)
            top_pair_values = sorted(pairs.values(), reverse=True)[:2]
            if len(top_pair_values) >= 2 and top_pair_values[0] == top_pair_values[1]:
                indecision = True

            # this mutates tokens_freqs
            self.apply_merge_slow(tokens_freqs, pair)

        if debug_output:
            flattener = debug_flat_seq
        else:
            flattener = flat_seq

        output = [
            [flattener(x) for word_id in line_ids for x in tokens_freqs[word_id][0]]
            for line_ids in tokens_ids
        ]

        if indecision_output:
            return output, indecision
        else:
            return output