import unicodedata
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP, TOKEN_UNK


class Tokenizer(object):

    def __init__(self,
                 token_dict,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 pad_index=0,
                 cased=False):
        """Initialize tokenizer.

        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self._cased = cased

    @staticmethod
    def _truncate(first_tokens, second_tokens=None, max_len=None):
        if max_len is None:
            return

        if second_tokens is not None:
            while True:
                total_len = len(first_tokens) + len(second_tokens)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(first_tokens) > len(second_tokens):
                    first_tokens.pop()
                else:
                    second_tokens.pop()
        else:
            del first_tokens[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]

    def _pack(self, first_tokens, second_tokens=None):
        first_packed_tokens = [self._token_cls] + first_tokens + [self._token_sep]
        if second_tokens is not None:
            second_packed_tokens = second_tokens + [self._token_sep]
            return first_packed_tokens + second_packed_tokens, len(first_packed_tokens), len(second_packed_tokens)
        else:
            return first_packed_tokens, len(first_packed_tokens), 0

    def _convert_tokens_to_ids(self, tokens):
        unk_id = self._token_dict.get(self._token_unk)
        return [self._token_dict.get(token, unk_id) for token in tokens]

    def tokenize(self, first, second=None):
        """Split text to tokens.

        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        tokens, _, _ = self._pack(first_tokens, second_tokens)
        return tokens

    def encode(self, first, second=None, max_len=None):
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        self._truncate(first_tokens, second_tokens, max_len)
        tokens, first_len, second_len = self._pack(first_tokens, second_tokens)

        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * first_len + [1] * second_len

        if max_len is not None:
            pad_len = max_len - first_len - second_len
            token_ids += [self._pad_index] * pad_len
            segment_ids += [0] * pad_len

        return token_ids, segment_ids

    def decode(self, ids):
        sep = ids.index(self._token_dict[self._token_sep])
        try:
            stop = ids.index(self._pad_index)
        except ValueError as e:
            stop = len(ids)
        tokens = [self._token_dict_inv[i] for i in ids]
        first = tokens[1:sep]
        if sep < stop - 1:
            second = tokens[sep + 1:stop - 1]
            return first, second
        return first

    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

    def _word_piece_tokenize(self, word):
        if word in self._token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

    @staticmethod
    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_space(ch):
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def rematch(text, tokens, cased=False, unknown_token=TOKEN_UNK):
        """Try to find the indices of tokens in the original text.

        >>> Tokenizer.rematch("All rights reserved.", ["all", "rights", "re", "##ser", "##ved", "."])
        [(0, 3), (4, 10), (11, 13), (13, 16), (16, 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.", ["all", "rights", "re", "##ser", "[UNK]", "."])
        [(0, 3), (4, 10), (11, 13), (13, 16), (16, 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.", ["[UNK]", "rights", "[UNK]", "##ser", "[UNK]", "[UNK]"])
        [(0, 3), (4, 10), (11, 13), (13, 16), (16, 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.", ["[UNK]", "righs", "[UNK]", "ser", "[UNK]", "[UNK]"])
        [(0, 3), (4, 10), (11, 13), (13, 16), (16, 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.",
        ...                  ["[UNK]", "rights", "[UNK]", "[UNK]", "[UNK]", "[UNK]"])  # doctest:+ELLIPSIS
        [(0, 3), (4, 10), (11, ... 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.", ["all rights", "reserved", "."])
        [(0, 10), (11, 19), (19, 20)]
        >>> Tokenizer.rematch("All rights reserved.", ["all rights", "reserved", "."], cased=True)
        [(0, 10), (11, 19), (19, 20)]
        >>> Tokenizer.rematch("#hash tag ##", ["#", "hash", "tag", "##"])
        [(0, 1), (1, 5), (6, 9), (10, 12)]
        >>> Tokenizer.rematch("嘛呢，吃了吗？", ["[UNK]", "呢", "，", "[UNK]", "了", "吗", "？"])
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        >>> Tokenizer.rematch("  吃了吗？    ", ["吃", "了", "吗", "？"])
        [(2, 3), (3, 4), (4, 5), (5, 6)]

        :param text: Original text.
        :param tokens: Decoded list of tokens.
        :param cased: Whether it is cased.
        :param unknown_token: The representation of unknown token.
        :return: A list of tuples represents the start and stop locations in the original text.
        """
        decoded, token_offsets = '', []
        for token in tokens:
            token_offsets.append([len(decoded), 0])
            if token == unknown_token:
                token = '#'
            if not cased:
                token = token.lower()
            if len(token) > 2 and token.startswith('##'):
                token = token[2:]
            elif len(decoded) > 0:
                token = ' ' + token
                token_offsets[-1][0] += 1
            decoded += token
            token_offsets[-1][1] = len(decoded)

        heading = 0
        text = text.rstrip()
        for i in range(len(text)):
            if not Tokenizer._is_space(text[i]):
                break
            heading += 1
        text = text[heading:]
        len_text, len_decode = len(text), len(decoded)
        costs = [[0] * (len_decode + 1) for _ in range(2)]
        paths = [[(-1, -1)] * (len_decode + 1) for _ in range(len_text + 1)]
        curr, prev = 0, 1

        for j in range(len_decode + 1):
            costs[curr][j] = j
        for i in range(1, len_text + 1):
            curr, prev = prev, curr
            costs[curr][0] = i
            ch = text[i - 1]
            if not cased:
                ch = ch.lower()
            for j in range(1, len_decode + 1):
                costs[curr][j] = costs[prev][j - 1]
                paths[i][j] = (i - 1, j - 1)
                if ch != decoded[j - 1]:
                    costs[curr][j] = costs[prev][j - 1]
                    paths[i][j] = (i - 1, j - 1)
                    if costs[prev][j] < costs[curr][j]:
                        costs[curr][j] = costs[prev][j]
                        paths[i][j] = (i - 1, j)
                    if costs[curr][j - 1] < costs[curr][j]:
                        costs[curr][j] = costs[curr][j - 1]
                        paths[i][j] = (i, j - 1)
                    costs[curr][j] += 1

        matches = [0] * (len_decode + 1)
        position = (len_text, len_decode)
        while position != (-1, -1):
            i, j = position
            matches[j] = i
            position = paths[i][j]

        intervals = [[matches[offset[0]], matches[offset[1]]] for offset in token_offsets]
        for i, interval in enumerate(intervals):
            token_a, token_b = text[interval[0]:interval[1]], tokens[i]
            if len(token_b) > 2 and token_b.startswith('##'):
                token_b = token_b[2:]
            if not cased:
                token_a, token_b = token_a.lower(), token_b.lower()
            if token_a == token_b:
                continue
            if i == 0:
                border = 0
            else:
                border = intervals[i - 1][1]
            for j in range(interval[0] - 1, border - 1, -1):
                if Tokenizer._is_space(text[j]):
                    break
                interval[0] -= 1
            if i + 1 == len(intervals):
                border = len_text
            else:
                border = intervals[i + 1][0]
            for j in range(interval[1], border):
                if Tokenizer._is_space(text[j]):
                    break
                interval[1] += 1
        intervals = [(interval[0] + heading, interval[1] + heading) for interval in intervals]
        return intervals
