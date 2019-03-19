import unicodedata
from .bert import TOKEN_CLS, TOKEN_SEP, TOKEN_UNK


class Tokenizer(object):

    def __init__(self,
                 token_dict,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 pad_index=0):
        self.token_dict = token_dict
        self.token_cls = token_cls
        self.token_sep = token_sep
        self.token_unk = token_unk
        self.pad_index = pad_index

    def tokenize(self, first, second=None):
        tokens = [self.token_cls] + self._tokenize(first) + [self.token_sep]
        if second is not None:
            tokens += self._tokenize(second) + [self.token_sep]
        return tokens

    def encode(self, first, second=None, max_len=None):
        tokens = self.tokenize(first, second)
        unknown_index = self.token_dict.get(self.token_unk)
        indices = [self.token_dict.get(token, unknown_index) for token in tokens]
        sep = len(tokens)
        for i in range(len(tokens)):
            if tokens[i] == self.token_sep:
                sep = i + 1
                break
        segments = [0] * sep + [1] * (len(tokens) - sep)
        if max_len is not None:
            if len(indices) < max_len:
                indices += [self.pad_index] * (max_len - len(indices))
                segments += [1] * (max_len - len(segments))
            elif len(indices) > max_len:
                indices = indices[:max_len]
                segments = segments[:max_len]
        return indices, segments

    def _tokenize(self, text):
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
        if word in self.token_dict:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.token_dict:
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
        return unicodedata.category(ch).startswith('C')
