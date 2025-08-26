from collections import defaultdict

merge = []

token_to_id = []

id_to_token = []


def compare_pair_freqs(splits, word_freqs):
    """
    统计最常见字符串

    Args:
        splits (字典):key是加过"</w>"的word;value将key强力拆分.如,<'hello</w>': ['h', 'e', 'l', 'l', 'o', '<', '/', 'w', '>']>
        word_freqs (字典):key是加过"</w>"的word;value是统计过的，在原text的词出现的频率，如<hello,3>

    Returns:
        pair_freqs(字典): key是相邻字符串元组，value是该元组在text出现的频率。
    """

    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pairs(a, b, splits):
    """
    最长见字符串合并

    Args:
        a (字符串): 最佳字符串A
        b (字符串): 最佳字符串B
        splits (字典): key是加过"</w>"的word;value将key强力拆分.如,<'hello</w>': ['h', 'e', 'l', 'l', 'o', '<', '/', 'w', '>']>
    Returns:
        splits (字典): key是加过"</w>"的word;value将key强力拆分，但是合并过A+B的新值.如,<'hello</w>': ['he', 'll', 'o', '<', '/', 'w', '>']>
    """
    for word in splits:
        split = splits[word]
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and a == split[i] and b == split[i + 1]:
                new_split.append(a + b)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        splits[word] = new_split
    return splits


# 函数：训练！重复“找最常见字符对 → 合并”这个过程，epoch == ？
def bpe_train(splits, words_freq, epoch):
    for i in range(epoch):
        pair_freqs = compare_pair_freqs(splits=splits, word_freqs=words_freq)
        if len(pair_freqs) == 0:
            print("没有可以合并的字符对")
            break
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]
        print(
            f"第{i + 1}次合并：把'{best_pair[0]}和'{best_pair[1]}'合并成{best_pair[0]+best_pair[1]},共出现{best_freq}次！"
        )
        splits = merge_pairs(best_pair[0], best_pair[1], splits=splits)
        merge.append(best_pair)
    return splits


# 构建最终词汇表（Vocabulary）
def construct_vocabulary(splits, words_freq_with_end):
    vocabulary = set()

    for split in splits.values():
        for token in split:
            vocabulary.add(token)
    all_chars = set()
    for word in words_freq_with_end:
        for c in word:
            all_chars.add(c)

    vocabulary.update(all_chars)
    vocabulary = sorted(vocabulary)
    vocabulary.append("<UNK>")

    return vocabulary


def construct_tokens_ids(vocabulary):
    token_to_id = {token: id for id, token in enumerate(vocabulary)}
    id_to_token = {id: token for token, id in token_to_id.items()}
    return token_to_id, id_to_token


def encode(text):
    if not text.endswith("</w>"):
        text += "</w>"

    tokens = []
    i = 0
    n = len(text)

    while i < n:
        matched = False
        for j in range(n + 1, i, -1):
            candidate = text[i:j]
            if candidate in token_to_id:
                tokens.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            tokens.append("<UNK>")
            i += 1

    return [token_to_id[token] for token in tokens]


def decode(ids):
    tokens = [id_to_token[id] for id in ids]
    raw = "".join(tokens)
    if raw.endswith("</w>"):
        raw = raw[:-4]
    return raw


def main():
    text = """
    hello world hello
    how are you
    i am fine thank you
    hello python world
    """
    text = text.lower()
    words = text.strip().split()

    word_freqs = defaultdict(int)

    for word in words:
        word_freqs[word] += 1

    word_freqs_with_end = {}
    for word, freq in word_freqs.items():
        word_with_end = word + "</w>"
        word_freqs_with_end[word_with_end] = freq

    splits = {}
    for word in word_freqs_with_end:
        splits[word] = list(word)

    splits = bpe_train(splits=splits, words_freq=word_freqs_with_end, epoch=10)

    vocabulary = construct_vocabulary(
        splits=splits, words_freq_with_end=word_freqs_with_end
    )

    print(vocabulary)

    global token_to_id, id_to_token
    token_to_id, id_to_token = construct_tokens_ids(vocabulary=vocabulary)

    test_words = ["hello", "world", "how", "are", "python", "fine"]

    for word in test_words:
        # 编码
        encoded_ids = encode(word)
        # 解码
        decoded_text = decode(encoded_ids)

        print(f"原始: '{word}'")
        print(f"编码: {encoded_ids}")
        print(f"tokens: {[id_to_token[tid] for tid in encoded_ids]}")
        print(f"解码: '{decoded_text}'")
        print(f"一致: {word == decoded_text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
