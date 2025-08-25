我们来写一个 **完整的、可运行的 BPE 分词器程序**，并为每一行都加上“保姆级”注释。

> ✅ 说明：为了让你能快速运行和理解，我们不用 1GB 的 enwik9，而是用一小段英文句子模拟真实训练过程。

---

## ✅ 完整代码 + 超详细注释

```python
# =============================
# 第一步：导入需要用到的工具包
# =============================

# 我们需要 defaultdict，它可以自动给新键设置默认值（比如 0 或 []）
# 比如：counts['新单词'] += 1，即使这个单词没出现过也不会报错
from collections import defaultdict

# =============================
# 第二步：定义我们要学习的文本（模拟真实语料）
# =============================

# 这是我们要训练 BPE 的“小课本”
# 就像你学英语时背的几句话，模型也从这里学怎么拆词
text = """
hello world hello
how are you
i am fine thank you
hello python world
"""

# =============================
# 第三步：预处理文本（清洗 + 切词 + 统计频率）
# =============================

# 把所有字母变成小写（避免 "Hello" 和 "hello" 被当成两个词）
text = text.lower()

# 把多余的空格、换行符变成一个空格，方便切词
# strip() 去掉开头结尾空格，split() 按空格切开，' '.join() 再用一个空格连起来
text = ' '.join(text.strip().split())

# 把文本按空格切分成一个个单词，比如 ['hello', 'world', 'hello', ...]
words = text.split()

# 创建一个“词频字典”，用来记录每个词出现了几次
# defaultdict(int) 表示：如果这个单词还没记过，默认次数是 0
word_freqs = defaultdict(int)

# 遍历每一个单词，给它的计数 +1
for word in words:
    # 比如第一次遇到 'hello'，word_freqs['hello'] 就从 0 变成 1
    # 第二次遇到，就变成 2，依此类推
    word_freqs[word] += 1

# 但是！BPE 需要知道一个词什么时候结束
# 所以我们给每个词后面加上一个特殊标记 </w>，表示“这个词到此为止”
# 比如 "hello" 变成 "hello</w>"，这样模型就知道这不是 "helloxxx" 的一部分

# 我们创建一个新的字典，保存加了 </w> 的词和它们的频率
word_freqs_with_end = {}
for word, freq in word_freqs.items():
    # 把原来的词 + </w>，比如 "hello" -> "hello</w>"
    word_with_end = word + "</w>"
    # 存进新字典
    word_freqs_with_end[word_with_end] = freq

# 现在我们的“课本”长这样：
# {'hello</w>': 2, 'world</w>': 2, 'how</w>': 1, 'are</w>': 1, ...}

# =============================
# 第四步：把每个词拆成“字符”（BPE 的起点）
# =============================

# 创建一个字典，保存每个词是怎么被拆成字符的
splits = {}

# 遍历每个“加了 </w>”的词
for word in word_freqs_with_end:
    # 把这个词的每一个字母（和</w>）拆开放进一个列表
    # 比如 "hello</w>" → ['h', 'e', 'l', 'l', 'o', '<', '/', 'w', '>']
    split_into_chars = list(word)
    # 把这个拆分结果存起来
    splits[word] = split_into_chars

# 现在 splits 长这样：
# {
#   'hello</w>': ['h','e','l','l','o','<','/','w','>'],
#   'world</w>': ['w','o','r','l','d','<','/','w','>'],
#   ...
# }

# =============================
# 第五步：定义一个函数：统计“哪两个字符经常挨在一起”
# =============================

def compute_pair_freqs(splits, word_freqs):
    """
    这个函数的作用是：
    扫描所有词的字符拆分结果，统计“哪两个字符经常连在一起出现”
    比如：'h' 后面经常跟着 'e'，'e' 后面经常跟着 'l'，等等
    """
    
    # 创建一个字典，用来记录每一对字符出现了多少次
    # 比如 ('h','e'): 2, ('e','l'): 2, ...
    pair_freqs = defaultdict(int)
    
    # 遍历每一个词（比如 'hello</w>'）
    for word, freq in word_freqs.items():
        # 拿到这个词的字符拆分列表，比如 ['h','e','l','l','o','<','/','w','>']
        split = splits[word]
        
        # 遍历这个列表，看每两个“挨着”的字符
        # i 从 0 开始，一直到倒数第二个字符
        for i in range(len(split) - 1):
            # 取出第 i 个字符和第 i+1 个字符，组成一对
            # 比如 i=0: ('h', 'e'), i=1: ('e', 'l'), i=2: ('l', 'l'), ...
            pair = (split[i], split[i+1])
            
            # 这对字符出现了 freq 次（因为这个词出现了 freq 次）
            # 所以给这对字符的总频率加上 freq
            pair_freqs[pair] += freq
    
    # 返回所有字符对的频率
    return pair_freqs

# =============================
# 第六步：定义一个函数：把最常见的两个字符“合并”成一个新符号
# =============================

def merge_pair(a, b, splits):
    """
    把所有连续的 a 和 b 合并成一个新东西：ab
    比如 a='l', b='l' → 合并成 'll'
    """
    
    # 遍历每一个词的字符拆分结果
    for word in splits:
        # 拿到当前词的字符列表
        split = splits[word]
        # 新列表，用来存放合并后的结果
        new_split = []
        i = 0  # 当前扫描到的位置
        
        # 用 while 循环，因为我们可能会跳过一些位置
        while i < len(split):
            # 如果不是最后一个字符，且当前字符是 a，下一个字符是 b
            if i < len(split) - 1 and split[i] == a and split[i+1] == b:
                # 那就把 a 和 b 合并成 ab，放进新列表
                new_split.append(a + b)
                # 跳过下一个字符（因为已经合并了）
                i += 2
            else:
                # 否则，只把当前字符放进去
                new_split.append(split[i])
                i += 1
        
        # 更新这个词的拆分结果
        splits[word] = new_split
    
    # 返回更新后的 splits
    return splits

# =============================
# 第七步：开始训练！重复“找最常见字符对 → 合并”这个过程
# =============================

# 我们想学习 10 条合并规则（比如 'l'+'l'→'ll', 'll'+'o'→'llo'...）
num_merges = 10

# 用来保存我们学到的所有合并规则
# 比如 {('l','l'): 'll', ('ll','o'): 'llo', ...}
merges = {}

# 开始循环，一共合并 10 次
for step in range(num_merges):
    # 第1步：统计当前所有字符对的频率
    pair_freqs = compute_pair_freqs(splits, word_freqs_with_end)
    
    # 如果没有字符对了，就停止
    if len(pair_freqs) == 0:
        print("没有更多可以合并的字符对了！")
        break
    
    # 第2步：找出频率最高的那一对字符
    # max() 函数：从 pair_freqs 中找“值最大”的那个键
    # key=pair_freqs.get 表示：按 value（频率）来比较
    best_pair = max(pair_freqs, key=pair_freqs.get)
    best_freq = pair_freqs[best_pair]
    
    # 打印我们学到了什么
    print(f"第 {step+1} 次合并：把 '{best_pair[0]}' 和 '{best_pair[1]}' 合并成 '{best_pair[0]+best_pair[1]}'，共出现 {best_freq} 次")
    
    # 记住这条规则
    merges[best_pair] = best_pair[0] + best_pair[1]
    
    # 第3步：真正执行合并
    splits = merge_pair(best_pair[0], best_pair[1], splits)

# =============================
# 第八步：构建最终词汇表（Vocabulary）
# =============================

# 训练完成后，收集所有出现过的 token（子词单元）
vocab = set()

# 遍历所有词的最终拆分结果，把每个 token 加入词汇表
for split in splits.values():
    for token in split:
        vocab.add(token)

# 也加入一些基础字符，确保能覆盖未登录词
# 这里我们可以把原始字符也加进去，比如 'a', 'b', 'l' 等
# 或者更严谨地：从所有单词中提取唯一字符
all_chars = set()
for word in word_freqs_with_end:
    for c in word:
        all_chars.add(c)
vocab.update(all_chars)

# 把词汇表转成排序列表，便于分配 ID
vocab = sorted(vocab)

print("\n" + "="*50)
print("最终词汇表（共 {} 个 token）:".format(len(vocab)))
print("="*50)
print(vocab)

# =============================
# 第九步：构建 token <-> ID 映射表
# =============================

# 创建两个字典：
# token_to_id: 把 token 映射成数字 ID（模型只能处理数字）
# id_to_token: 把 ID 还原成 token（用于解码）

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

print(f"\nToken 到 ID 映射示例（前10个）:")
for token in list(token_to_id.keys())[:10]:
    print(f"  '{token}' → {token_to_id[token]}")

# =============================
# 第十步：定义编码函数（Encode: 文本 → Token IDs）
# =============================

def encode(text):
    """
    将输入文本编码为 token ID 序列。
    使用“贪心最长匹配”策略：优先匹配最长的 token。
    """
    # 添加词尾标记（如果还没有）
    if not text.endswith("</w>"):
        text += "</w>"
    
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        matched = False
        # 从最长可能匹配开始（贪心）
        for j in range(n, i, -1):
            candidate = text[i:j]
            if candidate in token_to_id:
                tokens.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            # 如果没有匹配到，使用单个字符（或 <UNK>）
            tokens.append(text[i])
            i += 1
    
    # 转为 ID
    return [token_to_id[token] for token in tokens]

# =============================
# 第十一步：定义解码函数（Decode: Token IDs → 文本）
# =============================

def decode(token_ids):
    """
    将 token ID 序列解码为原始字符串。
    """
    # 先转成 token
    tokens = [id_to_token[tid] for tid in token_ids]
    # 拼接
    raw = ''.join(tokens)
    # 去掉词尾标记 </w>
    if raw.endswith("</w>"):
        raw = raw[:-4]
    return raw

# =============================
# 第十二步：测试编码与解码（完整闭环）
# =============================

print("\n" + "="*50)
print("编码与解码测试（端到端闭环）")
print("="*50)

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
```

