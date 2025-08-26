import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

tokens = enc.encode("鸡那么美你怎么不去娶鸡？🐔")
print(tokens)
text = enc.decode(tokens)

print(text)
