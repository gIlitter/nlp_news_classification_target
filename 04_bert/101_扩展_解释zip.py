"""
解释 texts,labels = zip(*batch)的用法

"""

batch = [("文本1", 0), ("文本2", 1), ("文本3", 0)]

print(*batch)

texts, labels = zip(*batch)
print(texts)
print(labels)