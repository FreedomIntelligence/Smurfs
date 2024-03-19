import math
from collections import Counter

#每一段对话都有一个id, id要改成index数，可以直接通过id访问该对话
#每一个role加一个id field，这样就可以通过两个id找到对话
#需要用bm25搜索：query，thought
#直接匹配：action，存一个字典，每个action一个key对应一个列表，存储二元组（对话id， role id）


class BM25:
    def __init__(self, docs, tokenizer=None, k1=1.5, b=0.75):
        """
        BM25算法的构造器
        :param docs: 文档列表，每个文档是一个string
        :param k1: BM25算法中的调节参数k1
        :param b: BM25算法中的调节参数b
        """
        if tokenizer != None:
            self.docs = [tokenizer.tokenize(doc[-1]) for doc in docs]
            self.tokenizer = tokenizer
        else:
            self.docs = docs
            self.tokenizer = None
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]  # 计算每个文档的长度
        self.avgdl = sum(self.doc_len) / len(docs)  # 计算所有文档的平均长度
        self.doc_freqs = []  # 存储每个文档的词频
        self.idf = {}  # 存储每个词的逆文档频率
        self.initialize()

    def initialize(self):
        """
        初始化方法，计算所有词的逆文档频率
        """
        df = {}  # 用于存储每个词在多少不同文档中出现
        for doc in self.docs:
            # 为每个文档创建一个词频统计
            self.doc_freqs.append(Counter(doc))
            # 更新df值
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        # 计算每个词的IDF值
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.docs) - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, doc, query):
        """
        计算文档与查询的BM25得分
        :param doc: 文档的索引
        :param query: 查询词列表
        :return: 该文档与查询的相关性得分
        """
        score = 0.0
        for word in query:
            if word in self.doc_freqs[doc]:
                freq = self.doc_freqs[doc][word]  # 词在文档中的频率
                # 应用BM25计算公式
                score += (self.idf[word] * freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl))
        return score
    
    def retrieve(self, query, top_k = 5):
        """
        retrieve跟query前top_k接近的indices
        query是string
        """
        if self.tokenizer != None:
            query = self.tokenizer.tokenize(query)
        scores = [self.score(i, query) for i in range(len(self.docs))]
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:top_k]

# 示例文档集和查询
docs = [["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog"],
        ["the", "quick", "dog"],
        ["the", "quick", "brown", "brown", "fox"]]
query = ["quick", "brown"]

# 初始化BM25模型并计算得分
bm25 = BM25(docs)
index = bm25.retrieve(query, 2)
# scores = [bm25.score(i, query) for i in range(len(docs))]
print(docs[index[0]])
## query和文档的相关性得分：
## sores = [1.0192447810666774, 0.0, 0.3919504878447609, 1.2045355839511414]