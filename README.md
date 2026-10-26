# ragaudit

**给检索增强链路做一次端到端体检。**

`ragaudit` —— 沿检索、拼接、生成三段分别体检，定位答案错在哪个环节。

- 零依赖，Python 3.10+ 标准库即可运行
- 单文件，`ragaudit.py` 拷走就能用
- 不联网、不写文件

<!-- ipos:install -->
## 安装

零依赖，Python 3.10+ 即可：

```bash
pip install dartfit-ragaudit
```

装完直接用命令名调用：

```bash
ragaudit --help
```

也可以不用安装，直接把 `ragaudit.py` 拷走用——它只依赖标准库。

## 为什么需要它

RAG 排障工具多只调参数；ragaudit 先定位错误归属到具体环节，再给出该环节的处置建议。

它遵守下面几条自己定的规矩：

1. **先定位错在哪一段**。检索、拼接、生成三段各自有独立的失效模式，归因错了就白调参数。
2. **归因要有证据**。每一段给出它凭什么被判定为有问题。
3. **不做自动调参**。只给那一段的处置建议，改不改是人的决定。

## 用法

```bash
python ragaudit.py --report rag_report.json
python ragaudit.py --report rag_report.json --json
```

加 `--json` 可以拿到机器可读的结果，便于接到流水线里。

## 已知局限

- 诊断基于检索结果的结构特征，不看回答质量。
- 不重跑检索，也不改索引。
- 结论偏规则化，遇到反直觉的现象要自己复核。

## 跑测试

```bash
python test_ragaudit.py
```

测试用独立的断言与临时数据，不污染工作目录。

## 来源与参考

- Python 标准库文档：<https://docs.python.org/3/>
- GitHub 仓库：<https://github.com/dartfit/ragaudit>

---

## 关于作者

dartfit 做检索增强链路归因相关的问题。这个工具先替你回答一个具体问题，再谈别的。

- 个人主页：<https://dartfit.github.io>
- 仓库：<https://github.com/dartfit/ragaudit>

## 许可

MIT，见 [LICENSE](LICENSE)。
