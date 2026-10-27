---
name: arxiv_search
description: 用 arxiv Atom API 检索论文元数据（标题/摘要/作者/分类/时间）
priority: 10
---

## 何时用

需要检索学习/研究相关的 arXiv 学术论文元数据时。用只读工具 `web_fetch` 直接打 arXiv 的
**Atom API**（本环境 `web_fetch` 可用的只读网络 GET；不需要 shell/curl）。

## API 端点

```
https://export.arxiv.org/api/query
```

## 查询参数

| 参数 | 说明 | 例 |
|---|---|---|
| `search_query` | 检索表达式：字段前缀 + 布尔运算符 | `all:reinforcement+AND+all:red-teaming` |
| `id_list` | 按 arxiv ID 精确查（逗号分隔） | `2606.09701,2605.19485` |
| `start` / `max_results` | 分页偏移 / 每页条数（最多 100） | `0` / `10` |
| `sortBy` | `submittedDate` / `lastUpdatedDate` / `relevance` | `submittedDate` |
| `sortOrder` | `ascending` / `descending` | `descending` |

字段前缀：`all:` 全字段 · `ti:` 标题 · `au:` 作者 · `abs:` 摘要 · `cat:` 分类（如 `cat:cs.CL`）。
布尔：`AND` / `OR` / `ANDNOT`，可用括号分组。**URL 编码**：空格用 `+`，括号 `%28` `%29`；
`language model` 会被当 `language OR model`，精确短语要用 `%22language+model%22`。

## 使用（web_fetch）

```text
web_fetch("https://export.arxiv.org/api/query?search_query=all:reinforcement+AND+all:red-teaming+AND+all:language&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending", max_chars=10000)
```

返回 Atom XML，每个 `<entry>` 含：`<title>` 标题、`<summary>` **完整摘要**、`<published>` 首次提交时间、
多个 `<author><name>`、多个 `<category term>` 分类、`<id>`（含 abs 链接）。

## 步骤

1. 用 `web_fetch` 带关键词检索，`max_results` 取小一点（5~10），避免 XML 太长被截断；
2. 从 `<summary>` 摘要里判断哪些论文相关，挑出候选；
3. 需要精确时，用 `id_list` 单独再查某几篇 `<id>`；
4. 综述/罗列时只保留 `title + 年份 + 一句要点`，别把整段摘要原样贴给用户。

## 限制

- **只读检索元数据**：本课的 `web_fetch` 只做 GET、有大小上限与超时；不取 PDF 全文、不写文件。
- 连续多次请求之间注意节奏，避免 arxiv 429（本环境单次 web_fetch 自带超时，但别并发猛打）。