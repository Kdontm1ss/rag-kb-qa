# RAG 知识库问答

上传 PDF、Word（`.docx`）、Markdown（`.md`）和 TXT 后，FastAPI 保存原文件并返回文档 ID。后台 worker 解析、按清洗后文本的 SHA-256 去重、切分、调用百炼生成向量，并将 chunk 和 `vector(1536)` 写入 PostgreSQL。已有 chunk 可按关键词或余弦相似度检索。

问答将 BM25 和向量检索结果融合，交给百炼重排，再由 `qwen-plus` 选择可回答问题的原文。通过原文校验的答案段落与引用会逐段通过 SSE 返回。

## 启动

需要运行中的 Docker Desktop 或 Docker Engine，以及 Docker Compose v2。在仓库根目录执行以下 PowerShell 命令。已有 `.env` 可以直接使用。

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在 `.env` 填入 `DASHSCOPE_API_KEY`，并确认 `LLM_MODEL=qwen-plus`。向量化、重排和问答共用这一个 Key。文档片段与问题会发送给配置的百炼服务，Key 和服务地址须对应同一地域。向量化使用 `qwen3.7-text-embedding`，每次请求显式传 `dimensions=1536`。 [百炼 OpenAI 兼容 Embedding 文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)

重排默认使用 `qwen3-rerank`，按 `DASHSCOPE_BASE_URL` 所在域调用百炼 DashScope 重排接口。可用 `RERANK_BASE_URL` 指定以 `/api/v1` 结尾的百炼地址。`RERANK_MIN_SCORE` 默认 `0.25`，可按实际命中和拒答结果调整。 [百炼文本重排 API](https://help.aliyun.com/zh/model-studio/text-rerank-api)

```powershell
docker compose up -d --build
```

`docker compose ps` 可以查看 API、worker、PostgreSQL 16 和 Redis 7。服务就绪后，用 `curl.exe -fsS http://127.0.0.1:8000/health` 检查 API 是否能访问文档表。API 默认只发布到本机 `127.0.0.1:8000`，可以用 `API_PORT` 修改宿主机端口。首次构建需要下载基础镜像、Python 依赖和 tokenizer 词表。 [接口文档](http://127.0.0.1:8000/docs)

`docker compose logs -f worker` 可查看处理日志。`docker compose down` 停止服务并保留数据卷。`sql/init.sql` 在 PostgreSQL 数据卷首次初始化时执行。

已有第 1 步数据卷时，直接执行同一条启动命令即可。API 启动会应用 `sql/migrations/002_chunk_lexical_search.sql`，为原有 chunk 计算关键词并建立索引，原文件和向量无需重做。首次回填期间，API 会等索引建好再接受请求。

## 入库与检索验收

1. 将 `$file` 改为本机真实文档的绝对路径并上传。接口返回 HTTP `202` 和 `data.document_id`。

   ```powershell
   $file = 'D:\资料\请假制度.pdf'
   curl.exe -fsS -F "file=@$file" http://127.0.0.1:8000/api/documents/upload
   ```

2. 将 `$id` 换成返回的 ID，检查状态从 `processing` 进入 `ready`，并确认 `chunk_count` 大于零。再次上传同一份内容，第二条记录应在 `duplicate_of` 指向首条 `ready` 记录，检索只返回原记录的 chunk。

   ```powershell
   $id = '<document_id>'
   curl.exe -fsS "http://127.0.0.1:8000/api/documents/$id"
   ```

3. 将 `$phrase` 改为原文中的短语。脚本默认运行关键词和向量检索，输出命中的全文片段、文档名、chunk 序号、页码或标题。`--expect` 会检查两种结果均包含预期原文，任一失败返回非零退出码。用 `--mode vector` 可单独验证同义表述。

   ```powershell
   $phrase = '年假申请'
   docker compose exec -T api python scripts/verify_chunks.py "$phrase" --expect "$phrase"
   docker compose exec -T api python scripts/verify_chunks.py '如何提交带薪年休假' --mode vector --expect "$phrase"
   ```

4. 上传一个损坏的 `.pdf`，确认它在复核列表中带有 `error_message`。可以用以下命令单独制作一个解析失败样本。

   ```powershell
   $broken = Join-Path ([IO.Path]::GetTempPath()) ("rag-broken-" + [guid]::NewGuid() + ".pdf")
   Set-Content -LiteralPath $broken -Value 'invalid PDF'
   curl.exe -fsS -F "file=@$broken" http://127.0.0.1:8000/api/documents/upload
   curl.exe -fsS 'http://127.0.0.1:8000/api/documents?status=needs_review'
   ```

5. 用 20 份真实文档重复上传和检索，记录每条验证查询命中的文档和 chunk 位置。真实文档检索效果与性能待测。

## 问答验收

先上传一份包含明确规则的真实文档，等待状态变为 `ready`。把下面的问题改成这份文档明确回答的事实，然后观察 SSE 输出。`-N` 关闭 curl 的输出缓冲。

```powershell
$body = @{ question = '年假申请需要提前几个工作日提交？' } | ConvertTo-Json -Compress
$requestFile = Join-Path ([IO.Path]::GetTempPath()) ("rag-ask-" + [guid]::NewGuid() + ".json")
[IO.File]::WriteAllText($requestFile, $body)
curl.exe -NsS -H 'Content-Type: application/json' --data-binary "@$requestFile" http://127.0.0.1:8000/api/chat/ask
```

- 有证据时，收到带 `[n]` 引用的 `answer_delta`。按 `citations` 中的 `number` 对应编号，核对 `document_name`、`chunk_index` 与原文。`chunk_index` 从 0 开始，PDF 的 `page` 从 1 开始。`document_id`、`chunk_id` 和 `section` 可用于进一步定位。
- 将同一命令中的问题换成一个原文没有交代的细节，再试一个与文档无关的问题。两个测试都应收到明确的拒答和空引用。
- 从评测问题中选 10 个逐个测试，按需求文档检查回答事实与引用是否一致、无证据的题是否拒答、流是否有 `done: true`。真实问题集的验收结果与耗时待测。

每个 `data:` 帧包含 `answer_delta`、`citations` 和 `done`。结束帧带 `status`，`answered` 表示回答完成，`refused` 表示已拒答，`error` 表示处理或流式输出失败。`reason` 提供机器可读的原因。客户端应拼接 `answer_delta`，并按 `number` 保存引用。以 `:` 开头的帧是 SSE 注释，可忽略。

本地 Python 3.12 安装依赖后，可以运行 `python -m unittest discover -s tests -v` 检查引用、拒答和流式交付。

## 接口

普通业务接口返回 `{code, message, data}`，成功时 `code` 为 `0`。问答流以 `text/event-stream` 返回。

| 方法与路径 | 用途 |
| --- | --- |
| `POST /api/documents/upload` | multipart 的 `file` 字段，单文件上限 20 MiB，异步处理 |
| `GET /api/documents` | 文档列表，支持 `status`、`limit`、`offset`。`status=needs_review` 为复核列表 |
| `GET /api/documents/{document_id}` | 查询处理状态、错误原因、去重指向和 chunk 数量 |
| `GET /api/chunks/search` | `q` 为检索文本，`mode=keyword` 或 `vector`，`limit` 默认 5 |
| `POST /api/chat/ask` | JSON 请求体 `{"question":"…"}`，SSE 返回带引用的回答、拒答或错误 |

`needs_review` 表示原件无法解析、无可提取文本或表格无法完整切分。`failed` 表示向量服务、存储或其他处理出错。修正配置后可重新上传失败的文件。

## 实现与限制

- 原文件保存在 API 和 worker 共享的 `uploads` 卷。上传在 PostgreSQL 登记处理任务，再通过 Redis 通知 worker。worker 会补扫没有收到通知的 `processing` 记录。chunk 写入与 `ready` 状态在同一事务中提交。
- 以 Markdown 标题、段落和句子为优先边界，正文目标约 600 token、重叠约 15%，按 `cl100k_base` 估算。短章节、PDF 分页和表格边界可能低于 300 token。每个 chunk 最多 800 token。表格优先整体保留，超长表格按完整行拆分并重复表头。
- PDF 支持文字层和可识别的带边框表格。扫描件需要先做 OCR，加密件需要先解密，旧版 `.doc` 需要另存为 `.docx`。复杂版式、无边框表格和 Word 嵌套表格的提取效果有限。
- 关键词使用 PostgreSQL 全文匹配和 GIN 索引，再用 Okapi BM25 打分。中文使用重叠二字词，保留单字查询能力。向量检索在 `ready` 文档的 chunk 上精确计算余弦相似度。两路各取最多 8 条，按 RRF 去重融合后取前 8 条。 [PostgreSQL 全文索引](https://www.postgresql.org/docs/16/textsearch-indexes.html)
- 重排后只提交达到 `RERANK_MIN_SCORE` 的前 4 段。符合条件的段落不足 4 个时，使用实际数量。无合格段落、模型判定资料不足或首段无法在引用原文核验时，接口会拒答。
- 回答采用原文摘录，每段引用由服务端从候选 chunk 补全。这样可以直接核验原文，但语义相关性和问题覆盖仍依赖检索、重排与模型判断。阈值在真实语料上待校准，大语料下 BM25 统计和精确向量扫描的性能待测。
