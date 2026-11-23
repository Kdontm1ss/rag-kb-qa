# RAG 知识库问答 · 第 1 步文档入库

上传 PDF、Word（`.docx`）、Markdown（`.md`）和 TXT 后，FastAPI 保存原文件并返回文档 ID。后台 worker 解析、按清洗后文本的 SHA-256 去重、切分、调用百炼生成向量，并将 chunk 和 `vector(1536)` 写入 PostgreSQL。已有 chunk 可按关键词或余弦相似度检索。

## 启动

需要运行中的 Docker Desktop 或 Docker Engine，以及 Docker Compose v2。在仓库根目录执行以下 PowerShell 命令。已有 `.env` 可以直接使用。

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

在 `.env` 填入 `DASHSCOPE_API_KEY`。Compose 用这一份 Key 配置 `EMBEDDING_API_KEY` 和 `LLM_API_KEY`，共用 `DASHSCOPE_BASE_URL`。当前入库使用 `qwen3.7-text-embedding`，每次请求显式传 `dimensions=1536`。文档片段和向量检索词会发送到所配置的百炼地址。Key 和服务地址须对应同一百炼地域。 [百炼 OpenAI 兼容 Embedding 文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)

```powershell
docker compose up -d --build
```

`docker compose ps` 可以查看 API、worker、PostgreSQL 16 和 Redis 7。服务就绪后，用 `curl.exe -fsS http://127.0.0.1:8000/health` 检查 API 是否能访问文档表。API 默认只发布到本机 `127.0.0.1:8000`，可以用 `API_PORT` 修改宿主机端口。首次构建需要下载基础镜像、Python 依赖和 tokenizer 词表。 [接口文档](http://127.0.0.1:8000/docs)

`docker compose logs -f worker` 可查看处理日志。`docker compose down` 停止服务并保留数据卷。`sql/init.sql` 在 PostgreSQL 数据卷首次初始化时执行。

## 手动验收

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

## 接口

所有业务接口返回 `{code, message, data}`。成功时 `code` 为 `0`。

| 方法与路径 | 用途 |
| --- | --- |
| `POST /api/documents/upload` | multipart 的 `file` 字段，单文件上限 20 MiB，异步处理 |
| `GET /api/documents` | 文档列表，支持 `status`、`limit`、`offset`。`status=needs_review` 为复核列表 |
| `GET /api/documents/{document_id}` | 查询处理状态、错误原因、去重指向和 chunk 数量 |
| `GET /api/chunks/search` | `q` 为检索文本，`mode=keyword` 或 `vector`，`limit` 默认 5 |

`needs_review` 表示原件无法解析、无可提取文本或表格无法完整切分。`failed` 表示向量服务、存储或其他处理出错。修正配置后可重新上传失败的文件。

## 实现与限制

- 原文件保存在 API 和 worker 共享的 `uploads` 卷。上传在 PostgreSQL 登记处理任务，再通过 Redis 通知 worker。worker 会补扫没有收到通知的 `processing` 记录。chunk 写入与 `ready` 状态在同一事务中提交。
- 以 Markdown 标题、段落和句子为优先边界，正文目标约 600 token、重叠约 15%，按 `cl100k_base` 估算。短章节、PDF 分页和表格边界可能低于 300 token。每个 chunk 最多 800 token。表格优先整体保留，超长表格按完整行拆分并重复表头。
- PDF 支持文字层和可识别的带边框表格。扫描件需要先做 OCR，加密件需要先解密，旧版 `.doc` 需要另存为 `.docx`。复杂版式、无边框表格和 Word 嵌套表格的提取效果有限。
- 关键词使用不区分大小写的字面子串匹配，支持中文短语。向量检索在 `ready` 文档的 chunk 上精确计算余弦相似度。更换 Embedding 模型后，已有向量需要用同一模型重新入库。
