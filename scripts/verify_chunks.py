import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="调用检索接口并显示命中的 chunk")
    parser.add_argument("query", help="关键词或向量检索文本")
    parser.add_argument("--mode", choices=["keyword", "vector", "both"], default="both")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--expect", help="至少一个结果应包含这段原文")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    modes = ["keyword", "vector"] if args.mode == "both" else [args.mode]
    failed = False
    with httpx.Client(base_url=args.base_url, timeout=45) as client:
        for mode in modes:
            try:
                response = client.get("/api/chunks/search", params={
                    "q": args.query, "mode": mode, "limit": args.limit,
                })
                body = response.json()
            except (httpx.HTTPError, ValueError) as error:
                print(f"{mode} 请求失败 ({type(error).__name__})", file=sys.stderr)
                failed = True
                continue
            if not response.is_success or body["code"] != 0:
                print(f"{mode} HTTP {response.status_code} {body['message']}", file=sys.stderr)
                failed = True
                continue

            hits = body["data"]
            print(json.dumps({"mode": mode, "hits": hits}, indent=2, ensure_ascii=False))
            if not hits or (args.expect and not any(args.expect in hit["content"] for hit in hits)):
                print(f"{mode} 未检索到预期的 chunk", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
