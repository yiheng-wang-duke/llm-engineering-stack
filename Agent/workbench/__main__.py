import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Qwen Agent Workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "workbench.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
