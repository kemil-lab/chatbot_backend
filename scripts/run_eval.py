from app.services.eval_service import run_rag_eval


def main():
    result = run_rag_eval("data/eval/eval_data.json")
    print(result)


if __name__ == "__main__":
    main()