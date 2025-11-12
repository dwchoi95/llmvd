import tiktoken


class Evaluation:
    def token_count(self, text:str) -> int:
        if text is None:
            return 0
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(str(text)))