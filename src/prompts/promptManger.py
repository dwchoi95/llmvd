class PromptManager:
    """Thread-safe template rendering.

    Templates are formatted from a LOCAL variable and cached per path: the
    previous implementation stored the template on `self` between read and
    format, so concurrent renders (prompt building runs in worker threads)
    could format one scope's kwargs against another scope's template
    (KeyError: 'file_code').
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    def render(self, **kwargs) -> str:
        file: str = kwargs.pop("file")
        template = self._cache.get(file)
        if template is None:
            with open(file, 'r', encoding='utf-8') as f:
                template = f.read()
            self._cache[file] = template
        return template.format(**kwargs)
