# Formula patterns
DISPLAY_FORMULA = r"\$\$(.*?)\$\$"
INLINE_FORMULA = r"(?<!\$)\$(?!\$)(.*?)\$(?!\$)"
LATEX_ENV = r"\\begin\{(equation|align|gather|multiline)\*?\}(.*?)\\end\{\1\*?\}"