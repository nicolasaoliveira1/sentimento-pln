import re

STOP = {
    "a", "o", "os", "as", "uma", "um",
    "e", "é", "de", "do", "da", "que",
    "para", "com", "foi", "em"
}

def prep(texto, stopwords):
    texto = texto.lower()
    palavras = re.findall(r"[a-záàâãéêíóôõúçü]+", texto)
    if stopwords:
        palavras = [p for p in palavras if p not in STOP]
    return ' '.join(palavras)
