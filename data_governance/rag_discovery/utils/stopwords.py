"""
Portuguese stopwords for TF-IDF vectorizer
Used in hybrid ranking for lexical matching
"""

PORTUGUESE_STOPWORDS = [
    # Artigos
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "as", "às",
    "da", "das", "de", "dela", "delas", "dele", "deles", "do", "dos", "duas",
    "esta", "estas", "este", "estes", "está", "estás",
    "o", "os", "um", "uma", "umas", "uns",

    # Preposições/contrações
    "com", "como", "contra", "desde", "em", "entre", "para", "perante", "por",
    "sem", "sob", "sobre", "trás", "pela", "pelas", "pelo", "pelos",
    "num", "numa", "nuns", "numas", "dum", "duma", "duns", "dumas",

    # Pronomes
    "ele", "eles", "ela", "elas", "eu", "lhe", "lhes", "me", "meu", "meus",
    "minha", "minhas", "nós", "se", "seu", "seus", "sua", "suas", "te", "tu",
    "tua", "tuas", "você", "vocês", "vos", "nos", "nosso", "nossa", "nossos", "nossas",

    # Conjunções
    "e", "mas", "nem", "ou", "porém", "que", "quer", "se", "então", "todavia",
    "contudo", "entretanto", "portanto", "porque", "pois", "logo", "assim",

    # Advérbios/interjeições
    "agora", "aí", "ainda", "ali", "amanhã", "antes", "aqui", "assim", "bem",
    "cedo", "depois", "hoje", "logo", "mais", "mal", "melhor", "menos", "muito",
    "não", "onde", "ontem", "pra", "qual", "quando", "quanto", "quê", "sim",
    "talvez", "tão", "tarde", "tem", "têm", "já", "só", "sempre", "nunca",
    "também", "apenas", "somente", "lá", "cá", "demais", "bastante",

    # Verbos auxiliares comuns
    "ser", "estar", "ter", "haver", "ir", "vir", "poder", "dever", "fazer",
    "foi", "eram", "era", "são", "será", "seria", "sendo", "sido",
    "está", "estão", "estava", "estavam", "esteve", "estiveram",
    "tinha", "tinham", "teve", "tiveram", "tenho", "temos",
    "há", "havia", "houve", "haja",

    # Outros
    "etc", "exemplo", "isso", "isto", "outro", "outros", "outra", "outras",
    "qualquer", "seja", "tempo", "vez", "vezes", "via", "coisa", "coisas",
    "todo", "toda", "todos", "todas", "cada", "mesmo", "mesma", "mesmos", "mesmas",
    "próprio", "própria", "próprios", "próprias", "algo", "alguém", "ninguém",
    "nada", "tudo", "parte", "partes", "tipo", "tipos", "forma", "formas",
    "modo", "modos", "caso", "casos", "dia", "dias", "ano", "anos",

    # Números por extenso
    "um", "uma", "dois", "duas", "três", "quatro", "cinco", "seis", "sete",
    "oito", "nove", "dez", "primeiro", "primeira", "segundo", "segunda",
]

# English stopwords (for mixed content)
ENGLISH_STOPWORDS = [
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "done", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "more", "my", "no", "not",
    "now", "of", "on", "or", "our", "out", "own", "said", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
]

# Combined stopwords for multilingual content
MULTILINGUAL_STOPWORDS = list(set(PORTUGUESE_STOPWORDS + ENGLISH_STOPWORDS))
