from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import numpy as np
import itertools
import spacy
import nltk
import re
import os
os.environ['CUDA_VISIBLE_DIVICES'] = "0"
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_lg')


def replace_patterns(text):
    text = text.replace(" . ", '. ')
    text = text.replace(" , ", ", ")
    text = text.replace(".. ",". " )
    text = text.replace('. . ', '. ')
    return text


def softClean(text):
    text = re.sub(r'\s+', " ", text)
    text = text.replace('"', '')
    text = replace_patterns(text)
    text = replace_patterns(text)
    text = re.sub(r"\s+", " ", text)
    return text


def sentenceTokenizer(text):
    text = softClean(text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def named_entity_ratio(text):
    doc = nlp(text)
    total_words = len(doc)
    named_entities = sum(1 for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"])
    return named_entities / total_words if total_words > 0 else 0


def proper_noun_ratio(text):
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    total_words = len(tagged_words)
    proper_nouns = sum(1 for word, pos in tagged_words if pos in ['NNP', 'NNPS'])
    return proper_nouns / total_words if total_words > 0 else 0


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.com')
    words = text.split()
    filtered_words = [word for word in words if not url_pattern.search(word)]
    new_text = " ".join(filtered_words)
    return new_text


def positional_score(sentence, text):
    lines = sentenceTokenizer(text)
    title = " ".join(lines[0:3])
    first_para = " ".join(lines[3:6])
    if sentence in title:
        return 1.22
    elif sentence in first_para:
        return 1.12
    return 1


def sentence_relevance(sentence, paragraph_word_counts, total_words):
    sentence_words = sentence.split()
    word_frequencies = [paragraph_word_counts[word] for word in sentence_words if word not in stop_words]
    relevance_score = sum(word_frequencies) / total_words
    return relevance_score


def convert_to_percentiles(phrases):
    numbers = [item[1] for item in phrases]
    min_val = np.min(numbers)
    max_val = np.max(numbers)
    if max_val == min_val:
        numbers = [0 for _ in numbers]
    else:
        numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
    for ind in range(len(phrases)):
        phrases[ind] = [phrases[ind][0], str(numbers[ind])]
    return phrases


def remove_stop_words(phrase_list):
    return [phrase for phrase in phrase_list if phrase not in stop_words]


def get_tokens(text):
    tokens = word_tokenize(text)
    return list(set([token for token in tokens if token.isalnum()]))

def rankPhrases(text, original_phrases):
    if text is None or text == "":
        return [[item, str(0)] for item in original_phrases]
    text = text.lower()
    mp = {remove_urls(phrase.lower()): phrase for phrase in original_phrases}
    phrases = mp.keys()
    if phrases == []:
        return []
    sentences = sentenceTokenizer(text.lower())
    phrases_words = {phrase: set(remove_stop_words(get_tokens(phrase))) for phrase in phrases}
    sentences_words = [set(get_tokens(sentence)) for sentence in sentences]

    graph = nx.DiGraph()
    phrase_counts = Counter()

    for phrase in phrases:
        graph.add_node(phrase, count=0)

    for sentence_words in sentences_words:
        co_occurring_phrases = set()
        for phrase, phrase_words in phrases_words.items():
            if phrase_words.issubset(sentence_words):
                co_occurring_phrases.add(phrase)
                phrase_counts[phrase] += 1
        for phrase1, phrase2 in itertools.combinations(co_occurring_phrases, 2):
            if graph.has_edge(phrase1, phrase2):
                graph[phrase1][phrase2]["weight"] += 1
            else:
                graph.add_edge(phrase1, phrase2, weight=1)

    for phrase, count in phrase_counts.items():
        graph.nodes[phrase]["count"] = count
    paragraph_words = text.split()
    total_words = len(paragraph_words)
    paragraph_word_counts = Counter(paragraph_words)
    pagerank_scores = nx.pagerank(graph, weight="weight")
    final_scores = {}
    for phrase in graph.nodes:
        pnr = proper_noun_ratio(phrase)
        ner = named_entity_ratio(phrase)
        tfidf = sentence_relevance(phrase, paragraph_word_counts, total_words)
        pos_score = positional_score(phrase, text)
        count = graph.nodes[phrase]["count"]
        page_score = pagerank_scores.get(phrase, 0)
        new_count = count + page_score + ((count + page_score)*(pnr + ner + tfidf))
        final_scores[phrase] = pos_score * new_count
    sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    result = [[mp[a], b] for a, b in sorted_scores]
    return convert_to_percentiles(result)

