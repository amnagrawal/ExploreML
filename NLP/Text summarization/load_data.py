"""
    This file contains fxns to load data following the URL: 
    https://machinelearningmastery.com/prepare-news-articles-text-summarization/
"""


from os import listdir
from tqdm import tqdm

def load_doc(filename):
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text


def load_stories(directory):
    stories = list()
    for name in tqdm(listdir(directory)):
        filename =  directory + '/' + name
        doc = load_doc(filename)
        story, highlights = split_story(doc)
        stories.append({'story':story, 'highlights':highlights})
    return stories

def split_story(doc):
    index=doc.find('@hihghlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h)>0]
    return story, highlights