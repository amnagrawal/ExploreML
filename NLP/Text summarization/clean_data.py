"""
This file contains fxns to clean the data following URL:
https://machinelearningmastery.com/prepare-news-articles-text-summarization/
"""
import string

def clean_lines(lines):
    cleaned = list()
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]
            
        line = line.split()
        line = [word.lower() for word in line]
        line = [w.translate(table) for w in line]
        line = [word for word in line if word.isalpha()]
        cleaned.append(' '.join(line))
        
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned