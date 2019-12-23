
"""
This file contains 
"""

from pickle import dump

def dump_stories(stories, name, path='./'):
    dump(stories, open(path+name, 'wb'))