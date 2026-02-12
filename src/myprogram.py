#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self): 
        self.bigrams = defaultdict(Counter) #for filling in the bigrams
        self.trigrams = defaultdict(Counter) #for filling in the trigrams


    @classmethod
    def load_training_data(cls):
        data = [] 
        import urllib.request 

        urls = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride & Prejudice
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
        ]

        for url in urls: 
            try: 
                response = urllib.request.urlopen(url)
                text = response.read().decode('utf-8')
                data.append(text)
                print(f"Downloaded {len(text)} charecters")
            except Exception as e: 
                print(f"Failed to download {url}: {e}")
        if len(data) == 0: 
            data = ["TEST DATA! Hello World!"]
        return data 
    
        # your code here
        # this particular model doesn't train

        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        for text in data: #going through text example (data pending)
            for i in range(len(text) - 1): #looping for bigrams
                current = text[i] #1st char
                next_char = text[i+1] #2nd char
                self.bigrams[current][next_char] += 1 #storing this sequence
            for i in range(len(text)-2):
                current = text[i:i+2] #last 2 chars
                next_char = text[i+2] #next char
                self.trigrams[current][next_char] += 1 #storing in trigrams

        print(self.bigrams)
        print(self.trigrams)
       #pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            if len(inp) >= 2: #if we have at least 2 characters 
                last_two_chars = inp[-2:] 
                if last_two_chars in self.trigrams:
                    counts = self.trigrams[last_two_chars] #counter for this 
                    top_3_char = counts.most_common(3)
                    top_guesses = []
                    for char, count in top_3_char:
                        top_guesses.append(char)
                else:
                    top_guesses = None #don't have any trigrams 
            else:
                top_guesses = None #too short 

            if top_guesses is None and len(inp) >= 1: #need at least 1 char (bigram method)
                last_char = inp[-1] 
                if last_char in self.bigrams:
                    counts = self.bigrams[last_char] #counter 
                    top_3_char = counts.most_common(3)
                    top_guesses = []
                    for char, count in top_3_char:
                        top_guesses.append(char)
                else:
                    top_guesses = None #don't have any trigrams 

            if top_guesses is None: #none of the previous ones worked
                if len(inp) == 0: 
                    top_guesses = ['T', 'I', 'A'] #starting chars most common
                elif inp[-1] == ' ':
                    top_guesses = ['t', 'i', 'a'] #words after first one (not capital)
                elif inp[-1] in '.,;:': #punctuation
                    top_guesses = [' ', 'T', 'I']
                else:
                    top_guesses = [' ', 'e', 't'] #spacing or e/t

            #top_guesses = [random.choice(all_chars) for _ in range(3)]
            while len(top_guesses) < 3: #at least 3 chars
                top_guesses.append(' ')
            preds.append(''.join(top_guesses[:3]))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
