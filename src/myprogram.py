import os
import string
import random
import pickle
import urllib.request
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter


class MyModel:

    def __init__(self):
        self.unigrams = Counter()    
        self.bigrams = defaultdict(Counter) #creating the empty dictonaries
        self.trigrams = defaultdict(Counter)
        self.fourgrams = defaultdict(Counter)
        self.fivegrams = defaultdict(Counter)
        self.precomputed = {} #the top 3 predictions for every key 
        self.lambdas = [0.05, 0.10, 0.15, 0.30, 0.40] #interpolation weight values 

    @classmethod
    def load_training_data(cls):
        data = []

        urls = [ #downlaoding the books so that you can use to train 
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride & Prejudice
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
            "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities 
            "https://www.gutenberg.org/files/135/135-0.txt",    # Les Miserables (french)
            "https://www.gutenberg.org/files/2000/2000-0.txt",  # Don Quijote (spanish)
            "https://www.gutenberg.org/files/2229/2229-0.txt",  # Faust (german)
            "https://www.gutenberg.org/files/1012/1012-0.txt",  # Divina Commedia (italian)
            "https://www.gutenberg.org/files/2413/2413-0.txt",  # Don Quijote Part 2 (spanish)
            "https://www.gutenberg.org/files/4650/4650-0.txt",  # Confessions (portuguese)
            "https://www.gutenberg.org/files/2600/2600-0.txt",  # War and Peace (russian)
            "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations (english)
        ]

        for url in urls:
            try:
                response = urllib.request.urlopen(url)
                text = response.read().decode('utf-8')
                data.append(text)
                print(f"Downloaded {len(text)} from {url}")
            except Exception as e:
                print(f"Skipping {url}: {e}")  #skipping it if it fails
        if len(data) == 0:
            data = ["Test data"] #fallback strategy if everything fails 
        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f: #opening the test file 
            for line in f:
                line = line.strip()
                data.append(line)  #taking off the last line charecter which is invisible 
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds: 
                f.write(p + '\n') #writing in the predictions along with the newline character

    def run_train(self, data, work_dir):
        for text in data:
            text = text.lower() #normalizing to lowercase before counting  
            for i in range(len(text)):
                self.unigrams[text[i]] += 1 #single character
            for i in range(len(text) - 1):
                self.bigrams[text[i]][text[i + 1]] += 1 #1 char character (following char)
            for i in range(len(text) - 2):
                self.trigrams[text[i : i + 2]][text[i + 2]] += 1 #2 char character 
            for i in range(len(text) - 3):
                self.fourgrams[text[i : i + 3]][text[i + 3]] += 1 #3 char character 
            for i in range(len(text) - 4):
                self.fivegrams[text[i : i + 4]][text[i + 4]] += 1 #4 char character 
            
        print(f"Unigrams: {len(self.unigrams)}, Bigrams: {len(self.bigrams)}, Trigrams: {len(self.trigrams)}, Fourgrams: {len(self.fourgrams)}, Fivegrams: {len(self.fivegrams)}")
        print("Precomputing the predictions!") #the top 3 predictons for every key at every level 
        self._precompute_predictions()
        print(f"Precomputed {len(self.precomputed)} keys")

    def _precompute_predictions(self):
        all_keys = set() #collecting all the key we have seen across the n gram levels 
        all_keys.update(self.bigrams.keys()) #1 char keys
        all_keys.update(self.trigrams.keys()) #2 char keys
        all_keys.update(self.fourgrams.keys()) #3 char keys
        all_keys.update(self.fivegrams.keys()) #4 char keys (most confidence in this one )

        all_chars = list(self.unigrams.keys()) #collecting all the characters we have seen 
        total_unigrams = sum(self.unigrams.values())

        for key in all_keys:
            scores = Counter()
            for char in all_chars:
                score = 0.0
                if total_unigrams > 0:
                    score += self.lambdas[0] * (self.unigrams[char] / total_unigrams)  # how common is the char
                else:
                    score += 0.0

                bigram_key = key[-1:] #taking the last character and finding what we have seen before
                if bigram_key in self.bigrams:
                    total = sum(self.bigrams[bigram_key].values())
                    if total > 0:
                        score += self.lambdas[1] * (self.bigrams[bigram_key][char] / total)
                    else:
                        score += 0.0
                else:
                    score += 0.0

                trigram_key = key[-2:] #what is follow the last two chars
                if trigram_key in self.trigrams:
                    total = sum(self.trigrams[trigram_key].values())
                    if total > 0:
                        score += self.lambdas[2] * (self.trigrams[trigram_key][char] / total)
                    else:
                        score += 0.0
                else:
                    score += 0.0

                fourgram_key = key[-3:] #following the last 3 characters
                if fourgram_key in self.fourgrams:
                    total = sum(self.fourgrams[fourgram_key].values())
                    if total > 0:
                        score += self.lambdas[3] * (self.fourgrams[fourgram_key][char] / total)
                    else:
                        score += 0.0
                else:
                    score += 0.0

                fivegram_key = key[-4:] #what is following the last 4 characters 
                if fivegram_key in self.fivegrams:
                    total = sum(self.fivegrams[fivegram_key].values())
                    if total > 0:
                        score += self.lambdas[4] * (self.fivegrams[fivegram_key][char] / total)
                    else:
                        score += 0.0
                else:
                    score += 0.0

                scores[char] = score

            top_3 = scores.most_common(3)
            result = []
            for char, score in top_3: 
                result.append(char) #adding the character
                    
            self.precomputed[key] = result

    def run_pred(self, data):
        preds = []
        top_3_unigrams = self.unigrams.most_common(3) #getting the 3 most common characters from trainig data
        fallback = []

        for char, _ in top_3_unigrams: #ignoring the count
            fallback.append(char)
        
        for inp in data: 
            try: 
                top_guesses = None

                if len(inp) >= 4 and inp[-4:].lower() in self.precomputed: #using the fivegram key approach
                    top_guesses = self.precomputed[inp[-4:].lower()]
                elif len(inp) >= 3 and inp[-3:].lower() in self.precomputed:
                    top_guesses = self.precomputed[inp[-3:].lower()]  #fourgram keys
                elif len(inp) >= 2 and inp[-2:].lower() in self.precomputed:
                    top_guesses = self.precomputed[inp[-2:].lower()]  #trigram keys
                elif len(inp) >= 1 and inp[-1:].lower() in self.precomputed:
                    top_guesses = self.precomputed[inp[-1:].lower()]  #bigram keys
                
                if top_guesses is None: 
                    top_guesses = fallback #using the character that was most popular 

                top_guesses = list(dict.fromkeys(top_guesses))
                while len(top_guesses) < 3: 
                    top_guesses.append(' ')
                preds.append(''.join(top_guesses[:3]))
            
            except Exception as e: 
                print(f"Prediction failed for input: {e}")
                preds.append(''.join(fallback[:3]))  #  fallback

        return preds

    def save(self, work_dir):
        checkpoint = {
            'unigrams': dict(self.unigrams), 
            'bigrams': dict(self.bigrams), #1 character context and so on 
            'trigrams': dict(self.trigrams),  
            'fourgrams': dict(self.fourgrams),        
            'fivegrams': dict(self.fivegrams),        
            'precomputed': self.precomputed,
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(checkpoint, f)
        print("Checkpoint saved")

    @classmethod
    def load(cls, work_dir):
        model = cls() #creating an empty model 
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model.unigrams = Counter(checkpoint['unigrams'])
        model.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['bigrams'].items()})
        model.trigrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['trigrams'].items()})
        model.fourgrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['fourgrams'].items()})
        model.fivegrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['fivegrams'].items()})
        model.precomputed = checkpoint['precomputed']
        return model


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
        print('Instantiating model')
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

