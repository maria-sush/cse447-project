#!/usr/bin/env python
import os
import string
import random
import pickle
import urllib.request
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter


class MyModel:

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self):
        self.bigrams = defaultdict(Counter) #creating the empty dictonaries
        self.trigrams = defaultdict(Counter)
        self.gpt2_model = None    
        self.gpt2_tokenizer = None  #this is for laoding later 

    @classmethod
    def load_training_data(cls):
        data = []

        urls = [ #downlaoding the books so that you can use to train 
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride & Prejudice
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
        ]

        for url in urls:
            try:
                response = urllib.request.urlopen(url)
                text = response.read().decode('utf-8')
                data.append(text)
                print(f"Downloaded {len(text)} from {url}")
            except Exception as e:
                print(f"Could not download {url}: {e}") #just skipping if it doesn't work

        if len(data) == 0:
            data = ["Test data"] #in case it doesn't work 
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
                f.write(p + '\n') #adding the predicions back in and the new line charecter

    def run_train(self, data, work_dir):
        for text in data:
            for i in range(len(text) - 1):
                self.bigrams[text[i]][text[i + 1]] += 1 #count what character follows the last 1 character
            for i in range(len(text) - 2):
                self.trigrams[text[i:i + 2]][text[i + 2]] += 1 #count what character follows the last 2

        print(f"Bigram keys: {len(self.bigrams)}, Trigram keys: {len(self.trigrams)}")

        print("Downloading GPT-2")
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            gpt2_dir = os.path.join(work_dir, 'gpt2') 
            os.makedirs(gpt2_dir, exist_ok=True) #making a a folder to save gpt-2 into 
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.save_pretrained(gpt2_dir)
            model.save_pretrained(gpt2_dir)
            print(f"GPT-2 saved to {gpt2_dir}")#save to disk so we only download once
        except Exception as e:
            print(f"Could not download GPT-2: {e}")
            print("Fall back to n-grams")

    def _load_gpt2(self, work_dir):
        if self.gpt2_model is not None:
            return True #already in memory 
        gpt2_dir = os.path.join(work_dir, 'gpt2')
        if not os.path.isdir(gpt2_dir):
            return False #gpt2 was never downloaded
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_dir) #converting the text to numbers 
            self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_dir)
            self.gpt2_model.eval() #setting it to the eval mode 
            print("GPT-2 loaded.") 
            return True
        except Exception as e:
            print(f"Could not load GPT-2: {e}")
            return False

    def _gpt2_predict(self, text, top_k=3):
        import torch
        inputs = self.gpt2_tokenizer(
            text[-500:], #only using the last 500 characters 
            return_tensors='pt',
            truncation=True, #cutting it off if it still too long 
            max_length=512 
        )
        with torch.no_grad():
            outputs = self.gpt2_model(**inputs) #turning off the gradient tracking (only predicting)
        logits = outputs.logits[0, -1, :] #getting a logit for possible next tokens
        top_indices = torch.topk(logits, 200).indices #getting the top 200 characters for 3 unique characters
        seen_chars = []
        for tok_id in top_indices:
            decoded = self.gpt2_tokenizer.decode(tok_id)
            if not decoded:
                continue #skipping the empty tokens 
            first_char = decoded[0].lower()
            if first_char not in seen_chars:
                seen_chars.append(first_char)
            if len(seen_chars) == top_k: #stop once have found the 3 unique first characters 
                break
        if len(seen_chars) == top_k:
            return seen_chars
        else:
            return None

    def _ngram_predict_with_confidence(self, inp, top_k=3):
        counter = None

        if len(inp) >= 2: #trying the rigrams approach first 
            last_two = inp[-2:]
            if last_two in self.trigrams:
                counter = self.trigrams[last_two] #what characters are usualy following this 

        if counter is None and len(inp) >= 1: #using bigram 
            last_one = inp[-1]
            if last_one in self.bigrams:
                counter = self.bigrams[last_one]

        if counter is None:
            return None, 0.0

        total = sum(counter.values()) #the amount of times this pattern was seen 
        top_items = counter.most_common(top_k)
        top_guesses = [c for c, _ in top_items] #getting the characters
        top_count = top_items[0][1] #how many times we saw the top char
        confidence = top_count / total if total > 0 else 0.0 #fraction of the time it was top 

        return top_guesses, confidence

    def _heuristic_predict(self, inp): #simple prediction approach
        if len(inp) == 0:
            return ['T', 'I', 'A']
        elif inp[-1] == ' ':
            return ['t', 'i', 'a']
        elif inp[-1] in '.,;:!?':
            return [' ', 'T', 'I']
        else:
            return [' ', 'e', 't']

    def run_pred(self, data):
        preds = []
        gpt2_available = self.gpt2_model is not None #if the gpt2 model has been loaded
        gpt2_calls = 0 

        for inp in data:
            top_guesses = None
            ngram_guesses, confidence = self._ngram_predict_with_confidence(inp, top_k=3) #getting the n-gram prediction and confidence score 
            if ngram_guesses is not None and confidence >= self.CONFIDENCE_THRESHOLD: #if we are meetin the threhold
                top_guesses = ngram_guesses
            elif gpt2_available:
                try:
                    top_guesses = self._gpt2_predict(inp, top_k=3) #using gpt2 for the prediction
                    gpt2_calls += 1
                except Exception as e:
                    print(f"GPT-2 prediction failed: {e}")
                    top_guesses = ngram_guesses #crashed then go back to ngram

            elif ngram_guesses is not None:
                top_guesses = ngram_guesses
            if top_guesses is None:
                top_guesses = self._heuristic_predict(inp) #the simple approach (basic rules)

            while len(top_guesses) < 3:
                top_guesses.append(' ')
            preds.append(''.join(top_guesses[:3]))

        return preds

    def save(self, work_dir):
        checkpoint = {
            'bigrams': dict(self.bigrams),
            'trigrams': dict(self.trigrams), #saving the ngram values 
        }
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(checkpoint, f)
        print("N-gram checkpoint saved.")

    @classmethod
    def load(cls, work_dir):
        model = cls() #creating an empty model 
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['bigrams'].items()})
        model.trigrams = defaultdict(Counter, {k: Counter(v) for k, v in checkpoint['trigrams'].items()})
        model._load_gpt2(work_dir) #loading the gpt2 from the disk 
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

