import argparse
import numpy as np
import sys
import os
#import levenshtein
import codecs
import nltk
import operator
#from remove_tags import *
import pdb

word_tokenizer = nltk.TreebankWordTokenizer()

def build_inverted_index(source_sentences):
    print('Computing token IDFs...')
    token_counts = {}
    num_tokens = 0
    for sentence in source_sentences:
        for tok in sentence:
            if tok not in token_counts:
                token_counts[tok] = 0
            token_counts[tok] += 1
            num_tokens += 1

    # Discard the K most frequent tokens.
    freq_thres = 100
    terms = list(token_counts.keys())
    freqs = list(token_counts.values())
    ind = list(np.argsort(freqs))
    ind.reverse()
    #import pdb; pdb.set_trace()
    terms = [terms[i] for i in ind]
    for i in range(freq_thres):
        tok = terms[i]
        print('Stopword: %s' % tok)
        del token_counts[tok]

    token_idfs = {}
    for tok in token_counts:
        count = token_counts[tok]
        idf = np.log(float(num_tokens)) - np.log(float(count))
        token_idfs[tok] = idf

    print('Building inverted index...')
    num_tokens = 0
    token_sents = {}
    token_freqs = {}
    doc_denoms = []
    for i, sentence in enumerate(source_sentences):
        term_freqs = {}
        for tok in sentence:
            if tok not in token_idfs:
                continue
            if tok in term_freqs:
                term_freqs[tok] += 1
            else:
                term_freqs[tok] = 1
            num_tokens += 1

        doc_denom = 0.0
        for tok in term_freqs:
            freq = term_freqs[tok]
            if tok not in token_sents:
                token_sents[tok] = [i]
                token_freqs[tok] = [freq]
            else:
                token_sents[tok].append(i)
                token_freqs[tok].append(freq)
            tfidf = float(freq) * token_idfs[tok]
            doc_denom += tfidf * tfidf
        doc_denoms.append(np.sqrt(doc_denom))
        if i != 0 and i % 10000 == 0: print(i)

    return token_sents, token_freqs, token_idfs, doc_denoms

def compute_similarities(query, token_sents, token_freqs, token_idfs,
                         doc_denoms):
    query_term_freqs = {}
    for tok in query:
        if tok in query_term_freqs:
            query_term_freqs[tok] += 1
        else:
            query_term_freqs[tok] = 1
    sent_scores = {}
    sent_denom = 0.0
    for tok in query_term_freqs:
        if tok not in token_idfs:
            idf = 1e-12
        else:
            idf = token_idfs[tok]
        tfidf = float(query_term_freqs[tok]) * idf
        sent_denom += tfidf * tfidf
        if tok not in token_idfs:
            continue
        sents = token_sents[tok]
        freqs = token_freqs[tok]
        for sent, freq in zip(sents, freqs):
            tfidf_sent = float(freq) * idf
            score = tfidf * tfidf_sent / doc_denoms[sent]
            if sent not in sent_scores:
                sent_scores[sent] = score
            else:
                sent_scores[sent] += score

    sent_denom = np.sqrt(sent_denom)
    sents = list(sent_scores.keys())
    values = list(sent_scores.values())
    ind = list(np.argsort(values))
    ind.reverse()
    sents = [sents[i] for i in ind]
    values = [values[i]/sent_denom for i in ind]

    return sents, values

def main():
    '''Main function.'''
    # Parse arguments.
    parser = argparse.ArgumentParser(
        prog='Search engine',
        description='Indexes source sentences (one per line, paired with \
        target sentences) and runs a set of query sentences to find the \
        closest matches. Returns files with closest source/target sentences, \
        along with their scores.')

    # Can also be 'quality_estimation' and 'entity_tagging'.
    parser.add_argument('-source_sentences', type=str, default='')
    parser.add_argument('-target_sentences', type=str, default='')
    parser.add_argument('-query_sentences', type=str, default='')
    parser.add_argument('-num_neighbors', type=int, default=4)
    parser.add_argument('-tm_filepath', type=str, default='')

    args = vars(parser.parse_args())
    print(args, file=sys.stderr)

    source_sentences = args['source_sentences']
    target_sentences = args['target_sentences']
    query_sentences = args['query_sentences']
    num_neighbors = args['num_neighbors']
    tm_filepath = args['tm_filepath']

    if query_sentences == '':
        query_sentences = source_sentences

    source_data = []
    with codecs.open(source_sentences, 'r', 'utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            #line, _ = remove_tags(line)
            sentence = word_tokenizer.tokenize(line)
            source_data.append(sentence)

    target_data = []
    with codecs.open(target_sentences, 'r', 'utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            #line, _ = remove_tags(line)
            sentence = word_tokenizer.tokenize(line)
            target_data.append(sentence)

    assert len(source_data) == len(target_data)

    # Build inverted index.
    token_sents, token_freqs, token_idfs, doc_denoms = \
        build_inverted_index(source_data)

    data = open(query_sentences, encoding="utf-8").readlines()
    f = open(tm_filepath, 'w', encoding="utf-8")
    for pos, s in enumerate(data):
        s = word_tokenizer.tokenize(s.rstrip('\n'))

        sents, values = compute_similarities(s, token_sents, token_freqs,
                                             token_idfs, doc_denoms)

        #distances = []
        #for source in [source_data[i] for i in sents[:k]]:
        #    distance, _ = levenshtein.compute_minimum_edit_distance(target,
        #                                                            source)
        #    distances.append(distance)
        #if len(distances) == 0:
        #    continue
        #i = np.argmin(np.array(distances))

        #if distances[i] > 0.4 * len(target):
        #    continue

        #print 'Target:\n\t%s' % ' '.join(target).encode('utf8')
        #print 'Closest match (%f):\n\t%s' % (distances[i],
        #                                  ' '.join(source_sentences[sents[i]]).
        #                                  encode('utf8'))
        #for sent, value in zip(sents[:k], values[:k]):
        #    print 'Match (%f): %s' % (value,
        #                              ' '.join(source_sentences[sent]).
        #                              encode('utf8'))

        #print
        line = '\t'.join([' '.join(source_data[sent])
                          for sent in sents[:num_neighbors]])
        line += '\t' + '\t'.join([' '.join(source_data[sent])
                                  for sent in sents[:num_neighbors]])
        line += '\t' + '\t'.join(str(value) for value in values[:num_neighbors])
        f.write(line + '\n')
        results = sents[:num_neighbors], values[:num_neighbors]
        #print(pos)
        if (pos != 0) and (pos % 10000 == 0): print(pos)

    #results = retrieve_sentences( sentence=u"looking for a job" , limit1=500 , limit2=5)
    #print(results)

    f.close()

if __name__ == "__main__":
    main()
