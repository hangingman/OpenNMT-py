#!/bin/bash

# Usage:
# ./run_search_eng.sh model training_data number_of_nearest_neighbors test.en <  test.fr
# translation memory comes to output


# TODO: finish this, it should print the sentences in target language by
# indeces from second field of fasttext output

tmp=`mktemp`

#sent2vec/fasttext nnSent $1 $2 $3 < $4 
cat t | cut -f 2 -d" " > $tmp

cat $tmp
echo python3 print_sent.py $4 $tmp
#python3 print_sent.py $5 $tmp

