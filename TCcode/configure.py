dataset = "sample"
savemodel = "models/"+dataset+".pkl"
loadmodel = "models/"+ dataset +".pkl-233"
savedset = "models/"+ dataset +".pkl.dset"

public_path = "data"
train_file = "train.txt"
dev_file = "valid.txt"
test_file = "test.txt"
relation2id = "class2id.txt"
char_emb_file = "vec.txt"
sense_emb_file = "sense.txt"
word_sense_map = "sense_map.txt"
max_length = 40

Encoder = 'GRU' # 'MGLattice' or 'GRU'
Optimizer = 'SGD' # 'SGD' or 'Adam'
lr = 0.016 # recommend: 0.015 for SGD ( with lr decay ) and 0.0005 for Adam
weights_mode = 'smooth'