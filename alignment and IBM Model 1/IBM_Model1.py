from collections import defaultdict
import optparse
import sys
from itertools import chain


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                     help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                     help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

sys.stderr.write("Initialization begins......\n")

#print unique_word
#unique_word = 1.0 / unique_word
e_text = [sentence.strip().split() for sentence in open(e_data)][:opts.num_sents]
word_type = set(chain(*e_text))
prob_dict = defaultdict(lambda: float(1)/len(word_type))

sys.stderr.write("Initializing probabilities......\n")


for runtime in range(5):
    count_dict = defaultdict(int)
    e_dict = defaultdict(float)
    sys.stderr.write("\nIteration %d\n" % (runtime + 1))
    for (x, (sent_f, sent_e)) in enumerate(bitext):
        for f in sent_f:
            Z = 0
            for e in sent_e:
                Z += prob_dict[(f, e)]
            for e in sent_e:
                c = prob_dict[(f, e)] / Z
                count_dict[(f, e)] += c
                e_dict[e] += c
        if x % 1000 == 0:
            sys.stderr.write("%d / %d\r" % (x, len(bitext)))
    for (n, (f, e)) in enumerate(count_dict):
        prob_dict[(f, e)] = count_dict[(f, e)] / e_dict[e]
        if n % 500000 == 0:
            sys.stderr.write("%d / %d\r" % (n, len(count_dict)))

for (sent_f, sent_e) in bitext:
    for (f, wordf) in enumerate(sent_f):
        best_prob = 0
        best_e = 0
        for (e, worde) in enumerate(sent_e):
            if prob_dict[(wordf, worde)] > best_prob:
                best_prob = prob_dict[(wordf, worde)]
                best_e = e
        sys.stdout.write("%d-%d " % (f, best_e))
    sys.stdout.write("\n")

#print "\nDone"


