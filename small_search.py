import sys
from files.porter import PorterStemmer
import math
from collections import Counter, defaultdict


class BM25(object):
    def __init__(self, docs):
        self.docs = docs
        self.doc_num = len(docs) # Total number of documents in the collection
        self.vocab = set([word for doc in self.docs for word in doc]) # all words in all documents with no repeat
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num # average length of a document in the collection
        # Constants for BM25 formula
        self.k1 = 1
        self.b = 0.75

        self.get_qn()
        self.score_dict = defaultdict(lambda: [])

    def get_qn(self):
        self.qn = {}  # all words in all documents
        for doc in self.docs:
            doc_set = set(doc)  # unduplicated collections for doc
            for word in doc_set:  # Calculate the term frequency in all documents
                if word in self.qn:
                    self.qn[word] += 1
                else:
                    self.qn[word] = 1

    def idf(self, word):
        if word not in self.vocab:
            word_idf = 0
        else:  # Common words are less important (similar to IDF)
            word_idf = math.log2((self.doc_num - self.qn[word] + 0.5) / (self.qn[word] + 0.5))
        return word_idf

    def score(self, word):
        score_list = []
        for index, doc in enumerate(self.docs):
            word_count = Counter(doc)
            if word in word_count.keys():  # frequency of the term appears in the document
                f = word_count[word] + 0.0
            else:
                f = 0.0
            # Calculate similarity
            r_score = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))  #
            score_list.append(self.idf(word) * r_score)
        return score_list

    # Calculate score
    def score_all(self, sequence):
        sum_score = []
        for word in sequence:
            score_list = []
            if word in self.score_dict:
                score_list = self.score_dict[word]
            else:
                score_list = self.score(word)
                self.score_dict[word] = score_list
            sum_score.append(score_list)
        sim = [0] * len(score_list)  # Generate score list
        for score in sum_score:
            for idx in range(len(score)):
                sim[idx] += score[idx]
        return sim


import os

# Checking text attributes
documents_dir = "./documents/"

# Check if it is a number
def is_number(uchar):
    if uchar >= '0' and uchar <= '9':
        return True
    else:
        return False

# Check if it is an alphabet
def is_alphabet(uchar):
    if (uchar >= 'a' and uchar <= 'z') or (uchar >= 'A' and uchar <= 'Z'):
        return True
    else:
        return False

# Check if it is a space
def is_space(uchar):
    if uchar == ' ' or uchar == '\t' or uchar == '\n':
        return True
    else:
        return False


# retain letters, numbers, and space characters
def format_str(content):
    content_str = ''
    for i in content:
        if is_space(i) or is_number(i) or is_alphabet(i):
            content_str = content_str + i
    return content_str


def get_document_list(documents_dir):
    doc_list = []
    file_name_list = []
    for file_name in os.listdir(documents_dir):
        try:
            # for file_name in os.listdir(documents_dir + "/" + documents_dir1):
            document_file_name = documents_dir + "/" + file_name
            content = ""
            with open(document_file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            file_name_list.append(file_name)
            # print("1111", file_name)
            content = content.replace("\n", " ")
            doc_list.append(content)
        except:
            pass
    doc_list = [format_str(doc).split(" ") for doc in doc_list]
    return doc_list, file_name_list

# Read and remove stopwords in the document.
def remove_stopwords(doc_list, stopwords_filename='./files/stopwords.txt'):
    stopwords = open(stopwords_filename).read().split('\n')  # Reading stopwords
    stopwords = set(stopwords)
    remove_stopwords_doc_list = []
    for doc in doc_list:
        new_doc = []
        for word in doc:
            if word not in stopwords:  # if the word is not stopwords
                new_doc.append(word)  # retain it
        
        remove_stopwords_doc_list.append(new_doc)  # get new document list without stopwords
    return remove_stopwords_doc_list

# Do stemming
def get_stemmer(doc_list):
    poster_stemmer = PorterStemmer()
    poster_stemmer_doc_list = []
    for doc in doc_list:
        new_doc = []
        # stem the word in each document
        for word in doc:
            word = poster_stemmer.stem(word)
            new_doc.append(word)
        poster_stemmer_doc_list.append(new_doc)
    return poster_stemmer_doc_list

# Save final document list
def save_doc_list(doc_list, file_name_list, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for doc in doc_list:
            file.write(" ".join(doc) + "\n")
        file.write("#######\n")
        for file_name in file_name_list:
            file.write(file_name + "\n")

# Get document list
def get_doc_list(file_name):
    doc_list = []
    file_name_list = []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [item.strip() for item in data]
    flag = True
    for item in data:
        if item == "#######":
            flag = False
            continue
        if flag:
            doc_list.append(item.split(" "))
        else:
            file_name_list.append(item)
    return doc_list, file_name_list


# Get docs
doc_list_file_name = "index.txt"
# If index.txt has been created, which means the program is not run for the first time
if os.path.exists(doc_list_file_name):
    print("Loading BM25 index from file, please wait.")
    doc_list, file_name_list = get_doc_list(doc_list_file_name)  # Read index.txt
else:  # If the program is run for the first time,
    print("Create BM25 index")   # Call the preprocessor function to generate the index text
    doc_list, file_name_list = get_document_list(documents_dir)
    doc_list = remove_stopwords(doc_list)
    doc_list = get_stemmer(doc_list)

    save_doc_list(doc_list, file_name_list, doc_list_file_name)

bm = BM25(doc_list)

# Sort by results and generate the top 15 results
def get_closen_doc(query, doc_list, bm, topn=15, is_show=True):
    if is_show:
        print("Results for query [{}]".format(query))

    doc_list = [format_str(query).split(" ")]
    doc_list = get_stemmer(doc_list)
    doc_list = remove_stopwords(doc_list)
    close_score = bm.score_all(doc_list[0])
    file_name_close_score = [[file_name, float(score)] for file_name, score in zip(file_name_list, close_score)]
    # Sorting results according to sim() from high to low
    file_name_close_score = sorted(file_name_close_score, key=lambda x: x[1], reverse=True)
    file_name_close_score = file_name_close_score[:topn]

    topn_close_score_file_name = [item[0] for item in file_name_close_score]
    topn_close_score = [item[1] for item in file_name_close_score]
    return topn_close_score_file_name, topn_close_score

# Outputting the top 15 results
def show_closen_doc(topn_close_score_index, topn_close_score):
    index = 1
    for close_score_index, close_score in zip(topn_close_score_index, topn_close_score):
        print("{}\t{}\t{}".format(index, close_score_index, close_score))
        index += 1

# Execute the query command
def manual_fun():
    query = input("Enter query:")
    while query != "QUIT":
        if query:
            try:
                topn_close_score_index, topn_close_score = get_closen_doc(query, doc_list, bm, 15)
                show_closen_doc(topn_close_score_index, topn_close_score)
            except:
                print("Please re-enter")
                pass
        query = input("Enter query:")


def evaluate_fun():
    # predict save to output.txt
    with open("./files/queries.txt", 'r', encoding='utf-8') as f:
        queries = f.readlines()
    index_list = [item.strip().split(" ")[0] for item in queries]
    sentences = [" ".join(item.strip().split(" ")[1:]) for item in queries]
    with open("./files/output.txt", 'w', encoding='utf-8') as f:
        index = 0
        for query in sentences:
            topn_close_score_index, topn_close_score = get_closen_doc(query, doc_list, bm, 15, False)
            sub_index = 1
            for close_score_index, close_score in zip(topn_close_score_index, topn_close_score):
                f.write(
                    "{} Q0 {} {} {} 19206227\n".format(index_list[index], close_score_index, sub_index, close_score))
                sub_index += 1
            index += 1

    # Prepare for evaluation
    from collections import defaultdict
    with open("./files/output.txt", 'r', encoding='utf-8') as f:
        output = f.readlines()
    output = [item.strip().split(" ") for item in output]

    # Read the relevance of the document corresponding to the query
    with open("./files/qrels.txt", 'r', encoding='utf-8') as f:
        qrels = f.readlines()
    qrels = [item.strip().split(" ") for item in qrels]

    output_dict = defaultdict(lambda: [])
    qrels_dict = defaultdict(lambda: [])

    for item in output:
        output_dict[str(item[0])].append(item[2])
    for item in qrels:
        qrels_dict[str(item[0])].append(item[2])

    P = 0
    R = 0
    for index in output_dict.keys():
        index = str(index)
        o_l = output_dict.get(index)  # Ret
        q_l = qrels_dict.get(index)  # Rel
        # Precision
        for o in o_l:
            r = 0
            if o in q_l:
                r += 1

            P += (r / len(o_l))
            R += (r / len(q_l))
        # Recall
        # for q in q_l:
        #     r = 0
        #     if q in o_l:
        #         r += 1

    print("Precision:{}".format(P / len(output_dict)))
    print("Recall:{}".format(R / len(output_dict)))

    # Calculate P@10
    P_at_10 = 0
    for index in output_dict.keys():
        index = str(index)
        o_l = output_dict.get(index)
        q_l = qrels_dict.get(index)

        # P
        for o in o_l[:10]:
            r = 0
            if o in q_l:
                r += 1
            P_at_10 += (r / 10)
    print("P@10:{}".format(P_at_10 / len(output_dict)))

    # Calculate R-precision
    R_precision = 0
    for index in output_dict.keys():
        index = str(index)
        o_l = output_dict.get(index)
        q_l = qrels_dict.get(index)

        # P
        for o in o_l[:len(q_l)]:
            r = 0
            if o in q_l:
                r += 1
            R_precision += (r / len(q_l))
    print("R-precision:{}".format(R_precision / len(output_dict)))

    # Calculate MAP
    MAP = 0
    for index in output_dict.keys():
        index = str(index)
        o_l = output_dict.get(index)
        q_l = qrels_dict.get(index)

        # P
        index = 1
        r = 0
        AP = 0
        for o in o_l:
            if o in q_l:
                r += 1
                AP += (r / index)
            index += 1
        AP /= len(q_l)
        MAP += AP
    print("MAP:{}".format(MAP / len(output_dict)))

    # Calculate bpref
    bpref = 0
    for index in output_dict.keys():
        index = str(index)
        o_l = output_dict.get(index)
        q_l = qrels_dict.get(index)

        # P
        r = 0  # number of un
        AP = 0
        for o in o_l:
            if o not in q_l:
                if r >= len(q_l):
                    break
                else:
                    r += 1
            if o in q_l:
                AP += (1 - r / len(q_l))

        bpref += AP / len(q_l)
    print("bpref:{}".format(bpref / len(output_dict)))

    output_rank_dict = defaultdict(lambda: [])
    qrels_rank_dict = defaultdict(lambda: defaultdict(lambda: ""))
    for item in output:
        output_rank_dict[item[0]].append(item[2])
    for index, item in enumerate(qrels):
        try:
            qrels_rank_dict[item[0]][item[2]] = int(item[3])
        except:
            pass

    def cal_dcg(y_true, y_pred, k):
        dcg = 0
        for i, (score, label) in enumerate(
                sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)[:k]):
            dcg += (math.pow(2, label) - 1) / math.log(i + 2, 2)
        return dcg

    def cal_ndcg(y_true, y_pred, k):
        dcg = cal_dcg(y_true, y_pred, k)
        ideal_dcg = cal_dcg(y_true, y_true, k)
        if ideal_dcg == 0:
            return 0
        return dcg / ideal_dcg

    # Calculate ndcg
    ndcg = 0
    for index in output_rank_dict.keys():
        index = str(index)
        o_l = output_rank_dict.get(index)
        q_l = qrels_rank_dict.get(index)
        y_true = []
        y_pred = []
        for index, o in enumerate(o_l):
            y_pred.append(len(o_l) - index)
            y_true.append(q_l.get(o, 0))
        try:
            ndcg += cal_ndcg(y_true, y_pred, 15)
        except:
            pass
    print("NDCG:{}".format(ndcg / len(output_rank_dict)))


if __name__ == "__main__":
    if sys.argv[2] == "manual":
        manual_fun()
    elif sys.argv[2] == "evaluation":
        evaluate_fun()
