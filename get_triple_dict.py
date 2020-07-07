from pprint import pprint
from collections import defaultdict

with open('triplets', 'r') as f:
    doc_text = f.read()

doc_list = [i for i in doc_text.split('\n') if i]  # get every non-empty newline
doc_list = [i for i in doc_list if not i[0].isupper()]  # remove redundant lines

# fixing going onto new line
del_rows = []
for i in range(1, len(doc_list)):
    if doc_list[i][0] != '-':  # checking if line starts with '-'
        doc_list[i-1] += doc_list[i]  # if not, adding this line to the previous
        del_rows.append(i)

final_list = [doc_list[i][2:] for i in range(len(doc_list)) if i not in del_rows]

# with open('new_triplets', 'w') as f:
#    f.write("\n".join(final_list))  # writing processed triplets into new file for manual checking

# Now we will create dictionary for all this features
pair_crosses_dict = defaultdict(list)
for string in final_list:
    triplet = string.split(' x ')
    
    # For now will only group by first feature of triplet
    pair_crosses_dict[triplet[0]].append(triplet[1:])

triple_crosses_dict = defaultdict(dict)
# Now we will create final dict for triple feature crosses
for key in pair_crosses_dict.keys():
    doubles = pair_crosses_dict[key]
    inner_dict = defaultdict(list)
    for double in doubles:
        second_term = double[1]
        if '[' in second_term:  # expanding terms: [123] -> 1, 2, 3
            opening_ind = second_term.index('[')
            closing_ind = second_term.index(']')
            all_variants = second_term[opening_ind+1:closing_ind]
            second_term = [double[1][:opening_ind] + var + \
                           double[1][closing_ind+1:] for var in all_variants]

        inner_dict[double[0]].append(second_term)
    triple_crosses_dict[key] = dict(inner_dict)

pprint(dict(triple_crosses_dict))
