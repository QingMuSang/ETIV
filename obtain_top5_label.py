the_dict = dict()
with open('../../data/tmall/node2label.txt', 'r') as r:
    for line in r:
        part = line.split()
        label = int(part[1])
        if label in the_dict:
            the_dict[label] += 1
        else:
            the_dict[label] = 0
    print(sorted(the_dict.items(), key=lambda item: item[1], reverse=-1))

with open('../../data/tmall/node2label.txt', 'r') as r:
    with open('../../data/tmall/top5label.txt', 'w') as w:
        for line in r:
            part = line.split()
            if part[1] in ('662', '737', '656', '1505', '389'):
                w.write(str(part[0]) + ' ' + str(part[1]) + '\n')

