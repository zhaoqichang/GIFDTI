drug_dict = {}
with open("./KIBA_compound.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split()
        drug_dict[line[0]] = line[1]

protein_dict = {}
with open("./KIBA_protein.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split()
        protein_dict[line[0]] = line[1]

drug_num = {}
for drug in drug_dict.keys():
    drug_num[drug] = 0
protin_num = {}
for protein in protein_dict.keys():
    protin_num[protein] = 0
pair = []
with open("./KIBA.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split()
        pair.append("{} {}".format(line[0],line[1]))
        drug_num[line[0]] += 1
        protin_num[line[1]] += 1
    with open("./KIBA_unique_train.txt","w") as writefile1:
        with open("./KIBA_unique_test.txt","w") as writefile2:
            for line in lines:
                line = line.strip().split()
                if drug_num[line[0]] == 1 or protin_num[line[1]] == 1:
                    writefile2.write("{} {} {} {} {}\n".format(line[0],line[1],line[2],line[3],line[4]))
                else:
                    writefile1.write("{} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], line[4]))



