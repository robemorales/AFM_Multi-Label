import torch.utils.data as data
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

def read_list(root, fileList):
    ingsList = []
    imgList = []
    c_dict = []
    k=-1

    with open(root + '/meta/ingredients_simplified.txt', 'r') as ingsFile:
        for line in ingsFile:
            ingsList.append(line.strip().split(","))

    ingsMLB = mlb.fit(ingsList)
    #print(ingsMLB.classes_, len(ingsMLB.classes_))
    ingsOneHot = ingsMLB.transform(ingsList)
    #print(ingsOneHot)
    with open(root + '/' + fileList, 'r') as file:
        for line in file.readlines():
            row = line.strip().split('\t')
            if len(row) == 1:
                label_name, _ = row[0].strip().split('/')
                imgP = line.strip()

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = ingsOneHot[k] #k

                    else:
                        label = ingsOneHot[k] #k
                    imgPath = root + '/images/' + imgP
                    imgList.append((imgPath, label))
                    #imgList.append((imgPath, int(label)))
            else:
                imgP = row[0]
                label_name, _ = row[0].strip().split('/')

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = ingsOneHot[k] #k

                    else:
                        label = ingsOneHot[k] #k
                    imgPath = root + '/images/' + imgP
                    #imgList.append((imgPath, int(label)))
                    imgList.append((imgPath, label))

    return imgList

class Food101N(data.Dataset):
    def __init__(self, root, transform):
        self.imgList = read_list(root, 'meta/imagelist.tsv')
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(imgPath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

"""
if __name__ == "__main__":

    food101nDS = Food101N("../data/Food-101N_release", None)
    for i in food101nDS:
        print("holi")
        break
"""
