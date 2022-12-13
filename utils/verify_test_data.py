import os

if __name__ == "__main__":
    path1 = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/stage_2_images"
    path2 = "/home/dhgbao/VinBrain/Pneumothorax_Segmentation/dataset/stage2_pngs"
    
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    
    list1 = [x.split(".")[0] for x in list1]
    list2 = [x.split(".")[0] for x in list2]
    print(len(list1))
    print(len(list2))
    print(set(list2) - set(list1))