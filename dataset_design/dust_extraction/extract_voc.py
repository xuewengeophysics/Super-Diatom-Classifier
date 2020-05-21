# %%
from lxml import etree
from os import listdir
from os.path import isfile, join
import cv2

# %%
def parse_annotation(annotation_path):
    annotation = {}
    tree = etree.parse(annotation_path)
    root = tree.getroot()
    annotation["folder"] = root.find("folder").text
    annotation["path"] = root.find("path").text
    annotation["filename"] = root.find("filename").text
    size_node = root.find("size")
    annotation["size"] = {
        "width": int(size_node.find("width").text),
        "height": int(size_node.find("height").text),
        "depth": int(size_node.find("depth").text)
    }
    annotation["objects"] = []
    for object_el in root.iterchildren(tag='object'):
        tmp = {}
        tmp["name"]=object_el.find("name").text
        bndbox = object_el.find("bndbox")
        tmp["xmin"]=int(bndbox.find("xmin").text)
        tmp["ymin"]=int(bndbox.find("ymin").text)
        tmp["xmax"]=int(bndbox.find("xmax").text)
        tmp["ymax"]=int(bndbox.find("ymax").text)
        annotation["objects"].append(tmp)
    return annotation

# %%
root_annotations = "C:\\Users\\pierr\\Desktop\\DIAMORPH\\annotations"
output_folder = ".\\data\\"
i=0
annotations_paths = [f for f in listdir(root_annotations) if isfile(join(root_annotations, f))]
for annotation_path in annotations_paths:
    if annotation_path.split(".")[-1]=="xml":
        annotation = parse_annotation(join(root_annotations,annotation_path))
        image_path = annotation["path"]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        for obj_bb in annotation["objects"]:
            ymin, xmin, ymax, xmax = obj_bb["ymin"], obj_bb["xmin"], obj_bb["ymax"], obj_bb["xmax"]
            patch = img[ymin:ymax, xmin:xmax]
            if not (patch.shape[0]==0 or patch.shape[1]==0):
                cv2.imwrite(join(output_folder, str(i)+".png"), patch)
                i+=1
print(i, "patches extracted!")

# %%
