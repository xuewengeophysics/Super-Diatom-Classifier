# %%
# PDF handling
import PyPDF4 as pypdf
from PIL import Image
# Regex
import re
# Plotting
import matplotlib.pyplot as plt

# %%
input1 = pypdf.PdfFileReader(open("./data/atlas_LIST.pdf", "rb"))
taxon_regex = "[A-Z]{4}"
taxons = {}
for section in input1.getOutlines():
    # Extracting the Taxon
    title = section["/Title"]
    p = re.compile(taxon_regex)
    match = p.search(title)
    if match:
        page_num = section['/Page']['/StructParents']+1
        taxons[match.group()] = page_num
taxons = {k: v for k, v in sorted(taxons.items(), key=lambda item: item[1])}

# %%
def extractImages(xObject, name):
    keys=[obj[0] for obj in xObject.items()]
    page_images = []
    i = 0
    for obj in keys:
        if xObject[obj]["/Subtype"] == "/Image":
            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
            data = xObject[obj].getData()

            # Setting mode
            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"

            image_name = "./tmp/"+ name + "_" + str(i)
            # Decoding
            if xObject[obj]['/Filter'] == '/FlateDecode':
                img = Image.frombytes(mode, size, data)
                img.save(image_name + ".png")
            elif xObject[obj]['/Filter'] == '/DCTDecode':
                img = open(image_name + ".jpg", "wb")
                img.write(data)
                img.close()
            elif xObject[obj]['/Filter'] == '/JPXDecode':
                img = open(image_name + ".jp2", "wb")
                img.write(data)
                img.close()
            i+=1

key_list = list(taxons.keys())
for i, name in enumerate(key_list):
    n_start = taxons[key_list[i]]
    n_end = taxons[key_list[i+1]]-1
    for page_num in range(n_start, n_end+1):
        print(name, page_num)
        page = input1.getPage(page_num)
        xObject = page['/Resources']['/XObject'].getObject()
        extractImages(xObject, name)

# %%
plt.imshow(img)
# img.save(obj[1:] + ".png")
