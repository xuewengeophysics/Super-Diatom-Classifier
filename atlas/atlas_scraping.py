# %%
# PDF handling
import PyPDF2
from PIL import Image
# Regex
import re
# Plotting
import matplotlib.pyplot as plt

# %%
input1 = PyPDF2.PdfFileReader(open("./data/atlas_LIST.pdf", "rb"))
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
key_list = list(taxons.keys())
for i, name in enumerate(key_list):
    n_start = taxons[key_list[i]]
    n_end = taxons[key_list[i+1]]-1
    for page_num in range(n_start, n_end+1):
        page = input1.getPage(page_num)
        xObject = page['/Resources']['/XObject'].getObject()

# %%
keys=[obj[0] for obj in xObject.items()]
page_images = []
for obj in keys:
    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
    data = xObject[obj].getData()
    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
        mode = "RGB"
    else:
        mode = "P"
    page_images.append(Image.frombytes(mode, size, data))
for page_images)

# %%
plt.imshow(img)
# img.save(obj[1:] + ".png")

# %%
