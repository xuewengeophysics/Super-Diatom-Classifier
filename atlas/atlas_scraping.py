# %%
# PDF handling
import PyPDF4 as pypdf
from PIL import Image
from PIL import ImageOps
# Regex
import re
# Plotting
import matplotlib.pyplot as plt
import cv2

# %%
input1 = pypdf.PdfFileReader(open("./data/atlas_LIST_uncompressed.pdf", "rb"))
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
            print(type(xObject[obj]))
            print(type(xObject[obj].getData()))
            print(np.frombuffer(xObject[obj].getData(), dtype=np.int8))
            # data = xObject[obj].getData()
            data = xObject[obj]._data
            # Setting mode
            color_space = xObject[obj]['/ColorSpace']
            print(color_space)
            print(color_space[1].getObject())
            print(color_space[1].getObject()[1].getObject())

            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                mode = "RGB"
            else:
                mode = "P"

            image_name = "./tmp/"+ name + "_" + str(i)

            
            # img = Image.frombytes(mode, size, data)
            # pix = np.array(img)
            # plt.figure(i)
            # plt.imshow(pix)
            # img.save(image_name + ".jpg")
            img = None
            if '/Filter' in xObject[obj]:
                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    # img = open(image_name + ".tiff", "wb")
                    # img.write(data)
                    # img.close()
                    # img = PIL.ImageOps.invert(img)
                    img.save(image_name + ".png")
                elif xObject[obj]['/Filter'] == '/DCTDecode':
                    img = open(image_name + ".jpg", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/JPXDecode':
                    img = open(image_name + ".jp2", "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                    img = open(image_name + ".tiff", "wb")
                    img.write(data)
                    img.close()
            else:
                print("Uncompressed")
                img = Image.frombuffer(mode, size, data, "raw", mode, 0, 1)
                # img = PIL.ImageOps.invert(img)
                img.save(image_name + ".png", quality=100)
            pix = np.array(img)
            plt.figure(i)
            # plt.imshow(pix)
            # nparr = np.frombuffer(data, dtype=np.int8)
            print(bytearray(data))
            # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # print(img_np)
            # plt.figure(i)
            # plt.imshow(img_np)
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
    break

# %%
import fitz
doc = fitz.open("./data/atlas_LIST_uncompressed.pdf")
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        print(type(xref), xref)
        # if pix.n < 5:       # this is GRAY or RGB
        #     pix.writePNG("./tmp/p%s-%s.png" % (i, xref))
        # else:               # CMYK: convert to RGB first
        #     pix1 = fitz.Pixmap(fitz.csRGB, pix)
        #     pix1.writePNG("./tmp/p%s-%s.png" % (i, xref))
        #     pix1 = None
        pix = None
# %%
