# %%
import PyPDF2
from PIL import Image

# %%
input1 = PyPDF2.PdfFileReader(open("./data/atlas_LIST_11.pdf", "rb"))
page0 = input1.getPage(0)
xObject = page0['/Resources']['/XObject'].getObject()


# %%
obj = "\Im1"
size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
data = xObject[obj].getData()
if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
    mode = "RGB"
else:
    mode = "P"

# %%
img = Image.frombytes(mode, size, data)
img.save(obj[1:] + ".png")