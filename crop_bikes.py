from pig.utils.pgx_handler import PGXHandler
from PIL import Image

handler = PGXHandler()

array = handler.read("./datasets/images/Bikes/0/000_000.pgx")
array = array[:432, :624]

handler.write("./datasets/images/Bikes_cropped/0/000_000.pgx", array)

img = Image.fromarray((array >> 2).astype("uint8"))
img.save("./datasets/images/bikes_cropped.pgm")