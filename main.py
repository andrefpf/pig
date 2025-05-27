from jpig.media.image import RawImage


def main():
    img = RawImage().load_file("./datasets/images/baboon.bmp")
    img.show()
    print(img.data.shape)
    print(img.width())
    print(img.height())
    print(img.channels())


if __name__ == "__main__":
    main()
