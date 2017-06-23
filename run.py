import argparse
from src import process_image

if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument(
	    "-i", "--image", required=True, help="path to input image file")
	args = vars(ap.parse_args())
	process_image(args['image'])