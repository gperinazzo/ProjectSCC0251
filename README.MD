# Final Project - SCC0251 Image Processing #
--------------------------------

## Requirements
To run the project, make sure you have python3 and virtualenv installed
```
sudo apt-get install python3 virtualenv
```
Create a virtualenv and install the python dependencies:
```
git clone https://github.com/gperinazzo/ProjectSCC0251.git
cd ProjectSCC0251
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```
This should install all the required dependencies.

## Running the demo
To run the demo, simply call the run.py script with the image as argument (-i)
```
python run.py -i images/test1.jpg
```
The resulting images will be placed in the project's folder.
