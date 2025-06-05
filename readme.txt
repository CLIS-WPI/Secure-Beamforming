chmod +x start.sh
docker build -t sionna-with-tf-container .
docker run --gpus all -it --rm -v $(pwd):/workspace sionna-with-tf-container

Ctrl+Shift+P in vs code and search for Ports: Focus on Ports View and Forward a Port set 6006 
after that in local host go to http://localhost:6006
