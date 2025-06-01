docker build -t sionna-with-tf-container .
docker run --gpus all -it --rm -v .:/workspace sionna-with-tf-container