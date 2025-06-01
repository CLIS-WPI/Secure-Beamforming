# Use the official NVIDIA TensorFlow image as the base.
# This container already has TensorFlow, CUDA, and a matching CuDNN.
# The 25.02 tag provides TensorFlow ~2.17.0, which is compatible with Sionna.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# Copy ONLY your minimal requirements file (which does NOT list tensorflow)
COPY requirements-minimal.txt .

# Install only the additional packages. TensorFlow is already here.
RUN python3 -m pip install --no-cache-dir -r requirements-minimal.txt

# Copy all your other project files (like main.py) into the container
COPY . .

# Set the default command to run your script when the container starts
CMD ["python3", "main.py"]