# Use the official NVIDIA TensorFlow image as the base.
# This container already has TensorFlow, CUDA, and a matching CuDNN.
# The 25.02 tag provides TensorFlow ~2.17.0, which is compatible with Sionna.
# Dockerfile

# Use the official NVIDIA TensorFlow image as the base.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# --- START: Timezone Configuration (Minimal) ---
# Set the timezone to your local timezone (e.g., America/New_York for Worcester, MA)
# This assumes the necessary zoneinfo files are already present in the base image.
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# --- END: Timezone Configuration (Minimal) ---

# Copy the requirements file
COPY requirements-minimal.txt .

# Install the additional packages from the requirements file
RUN python3 -m pip install --no-cache-dir -r requirements-minimal.txt

# Copy all your project files (main.py, start.sh, etc.) into the container
COPY . .

# Make the start script executable
#RUN chmod +x start.sh

# Expose the default TensorBoard port (Kept commented as in your provided file)
#EXPOSE 6006

# Set the start script as the default command when the container starts
#CMD ./start.sh && /bin/bash
CMD ["python3", "main.py"]