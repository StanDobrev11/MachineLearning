# Use the official Miniconda image as the base
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment file to the container
COPY environment.yml .

# Create the Conda environment using the provided environment.yml file
RUN conda env create -f environment.yml && \
    conda clean -afy && \
    conda init bash

# Activate the environment and ensure it's used as the default in the container
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Expose the Jupyter Lab default port
EXPOSE 8888

# Start Jupyter Lab as the default command
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
