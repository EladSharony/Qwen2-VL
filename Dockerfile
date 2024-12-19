# Use the existing image as the base
FROM qwenllm/qwenvl:2-cu121

RUN pip install --no-cache-dir ogbench

COPY ogbench_dataset.py /app/ogbench_dataset.py
COPY dataset_preprocess.py /app/dataset_preprocess.py

# Set the working directory
WORKDIR /app

# Run the command
CMD ["python", "ogbench_dataset.py"]




