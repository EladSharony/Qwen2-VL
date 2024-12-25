# Use the existing image as the base
FROM qwenllm/qwenvl:2-cu121

RUN pip install --no-cache-dir ogbench \
    sentence-transformers==3.2.1  \
    autoawq==0.2.7  \
    transformers==4.46.3

COPY setup_ogbench.py /workspace/setup_ogbench.py
COPY run_qwen2.py /workspace/run_qwen2.py

# Set the working directory
WORKDIR /workspace

# Run the command
CMD ["python", "ogbench_dataset.py"]





