# Checks the revised result-level evidence. Build from the repository
# root (one level up) so the result files are in the build context:
#
#   docker build -f repro/Dockerfile -t repro .
#   docker run --rm repro
#
# Legacy regenerate_all.py checks 72 historical fixtures separately.
# No simulator, GPU, or network is required: the forward-model runs are deposited
# as result files and this image reproduces the analysis layer on top of them.

FROM python:3.11-slim

WORKDIR /deposit
COPY repro/requirements.txt /deposit/repro/requirements.txt
RUN pip install --no-cache-dir -r /deposit/repro/requirements.txt

# deposited result files (the analysis inputs) + the repro scripts
COPY results/ /deposit/results/
COPY repro/ /deposit/repro/
COPY paper/ /deposit/paper/
COPY scripts/ /deposit/scripts/

WORKDIR /deposit/repro
ENV RESULTS_DIR=/deposit/results
CMD ["python", "verify_revision.py"]
