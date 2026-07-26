# Reproduces every quantitative claim and the hero figure of Paper 2 from the
# deposited result JSONs. Build from the PAPER directory (one level up) so the
# result files are in the build context:
#
#   docker build -f repro/Dockerfile -t repro .
#   docker run --rm repro
#
# Expected output: "NUMBER AUDIT: 23/23 PASS" and a regenerated check figure.
# No simulator, GPU, or network is required: the forward-model runs are deposited
# as result files and this image reproduces the analysis layer on top of them.

FROM python:3.11-slim

WORKDIR /deposit
COPY repro/requirements.txt /deposit/repro/requirements.txt
RUN pip install --no-cache-dir -r /deposit/repro/requirements.txt

# deposited result files (the analysis inputs) + the repro scripts
COPY results/ /deposit/results/
COPY repro/ /deposit/repro/

WORKDIR /deposit/repro
ENV RESULTS_DIR=/deposit/results
CMD ["python", "regenerate_all.py"]
