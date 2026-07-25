# Reproduces every quantitative claim and the hero figure of Paper 2 from the
# deposited result JSONs. Build from the PAPER directory (one level up) so the
# result files are in the build context:
#
#   docker build -f repro/Dockerfile -t paper2-repro .
#   docker run --rm paper2-repro
#
# Expected output: "NUMBER AUDIT: 23/23 PASS" and a regenerated check figure.
# No simulator, GPU, or network is required: the forward-model runs are deposited
# as result files and this image reproduces the analysis layer on top of them.

FROM python:3.11-slim

WORKDIR /paper2
COPY repro/requirements.txt /paper2/repro/requirements.txt
RUN pip install --no-cache-dir -r /paper2/repro/requirements.txt

# deposited result files (the analysis inputs) + the repro scripts
COPY results/ /paper2/results/
COPY repro/ /paper2/repro/

WORKDIR /paper2/repro
ENV PAPER2_DATA=/paper2/results
CMD ["python", "regenerate_all.py"]
