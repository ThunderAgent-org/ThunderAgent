FROM python:3.12-bookworm

COPY ./ /opt/ThunderAgent/

RUN pip install --no-build-isolation --no-cache-dir "setuptools>=64" wheel && \
    pip install --no-build-isolation --no-cache-dir -e /opt/ThunderAgent/
