# TODO: This is not tested enough, it's recommended to use local environment.

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt update -y

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
RUN echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

RUN curl -ssL https://magic.modular.com/a7d1b1a1-592e-4d8c-89d9-712e452f4be2 | bash

RUN echo 'eval "$(magic completion --shell bash)"' >> ~/.bashrc

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
