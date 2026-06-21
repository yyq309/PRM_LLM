FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN printf '%s\n' \
        'deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware' \
        'deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware' \
        'deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware' \
        > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffuf \
        gobuster \
        nikto \
        nmap \
        sqlmap \
        whatweb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["bash"]
