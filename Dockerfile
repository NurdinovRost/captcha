# python
FROM python:3.7

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make

# install pip
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/captcha


# copy project
COPY . .

# install requirements project
RUN pip install --no-cache-dir -r requirements.txt

# ENVs
ENTRYPOINT ["python","./main.py"]


