# python ver
FROM python:3.9

# install pre compiled opencv...
RUN apt-get update && apt-get install -y libopencv-dev

# set work dir
WORKDIR /Med_TA

# copy files into the dir
COPY . .

# python depedencies
RUN pip install --no-cache-dir -r requirements.txt

# admin execution
RUN chmod +x ./start.sh

# run script
CMD ["./start.sh"]