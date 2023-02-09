FROM ubuntu:18.04


FROM python:3.8

FROM pytorch/pytorch:latest

RUN mkdir -p /avg/home/AVG-NIA/HRNet

WORKDIR /avg/home/AVG-NIA/HRNet


#install dependencies

RUN pip install Cython==0.29.28
RUN pip install easydict==1.7
RUN pip install json_tricks==3.15.5
RUN pip install ninja==1.10.2.3
RUN pip install opencv-python==4.5.5.62
RUN pip install pandas==1.4.1
RUN pip install pyyaml==6.0
RUN pip install scikit-image==0.19.2
RUN pip install scipy==1.8.0
RUN pip install shapely==1.8.1.post1
RUN pip install tensorboardX==2.5
RUN pip install torch==1.7.0
RUN pip install torchvision==0.8.0 
RUN pip install torchaudio==0.7.0 
RUN pip install tqdm==4.63.0
RUN pip install yacs==0.1.8

#define the port number the container should expose
EXPOSE 5000

#run the train command

CMD ["python", "-m torch.distributed.launch", "--nproc_per_node=4", "code/train.py", "--cfg file/NIA.yaml"]


