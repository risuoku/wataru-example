FROM wataru-analysis-base
ENV APPLICATION_ENV docker

ENV JUPYTER_PATH /dockerwork/.jupyter
ENV IPYTHONDIR /dockerwork/.ipython
ENV PYTHONPATH /dockerwork/modules

COPY . .

RUN pip3 install -r requirements.txt
