# docker run -it -v `pwd`:/space/ -p 8888:8888 --name keras -w /space/ --rm bethgelab/jupyter-torch:ubuntu-14.04 bash
docker run -it -v `pwd`:/space/ -p 8888:8888 -p 6006:6006 --name dl -w /space/ --rm utensil/dl:models_notop bash
