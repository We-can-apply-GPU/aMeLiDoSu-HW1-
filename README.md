Usage
------

* When initializing , use `./train.py random`,

then , it will set the weights between layers to  Gaussian RVs and set biases to zero.

* If you have the pre-trained model , put it in model 
  (<i>But you need to ensure the number of neurons in each layer 
 is corresponding to the current setting in train.py</i>)
 
 then use `./train.py <modelName>`

* Both method to execute `./train.py` will produce the tmp file each 10 epochs ,

  the fileName is <b>accuracyx#OfDatum</b>,

  you can easily move these files to <b>model</b> directory to do more training.



* To predict test.ark , use `./predict.py <modelName>` ,

  it will generate the csv file in output directory

Environment setting : 
-----------------------

1. edit ~/.theanorc as following:
    [global]
    floatX = float32
    device=cpu

    [nvcc]
    fastmath=True

2. make sure you have the directorys called output,tmpModel,model,data

3. in train.py , 
   you can modify "sizes" to change number of layers , 
   also the number of neurons in each layer

4. in train.py , you can modify "BATCHSIZE" to change the number of data in each batch

Package dependency
---------------------
*  python3.4

    * Theano==0.7.0

    * nose==1.3.1

    * numpy==1.8.2

    * pandas==0.13.1

    * scipy==0.13.3

    * python3.4-dev

* g++

* BLAS

