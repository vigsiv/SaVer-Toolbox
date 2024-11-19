# SAVER: A toolbox for SAmpling-based, probabilistic VERification of neural networks

This package provides a collection of sampling-based tools which allow one to verify neural networks just using samples. 

## Running the Package (using Docker): 

Please ensure you have docker installed on your computer following the instructions [here](https://docs.docker.com/get-docker/):

1. Download the package to your computer either by downloading the zip file from **Releases** on the right: 

2. Once downloaded open your terminal or command line interface and move to directory the files reside:
    ```
    cd folder_where_SaVer_Toolbox_is/SaVer_Toolbox
    ```
3. Run the followin Docker command to buld the image containing the package: 
    > **Note:** If you are using Linux, you may need to run the Docker command with `sudo docker` rather than just `docker`.
    ```
    docker build -t saver_toolbox .
    ```
4. For each example in the paper, run the following commands:
    
    - Feedforward Neural Network with ReLU Activations:
        ```
        docker run --rm -it -v ./:/current_run saver_toolbox sh -c "cd /current_run && python3 ./examples/feedForwardNeuralNetwork/2dNNOutputExample.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer_Toolbox/examples/feedForwardNeuralNetwork/ffNNExpectedOutput.txt` where two runs of the example are printed. 
        - Figure 5 will be produced under: `folder_where_SaVer_Toolbox_is/SaVer_Toolbox/examples/feedForwardNeuralNetwork/Figure5.png`
    - Image Classification:
        ```
        docker run --rm -it -v ./:/current_run saver_toolbox sh -c "cd /current_run && python3 ./examples/imageClassification/CNN_example.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer_Toolbox/examples/imageClassification/imageClassificationExpectedOutput.txt`
    - TaxiNet: 
        ```
        docker run --rm -it -v ./:/current_run saver_toolbox sh -c "cd /current_run && python3 ./examples/TaxiNet/TaxiNet.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer_Toolbox/examples/TaxiNet/taxiNetExpectedOutput.txt` where two runs of the example are printed. 
        - Figure 8 will be produced under: `folder_where_SaVer_Toolbox_is/SaVer_Toolbox/examples/TaxiNet/Figure8.png`