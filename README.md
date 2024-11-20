# SAVER: A toolbox for SAmpling-based, probabilistic VERification of neural networks

This package provides a collection of sampling-based tools which allow one to verify neural networks just using samples. 

## Running the Package (using Docker): 

Please ensure you have docker installed on your computer following the instructions [here](https://docs.docker.com/get-docker/).

1. Download the package to your computer either by downloading the zip file from:
    -  **Releases** on the right (if using GitHub). 
    - **Download Repository** on top right (if using Anonymous GitHub) 

2. Once downloaded, unzip the file. 
3. Open your terminal or command line interface and move to folder where the toolbox is:
    ```
    cd folder_where_SaVer_Toolbox_is/SaVer-Toolbox
    ```
3. Run the following Docker command to build or load the image using the Dockerfile for your system: 
    > **Note:** If you are using Linux, you may need to run the Docker command with `sudo docker` rather than just `docker`.
    
    **Load (Reccomended for Intel/AMD CPUs on Windows/macOS/Linux)**: 

    Run the following command: 
    ```
    docker load -i saver_toolbox.tar
    ```
    **Build from Dockerfile (Reccomended for Apple Silicon and ARM CPUs)**:
    ```
    docker build -t saver-toolbox .
    ``` 
4. For each example in the paper, run the following commands:
    
    - **Feedforward Neural Network with ReLU Activations:**
        
        macOS or Linux: 
        ```
        docker run --rm -it -v ./:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/feedForwardNeuralNetwork/2dNNOutputExample.py"
        ```
        Windows: 
        ```
        docker run --rm -it -v .\:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/feedForwardNeuralNetwork/2dNNOutputExample.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer-Toolbox/examples/feedForwardNeuralNetwork/ffNNExpectedOutput.txt` where two runs of the example are printed. 
        - Figure 5 will be produced under: `folder_where_SaVer_Toolbox_is/SaVer-Toolbox/examples/feedForwardNeuralNetwork/Figure5.png`
    - **Image Classification:**

        macOS or Linux:
        ```
        docker run --rm -it -v ./:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/imageClassification/CNN_example.py"
        ```
        Windows:
        ```
        docker run --rm -it -v .\:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/imageClassification/CNN_example.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer-Toolbox/examples/imageClassification/imageClassificationExpectedOutput.txt`
    - **TaxiNet:** 

        macOS or Linux: 
        ```
        docker run --rm -it -v ./:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/TaxiNet/TaxiNet.py"
        ```
        Windows:
        ```
        docker run --rm -it -v .\:/current_run saver-toolbox sh -c "cd /current_run && python3 ./examples/TaxiNet/TaxiNet.py"
        ```
        - You can compare the command line output with the file under (since the approach is sample-based there will be slight variation): `folder_where_SaVer_Toolbox_is/SaVer-Toolbox/examples/TaxiNet/taxiNetExpectedOutput.txt` where two runs of the example are printed. 
        - Figure 8 will be produced under: `folder_where_SaVer_Toolbox_is/SaVer-Toolbox/examples/TaxiNet/Figure8.png`


## Using your toolbox to your usecase: 

WIP

## 