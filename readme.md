# Use Fast Api to Deploy the Pyspark Model

## Import the project into Pycharm or VS code 
    - Make sure all the dependencies are installed properly from requirements.txt

## Run the code from main.py to launch the fast api endpoint
    - Two ways of testing the endpoint
        - Run curl GET http://localhost:8000/ to test the endpoint is up and running, hello world should return
        - import the postman collection into your postman, /fastApiProject/DiamondTest.postman_collection.json
 
## Use a curl call or postman to make a Post call against the endpoint
     - Two ways of getting the predicted log price
        - curl -X POST -H "Content-Type: application/json" -d '{"carat": 0.2, "cut": "Premium", "color": "E", "clarity": "VS2", "depth": 61.1, "table": 59.0, "x": 3.81, "y": 3.78, "z": 2.32}' http://localhost:8000/logprice
        - import the postman collection into your postman, /fastApiProject/DiamondTest.postman_collection.json
