This is the server for our future Complice experiments with Workflowy task lists.

# Local usage

In order to be able to run the code, you need to have a [mongo database](https://www.mongodb.com/).

After installing MongoDB, you should do the following steps:
- Open a terminal window and start a connection with the database with the command `mongod --dbpath <pathToDB> --port 27017`
- Open another terminal window and start the DB editor with the command `mongo`.
  - In order to create a user, initialize a collection etc., please refer to [MongoDB's official websie](https://www.mongodb.com/).
- In the `if __name__ == '__main__'` part of the `app.py` file, the following info should be provided. Here is an example:
    ```
    uri = "mongodb://ai4productivity:ai4productivity@127.0.0.1/ai4productivity"
    client = MongoClient(uri)
    db = client["ai4productivity"]
    collection = db["ai4productivity"]
    ```
- Start the application by typing `python app.py` in the root folder of the API.
- Send requests to the API. For this purpose, we recommend using [Postman](https://www.postman.com/) for sending requests.

## Sending requests

The general URL for local testing looks like this: 
`http://127.0.0.1:6789/api/<compulsoryParameters>/<additionalParameters>/tree/<userID>/<functionName>`

The general URL for testing the server online (on Heroku) looks like this
`http://safe-retreat-20317.herokuapp.com/api/<compulsoryParameters>/<additionalParameters>/tree/<userID>/<functionName>`

Compulsory parameters (`compulsoryParameters`) are:
- `method`: Method by which points are assigned
    - `constant`: Constant point assignment
    - `length`: Length heuristics
    - `random`: Random point assignment from a standard normal distribution
    - `dp`: Calculates optimal points by using a dynamic programming algorithm [Reference](http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf) [pages: 5-8]
    - `old-report`: Calculates optimal points by using the backward induction algorithm [Find reference!]() to solve the problem. 
- `scheduler`: Procedure by which tasks are scheduled
    - Schedulers for `constant`, `length` and `random` point-assignment methods:
        - `basic`: Basic scheduler
        - `deadline`: Deadline scheduler
    - Schedulers for `dp` and `old-report` point-assignment methods:
        - `mdp`: <i><u>Not used, but still necessary to have it as a URL input.</u></i>
- `default_duration`: Default task time estimation (in minutes) to fill in if it is not provided by the user.
- `default_deadline`: Default deadline (number of days, starting from today) to fill in if it is not provided by the user.
- `allowed_task_time`: Time-estimation restriction for tasks, so that users do not enter long time estimations. If no restriction to impose is necessary, then the input should be `inf`.
- `round_param`: If `cite`, then all points will be rounded on 2 decimals. For any other input, the points will be rounded to the closest integer.
- `user_key`: User ID according to <i><u>Complice or WorkFlowy?</u></i>
- `api_method`: Type of request
  - `updateTree`: Updates the stored tree.
  - `getTasksForToday`: Outputs list of task for today.

The additional parameters are dependent on the method that has been used. They are described in the following (sub-)sections. 
Important: The order of all the parameters provided in the URL matters!

### Constant point-assignment point-assignment method (`const`)
- Additional parameters
  - `default_task_value`: Constant value of points to be assigned to each task.

- URL example: `http://127.0.0.1:6789/api/constant/basic/30/14/inf/10/cite/tree/u123/getTasksForToday`

### Dynamic programming point-assignment method (`dp`)
- Additional parameters
    - `mixing_parameter`:
        - Basically, it represents a level of mixing tasks from different goals. That is, the level rigidity/flexibility of a user to work on different goals in a (relatively) short time period. 
        - It is a value between 0 (included; represents rigidity/no mixing) and 1 (excluded; represents flexibility complete mixing).
    - `scale_min` (optional): It represents the lower interval bound, which scales the proposed task values to the provided interval.
    - `scale_max` (optional): It represents the higher interval bound, which scales the proposed task values to the provided interval.

- URL example: `http://127.0.0.1:6789/api/dp/mdp/30/14/inf/0/5/10/cite/tree/u123/getTasksForToday`

### Length heuristics point-assignment method (`length`)
- There are no additional parameters for this method/

= URL example: `http://127.0.0.1:6789/api/length/deadline/30/14/inf/cite/tree/u123/getTasksForToday`

### Random point-assignment method (`random`)
- Additional parameters
  - `distribution`: The name of the probability distribution (according to NumPy) and their own parameters. So far, these distributions have been implemented:
    - `uniform`: Uniform distribution with parameters `low` (lower interval bound) and `high` (higher interval bound)
    - `normal`: Normal (Gaussian) distribution with parameters `loc` (mean value) and `scale` (standard deviation)
    
- URL example: `http://127.0.0.1:6789/api/random/deadline/30/14/inf/uniform/1/100/cite/tree/u123/getTasksForToday`
  
## Required Python Packages

All the required Python packages are listed in the `requirements.txt` file.

## Contact

If you encounter any errors, please contact us at <i><u>???</u></i>.