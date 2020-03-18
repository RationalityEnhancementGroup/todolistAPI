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

For successful communication with the API, a `POST` request should be sent via the following URLs with a pre-defined body structure (described in detail in the report).

The general URL for local testing looks like this: 
`http://127.0.0.1:6789/api/<compulsoryParameters>/<additionalParameters>/<roundParameter>/tree/<userID>/<functionName>`

The general URL for testing the server online (on Heroku) looks like this
`http://<HerokuAppCode>.herokuapp.com/api/<compulsoryParameters>/<additionalParameters>/<points_per_hour>/<roundParameter>/tree/<userID>/<functionName>`

Compulsory parameters (`compulsoryParameters`) are:
- `method`: Method by which points are assigned
    - `constant`: Constant point assignment
    - `dp`: Calculates optimal points by using a dynamic programming algorithm [Reference](http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf) [pages: 5-8]
    - `greedy`: Calculates optimal points by using the a greedy algorithm to solve the problem. 
    - `length`: Length heuristics
    - `random`: Random point assignment from a standard normal distribution
- `scheduler`: Procedure by which tasks are scheduled
    - Schedulers for `constant`, `length` and `random` point-assignment methods:
        - `basic`: Basic scheduler
        - `deadline`: Deadline scheduler
    - Schedulers for `dp` and `greedy` point-assignment methods:
        - `mdp`: <i><u>Not used, but still necessary to have it as a URL input.</u></i>
- `default_duration`: Default task time estimation (in minutes) to fill in if it is not provided by the user.
- `default_deadline`: Default deadline (number of days, starting from today) to fill in if it is not provided by the user.
- `allowed_task_time`: Time-estimation restriction for tasks, so that users do not enter long time estimations. If no restriction to impose is necessary, then the input should be `inf`.
- `min_sum_of_goal_values`: Lower interval bound on the sum of goal values. 
- `max_sum_of_goal_values`: Upper interval bound on the sum of goal values.
- `min_goal_value_per_goal_duration`: Lower interval bound on the ratio between a goal value and its duration (in minutes).
- `max_goal_value_per_goal_duration`: Upper interval bound on the ratio between a goal value and its duration (in minutes).
- `<additionalParameters>`: (Differ for each method. Described in their own section.)
- `points_per_hour`: if 'true'-valued (`true`, `t`, `1`), we assign points per hour. otherwise, we assign points for task completion.
- `roundParameter`: The number of decimals to round to. For input of 0, the points will be rounded to the closest integer.
- `userID`: Unique user identification code.
- `functionName`: Type of request.
  - `updateTree`: Updates the stored tree.
  - `getTasksForToday`: Outputs list of task for today.

The additional parameters are dependent on the method that has been used. They are described in the following (sub-)sections. 
Important: The order of all the parameters provided in the URL matters!

You can use our URL generator to get the general URL to post to, before the last three parameters (userID, tree, and function): [https://aqueous-hollows-34193.herokuapp.com/static/urlgenerator.html](https://aqueous-hollows-34193.herokuapp.com/static/urlgenerator.html) Please submit a Github issue if you encounter any problems with this generator.

### Constant point-assignment point-assignment method (`const`)
- Additional parameters
  - `default_task_value`: Constant value of points to be assigned to each task.

- URL example: `http://127.0.0.1:6789/api/constant/basic/30/14/inf/0/1000/40/60/10/cite/tree/u123/getTasksForToday`

### Dynamic programming point-assignment method (`dp`)
- Additional parameters
    - `mixing_parameter`:
        - Basically, it represents a level of mixing tasks from different goals. That is, the level rigidity/flexibility of a user to work on different goals in a (relatively) short time period. 
        - It is a value between 0 (included; represents rigidity/no mixing) and 1 (excluded; represents flexibility complete mixing).
    - `scale_type` (optional): It represents the method by which points are scaled. If no scaling to be used, the inputting this parameter (and the `scale_min` and `scale_max` parameters) should be omitted.
        - `no_scaling`: Points are assigned according to this formula `task_reward = (goal_reward / goal_time_est) * task_time_est`. 
        - `min_max`: Points are assigned according to this formula `task_reward = (task_reward - min_value) / (max_value - min_value) * (scale_max - scale_min) + scale_min`.
        - `mean_value`: Points are assigned according to this formula `task_reward = (task_reward - mean_reward) / (max_value - min_value) * (scale_max - scale_min) / 2 + ((scale_max + scale_min) / 2)`
    - `scale_min` (optional): It represents the lower interval bound, which scales the proposed task values to the provided interval. If `inf`, then the lower interval bound is not set.
    - `scale_max` (optional): It represents the higher interval bound, which scales the proposed task values to the provided interval. If `inf`, then the upper interval bound is not set.

- URL example: `http://127.0.0.1:6789/api/dp/mdp/30/14/inf/0/1000/40/60/0/min_max/5/10/cite/tree/u123/getTasksForToday`

### Greedy-algorithm point-assignment method (`greedy`)
- Additional parameters
    - (Same as the `dp` method. See above.)

- URL example: `http://127.0.0.1:6789/api/greedy/mdp/30/14/inf/0/1000/40/60/0/min_max/5/10/cite/tree/u123/getTasksForToday`

### Length heuristics point-assignment method (`length`)
- There are no additional parameters for this method/

- URL example: `http://127.0.0.1:6789/api/length/deadline/30/14/inf/0/1000/40/60/cite/tree/u123/getTasksForToday`

### Random point-assignment method (`random`)
- Additional parameters
  - `distribution`: The name of the probability distribution (according to NumPy) and their own parameters. So far, these distributions have been implemented:
    - `uniform`: Uniform distribution with parameters `low` (lower interval bound) and `high` (higher interval bound)
    - `normal`: Normal (Gaussian) distribution with parameters `loc` (mean value) and `scale` (standard deviation)
    
- URL example: `http://127.0.0.1:6789/api/random/deadline/30/14/inf/0/1000/40/60/uniform/1/100/cite/tree/u123/getTasksForToday`
  
## Required Python Packages

All the required Python packages are listed in the `requirements.txt` file.
