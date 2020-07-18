This is the code for an API which, given a to-do list, incentivizes tasks and schedules them. It will be used in future CompliceX experiments with Workflowy to-do lists.

# Local usage

In order to be able to run the code, you need to have a [mongo database](https://www.mongodb.com/).

After installing MongoDB, you should do the following steps:
- Open a terminal window and start a connection with the database with the command `mongod --dbpath <pathToDatabase> --port 27017`
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
- Send requests to the API. For this purpose, we recommend using [Postman](https://www.postman.com/).

## Sending requests

For successful communication with the API, a `POST` request should be sent via the following URLs with a pre-defined body structure (described in detail in the report).

The general URL for local testing looks like this: 
`http://127.0.0.1:6789/api/<compulsoryParameters>/<additionalParameters>/tree/<userID>/<functionName>`

The general URL for using the server online (on Heroku) looks like this:
`https://<HerokuAppCode>.herokuapp.com/api/<compulsoryParameters>/<additionalParameters>/tree/<userID>/<functionName>`

Description of the URL parameters:
- `<method>`: Method by which points are assigned
    - `constant`: Constant point assignment
    <!--
    - `dp`: Calculates optimal points by using a dynamic programming algorithm [Reference](http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf) [pages: 5-8]
    - `greedy`: Calculates optimal points by using the a greedy algorithm to solve the problem. 
    -->
    - `length`: Length heuristics
    - `random`: Random point assignment from a Normal distribution.
    - `smdp`: Calculates optimal points by using semi-Markov decision processes.
- `<scheduler>`: Procedure by which tasks are scheduled
    - Schedulers for `constant`, `length` and `random` point-assignment methods:
        - `basic`: Basic scheduler
        - `deadline`: Deadline scheduler
    - Schedulers for `smdp` <!-- `dp` and `greedy` --> point-assignment method:
        - `mdp`: Method used by the SMDP incentivizing method.
- `<compulsoryParameters>`
    - `default_time_est`: Default task time estimate (in minutes) to fill in if it is not provided by the user.
    - `default_deadline`: Default deadline (number of days, starting from today) to fill in if it is not provided by the user.
    - `allowed_task_time`: Time-estimation restriction for tasks, so that users do not enter long time estimations. If no restriction to impose is necessary, then the input should be `inf`.
    - `min_sum_of_goal_values`: Lower interval bound on the sum of goal values. 
    - `max_sum_of_goal_values`: Upper interval bound on the sum of goal values.
    - `min_goal_value_per_goal_duration`: Lower interval bound on the ratio between a goal value and its duration (in minutes).
    - `max_goal_value_per_goal_duration`: Upper interval bound on the ratio between a goal value and its duration (in minutes).
    - `points_per_hour`: if 'true'-valued (`true`, `t`, `1`), we assign points per hour. otherwise, we assign points for task completion.
    - `rounding`: The number of decimals to round to. For input of 0, the points will be rounded to the closest integer.
- `<additionalParameters>`: (Differ for each method. Described in their own section.)
- `<userID>`: Unique user identification code.
- `<functionName>`: Type of request.
    - `updateTree`: Updates the stored tree.
    - `getTasksForToday`: Outputs list of task for today.

The additional parameters (`<additionalParameters>`) are dependent on the method that has been used (described below). 
Important: The order of all URL parameters is **fixed**!

You can use our [URL generator](https://aqueous-hollows-34193.herokuapp.com/static/urlgenerator.html) to get the general URL to post to, before the last three parameters (`userID`, `tree`, and `functionName`).

### Constant point-assignment point-assignment method (`const`)
- Additional parameters
  - `default_task_value`: Constant value of points to be assigned to each task.


```
URL example: http://127.0.0.1:6789/api/constant/basic/30/14/inf/0/3000/0/60/t/2/10/tree/user123/getTasksForToday

<method>: constant
<scheduler>: basic
default_time_est: 30
default_deadline: 14
allowed_task_time: inf
min_sum_of_goal_values: 0 
max_sum_of_goal_values: 3000
min_goal_value_per_goal_duration: 0
max_goal_value_per_goal_duration: 60
points_per_hour: t
rounding: 2
default_task_value: 10
<userID>: user123
<functionName>: getTasksForToday
```

<!--
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
-->

### Length heuristics point-assignment method (`length`)
- There are no additional parameters for this method.

```
URL example: http://127.0.0.1:6789/api/length/deadline/30/14/inf/0/inf/0/inf/true/0/tree/user123/getTasksForToday

<method>: length
<scheduler>: deadline
default_time_est: 30
default_deadline: 14
allowed_task_time: inf
min_sum_of_goal_values: 0 
max_sum_of_goal_values: inf
min_goal_value_per_goal_duration: 0
max_goal_value_per_goal_duration: inf
points_per_hour: true
rounding: 0
<userID>: user123
<functionName>: getTasksForToday
```
    

### Random point-assignment method (`random`)
- Additional parameters
  - `distribution`: The name of the probability distribution (according to [NumPy](https://numpy.org/)) and their own parameters. So far, these distributions have been implemented:
    - `uniform`: Uniform distribution with parameters `low` (lower interval bound) and `high` (higher interval bound)
    - `normal`: Normal (Gaussian) distribution with parameters `loc` (mean value) and `scale` (standard deviation)

```
URL Example: http://127.0.0.1:6789/api/random/deadline/30/14/inf/0/10000/0/10/false/2/normal/1/100/tree/user123/getTasksForToday

<method>: random
<scheduler>: deadline
default_time_est: 30
default_deadline: 14
allowed_task_time: inf
min_sum_of_goal_values: 0 
max_sum_of_goal_values: 10000
min_goal_value_per_goal_duration: 0
max_goal_value_per_goal_duration: 10
points_per_hour: false
rounding: 2
distribution: normal
loc: 1
scale: 100
<userID>: user123
<functionName>: getTasksForToday
```

### SMDP point-assignment method (`smdp`)
- Additional parameters
    - `choice_mode`: Mode of making time transitions while executing optimal policy
        - `max`: Choose the path that is most-likely to happen.
        - `random`: Make a random choice w.r.t. probabilities assigned to time transitions.
    - `gamma`: Discount factor `float(0, 1)`
    - `loss_rate`: Unit-time value that models cognitive effort `float[0, inf)`
    - `num_bins`: Number of time transitions `int[1, inf)`
    - `planning_fallacy_const`: Value that scales time estimates `float(0, inf)`
    - `slack_reward`: Reward associated with slack-off actions `float[-inf, inf)`
    - `unit_penalty`: Unit-time value that penalizes `float[0, inf]`
    - `goal_pr_loc`: Bias value for goal pseudo-rewards `float(-inf, inf)`
    - `goal_pr_scale`: Scaling value for goal pseudo-rewards `float(-inf, inf)`
    - `task_pr_loc`: Bias value for task pseudo-rewards `float(-inf, inf)`
    - `task_pr_scale`: Scaling value for task pseudo-rewards `float(-inf, inf)`
    - `scale_type` (optional): It represents the method by which points are scaled. If no scaling to be used, the inputting this parameter (and the `scale_min` and `scale_max` parameters) should be omitted.
        - `no_scaling`: Points are assigned according their pseudo-rewards (no change). 
        - `min_max`: Points are assigned according to this formula `task_reward = (task_reward - min_value) / (max_value - min_value) * (scale_max - scale_min) + scale_min`.
        - `mean_value`: Points are assigned according to this formula `task_reward = (task_reward - mean_reward) / (max_value - min_value) * (scale_max - scale_min) / 2 + ((scale_max + scale_min) / 2)`
    - `scale_min` (optional): Lower interval bound. If `inf`, then the lower interval bound is not set.
    - `scale_max` (optional): Upper interval bound. If `inf`, then the upper interval bound is not set.

Notation:
- `[lower_bound, upper_bound]`: closed interval
- `(lower_bound, upper_bound]`: half-open interval
- `[lower_bound, upper_bound)`: half-open interval
- `(lower_bound, upper_bound)`: open interval

```
URL example: http://127.0.0.1:6789/api/smdp/mdp/30/14/inf/0/inf/0/inf/false/2/max/0.9999/1/2/1.39/0/0/0/1/0/1/min_max/1/2/tree/u123/getTasksForToday

<method>: random
<scheduler>: deadline
default_time_est: 30
default_deadline: 14
allowed_task_time: inf
min_sum_of_goal_values: 0 
max_sum_of_goal_values: inf
min_goal_value_per_goal_duration: 0
max_goal_value_per_goal_duration: inf
points_per_hour: false
rounding: 2
choice_mode: max
gamma: 0.9999
loss_rate: 1
num_bins: 2
planning_fallacy_const: 1.39
slack_reward: 0
unit_penalty: 0
goal_pr_loc: 0
goal_pr_scale: 1
task_pr_locL 0
task_pr_scale: 1
scale_type: min_max
scale_min: 1
scale_max: 2
<userID>: user123
<functionName>: getTasksForToday
```
  
## Potential issues

If you encounter any problem related to the API, please submit a Github issue.

  
## Required Python Packages

All required Python packages are listed in the `requirements.txt` file.

## Citation

If you use this code in academic work, please cite the report:

\<TBD\>

## Acknowledgements

This project uses code from:
- [The Pacman Projects](http://ai.berkeley.edu)
- [todolistMDP by Andrew Tan](https://github.com/andrewztan/todolistMDP)
